"""流式音频 → 音频嵌入。**让它跟离线走同一条时间轴。**

## 为什么这个文件必须存在

上游的流式路径是这么拿音频特征的（`causal_s2v_pipeline_tpp_blockwise.py`）：

    chunk = self.get_audio_callback()                    # 正好一块 = 0.48 s
    audio_embed, _ = self.encode_audio_from_array(chunk, infer_frames=12)
    return audio_embed[..., :12]

**每一块自己单独过一遍 wav2vec。** 看起来天经地义，实际上三处都错，而且
三处都不报错、都只表现为「口型对不上」。2026-09-19 用真 torch 逐条量过：

| | 流式（每块 0.48 s 单独编码） | 离线（整段一次编码） |
|---|---|---|
| 嵌入帧间距 | **36.667 ms** | 33.410 ms |
| 时间轴 | **拉伸 1.100×** | 1.0023× |
| 每 12 帧补零 | **1 帧**（第 12 帧索引越界） | 0（只在整段末尾） |
| 响度归一化 | 每 0.48 s 重算一次 | 整段一次 |

### ① 时间轴被拉伸 10%

`linear_interpolation` 用的是 `align_corners=True`，并且
`output_len = int(seq_len * 30)` —— **截断**。

7680 个采样 → wav2vec 出 23 帧（不是 24：卷积感受野 400 吃掉了边上那点）
→ `int(23/50*30) = int(13.8) = 13` 帧。而 `align_corners=True` 要把 23 帧
**铺满** 这 13 个位置，于是每帧实际跨 22/12 = 1.8333 个 wav2vec 帧
= 36.667 ms，而下游 `get_audio_embed_bucket_fps` 按 33.333 ms 读。

结果是块内口型误差从 +5.8 ms 爬到 +59.2 ms，**下一块归零重来** ——
一条周期 0.48 s 的锯齿波。整段总时长完全对得上（所以时长校验发现不了），
只是块内越走越快。

整段编码时同一个截断只造成 0.23% 误差（873 帧截成 523），可以忽略。
**误差跟块长成反比 —— 这就是为什么只有流式这条犯病。**

### ② 每 12 帧有 1 帧收到「静音」

同一个截断的第二个后果：桶索引算出 `[0,1,2,4,5,6,7,8,10,11,12,13]`，
而嵌入只有 13 帧（0–12）。第 12 个视频帧索引 13 越界，
上游那支 `else` 填 `torch.zeros` —— 模型被告知「这一帧没声音」。

25 fps 下就是 **2.08 Hz 的嘴部抽动**，看起来像掉帧，其实是喂了静音。

### ③ 响度归一化的窗口太短

`preprocessor_config.json` 里 `do_normalize: true` —— processor 对**每个
送进去的窗口**做零均值单位方差。0.48 s 一个窗口意味着：句中一个短停顿会被
放大到跟正常说话一样的量级，模型于是把嘴张开。窗口拉长到一两秒，
方差被真正的语音主导，这个病自然就好了。

## 修法：滑动窗口 + 只回看，不预看

三条全是「喂进去的音频太短」的后果，所以修法只有一条：
**每次编码一个长窗口，只取其中最后一块对应的那 12 帧。**

用**回看**不用预看 —— 过去的音频我们本来就有，不额外增加一毫秒延迟。

窗口长度不能随便取。要让插值出来的帧间距**正好**是 1/30 s，必须

    (T-1) / (out_len-1) == 50 / 30 == 5/3          # align_corners=True 的约束
    T = (W - 400) // 320 + 1                       # wav2vec 卷积栈

联立解出 `W = 400 + 1600k`，此时 `T = 5k+1`、`out_len = 3k+1`，
帧间距**精确** 1/30 s，第 j 帧对应窗口内 `0.0125 + j/30` 秒
（0.0125 = 400 样本感受野的中心）。

⚠️ **不要顺手把窗口取成整秒。** 16000 样本算出 T=49，(49-1)/5 不是整数，
   间距又变成 33.898 ms —— 看着「差不多」，一样是每块几十毫秒的锯齿。
   这个约束是本文件存在的理由，改窗口长度前先跑 `tests/test_audio_feat.py`。
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger("closecrab.avatar.audiofeat")

W2V_FIELD = 400
"""wav2vec2 卷积栈的感受野（样本）。七层卷积核 [10,3,3,3,3,2,2] 叠出来的。"""

W2V_STRIDE = 320
"""wav2vec2 卷积栈的总步长（样本）= 5×2×2×2×2×2×2。输出名义 16000/320 = 50 Hz。"""

W2V_FPS = 50


def w2v_frames(n_samples: int) -> int:
    """`n_samples` 个采样进 wav2vec2 出几帧。

    逐层 `(L-k)//s + 1` 化简的结果就是 `(n-400)//320 + 1`
    —— 已跟 `Wav2Vec2Config` 的真实 `conv_kernel/conv_stride` 对拍过
    （7680→23、16000→49、279680→873，三个都中）。
    """
    if n_samples < W2V_FIELD:
        return 0
    return (n_samples - W2V_FIELD) // W2V_STRIDE + 1


def window_grid(video_rate: int = 30) -> tuple[int, int]:
    """返回 `(样本步长, 嵌入帧步长)` —— 合法窗口长度只能是这个格子上的点。

    合法窗口 `W = W2V_FIELD + k × 样本步长`，对应 `out_len = 1 + k × 帧步长`。
    """
    from math import gcd
    g = gcd(W2V_FPS, video_rate)
    k_w2v = W2V_FPS // g                     # video_rate=30 → 5
    k_out = video_rate // g                  # video_rate=30 → 3
    return W2V_STRIDE * k_w2v, k_out


def choose_window(min_samples: int, video_rate: int = 30) -> tuple[int, int]:
    """选一个 **≥ `min_samples`** 的合法窗口。返回 `(窗口样本数, 嵌入帧数)`。

    「合法」＝ 插值出来的帧间距正好 1/`video_rate` 秒（见模块文档的联立）。
    """
    step_w, step_o = window_grid(video_rate)
    k = max(1, -(-(min_samples - W2V_FIELD) // step_w))   # 向上取整
    return W2V_FIELD + k * step_w, 1 + k * step_o


def block_indices(window_samples: int, block_samples: int, block_frames: int,
                  *, fps: int, sample_rate: int = 16000,
                  video_rate: int = 30) -> list[int]:
    """窗口末尾那一块的第 i 个视频帧，该取第几个嵌入帧。

    块钉在窗口**末尾**（回看在前）。第 i 帧在窗口内的时刻是

        t_i = (W - block_samples)/sr + i/fps

    ## ⚠️ 不要补感受野中心那 12.5 ms

    物理上第 j 个嵌入帧的中心在 `W2V_FIELD/2/sr + j/video_rate`
    （= +12.5 ms），加上它「更准」。**但离线不加。** 上游
    `get_sample_indices` 把第 g 个视频帧直接映到 `round(1.2g)`，
    等于认定「第 j 帧就在 j/30 秒」—— 模型是在这个约定下训练的。

    实测（真录音 30 s / 720 帧，跟整段离线编码比余弦）：

        补 t0    均值 0.714
        不补     均值 0.792     ← 差 0.078

    **目标是追平离线，不是追平物理。** 这类「老约束在新语境下已经反了」的
    地方，判据只能是实测，不能是推理。

    取整用 `round`，跟上游一致；两边都带 ±1/(2·video_rate) 的量化，
    那是训练时就有的，不去修它。
    """
    base = (window_samples - block_samples) / sample_rate
    return [int(round((base + i / fps) * video_rate))
            for i in range(block_frames)]


def _float_env(name: str, default: float) -> float:
    """浮点旋钮。读不出来就用默认值，绝不抛 —— 一个手滑的值不该让会话起不来。"""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r 不是数，用默认值 %s", name, raw, default)
        return default


def _lookback_seconds() -> float:
    raw = (os.environ.get("CCA_AUDIO_LOOKBACK_S") or "").strip()
    try:
        return max(0.0, float(raw)) if raw else 1.5
    except ValueError:
        log.warning("CCA_AUDIO_LOOKBACK_S=%r 不是数字，回落 1.5", raw)
        return 1.5


class StreamingAudioFeat:
    """把「下一块音频」编码成上游要的 `[1, L, D, block_frames]`。

    **装在 pipeline 实例上替掉上游那个方法**，不改上游源码：

        feat = StreamingAudioFeat(pipe.audio_encoder, pull=inbox.pull_block, ...)
        pipe._streaming_encode_next_audio_block_or_random = feat.next_block

    上游那行是 `self._streaming_encode_next_audio_block_or_random(block_frames=…)`，
    实例属性会遮住类方法，所以替换生效。
    """

    def __init__(self, audio_encoder, *, pull, block_samples: int, fps: int,
                 device=None, dtype=None, sample_rate: int = 16000,
                 lookback_s: float | None = None, fallback=None):
        self._enc = audio_encoder
        self._pull = pull
        self._fallback = fallback
        """`fallback(chunk, block_frames)` —— 我们这条炸了的时候顶上。

        ⚠️ **必须接收已经取出来的那一块**，不能是「再调一次上游那个方法」——
        上游那个会自己再 `get_audio_callback()` 拉一块，于是每次回退都吃掉
        两块音频。持续回退的话画面会跑成两倍速，比口型不准难查得多。
        """
        self._block_samples = int(block_samples)
        self._fps = int(fps)
        self._sr = int(sample_rate)
        self._device = device
        self._dtype = dtype
        self._video_rate = int(getattr(audio_encoder, "video_rate", 30))

        look = _lookback_seconds() if lookback_s is None else max(0.0, lookback_s)
        need = self._block_samples + int(look * self._sr)
        self._win, self._out_len = choose_window(need, self._video_rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._pending_reset = False
        self._warned_clip = False
        log.info("音频特征窗口 %d 采样（%.3f s，回看 %.3f s）→ %d 个嵌入帧，"
                 "帧间距精确 %.4f ms",
                 self._win, self._win / self._sr,
                 (self._win - self._block_samples) / self._sr,
                 self._out_len, 1000 / self._video_rate)

    def reset(self) -> None:
        """被打断了：回看缓冲作废。

        不清的话下一句的第一块会带着上一句的尾巴当上下文 —— wav2vec 是带
        自注意力的，那段上下文会实实在在改变特征值。

        ⚠️ **只置标志，不直接清。** 这个方法在事件循环线程上调，而
        `next_block` 在模型线程上，此刻多半正阻塞在 `pull()` 里。直接清的话
        它醒来后那一句 `self._buf = concat(self._buf, chunk)` 会把清空覆盖掉
        —— 而它读到的 `self._buf` 是清空**之前**的。置标志则由模型线程自己
        在 `pull()` 返回之后执行，顺序天然正确。
        """
        with self._lock:
            self._pending_reset = True

    # ── 上游调的就是这一个 ────────────────────────────────────────

    def next_block(self, block_frames: int):
        """⚠️ **绝不能让异常跑出这个函数。**

        它跑在 VAE rank 上，算完要 `dist.send` 给四个 DiT rank。抛出去的话
        那四个永远堵在 `dist.recv`，**整组五张卡一起死** —— 比「这一块口型
        不准」糟得多。所以裹一层，失败就用上游的老办法把这一块顶过去。
        """
        chunk = np.asarray(self._pull(), dtype=np.float32).reshape(-1)
        try:
            return self._encode_block(chunk, block_frames)
        except Exception:                                # noqa: BLE001
            # **每块都记**，不去重：走上这条路是持续的口型退化，
            # 压成一条会让人以为是偶发。
            log.warning("滑窗音频编码失败，这一块回退老实现（口型会差一截）",
                        exc_info=True)
            if self._fallback is None:
                raise
            return self._fallback(chunk, block_frames)

    def _encode_block(self, chunk: np.ndarray, block_frames: int):
        import torch

        if chunk.size != self._block_samples and not self._warned_clip:
            # 不致命（下面按实际长度对齐），但说明几何算错了，值得知道。
            log.warning("回调给了 %d 个采样，期望 %d —— 块几何可能不一致",
                        chunk.size, self._block_samples)
            self._warned_clip = True

        with self._lock:
            if self._pending_reset:
                self._pending_reset = False
                self._buf = np.zeros(0, dtype=np.float32)
            self._buf = np.concatenate([self._buf, chunk])[-self._win:]
            pad = self._win - self._buf.size
            if pad > 0:
                # 会话刚开始，回看还没攒够。**在前面补零**，块仍然钉在末尾，
                # 索引因此不用分两套。前一两块的上下文偏弱，仅此而已。
                window = np.concatenate([np.zeros(pad, dtype=np.float32), self._buf])
            else:
                window = self._buf.copy()

        z = self._encode(window, torch)                    # [L, out_len, D]
        idx = block_indices(self._win, chunk.size or self._block_samples,
                            block_frames, fps=self._fps, sample_rate=self._sr,
                            video_rate=self._video_rate)
        hi = z.shape[1] - 1
        clipped = [i for i in idx if i < 0 or i > hi]
        if clipped and not self._warned_clip:
            log.warning("嵌入索引越界 %s（可用 0..%d）—— 窗口选小了", clipped, hi)
            self._warned_clip = True
        sel = torch.tensor([min(max(i, 0), hi) for i in idx],
                           device=z.device, dtype=torch.long)

        # [L, out_len, D] --取那 n 帧--> [L, n, D] → [L, D, n] → [1, L, D, n]
        # 最后这个形状是上游的硬契约：`audio_template` 就是
        # `torch.Size([1, 25, 1024, 12])`，DiT rank 靠 `empty_like` 它来收包。
        out = z.index_select(1, sel).permute(0, 2, 1).unsqueeze(0)
        return out.to(self._device, self._dtype).contiguous()

    # ── 内部 ──────────────────────────────────────────────────────

    def _encode(self, window: np.ndarray, torch):
        """跑 wav2vec，并**显式指定 output_len** —— 上游那一步正是没指定才出错。"""
        from liveavatar.models.wan.wan_2_2.modules.s2v.audio_encoder import (
            linear_interpolation)

        enc = self._enc
        values = enc.processor(window, sampling_rate=self._sr,
                               return_tensors="pt").input_values
        res = enc.model(values.to(device=enc.model.device, dtype=enc.model.dtype),
                        output_hidden_states=True)
        feat = torch.cat(res.hidden_states)                # [L, T, D]
        # ⭐ 传 output_len。不传的话 `int(T/50*30)` 截断，帧间距就不是 1/30 了。
        return linear_interpolation(feat.float(), input_fps=W2V_FPS,
                                    output_fps=self._video_rate,
                                    output_len=self._out_len)


class UtteranceAudioFeat:
    """**整句到齐再编码**，然后按块切片喂给模型。

    跟 `StreamingAudioFeat` 的区别只有一句话：那个是「边来边翻」，
    这个是「听完整句再翻」。为什么要换，见 `PcmInbox.pull_utterance`
    的那张余弦表 —— 一句话概括：因果编码的天花板是 0.81，够不着。

    编码这一步**原封不动用上游离线那两步**：

        extract_audio_feat_from_array(...)  →  get_audio_embed_bucket_fps(...)

    ⚠️ **不要在这里重新实现帧映射。** 我们上一版自己算了窗口和索引
    （`choose_window` / `block_indices`），每一处都有理有据，加起来还是
    对不上 —— 因为对不对不是推出来的，是那两个函数定义的。既然离线那条
    已经被确认 100% 对，就照抄它的调用，不要照抄它的道理。
    """

    def __init__(self, audio_encoder, *, pull_utterance, fps: int,
                 device=None, dtype=None, sample_rate: int = 16000,
                 max_utterance_s: float | None = None, fallback=None,
                 on_block=None):
        self._enc = audio_encoder
        self._pull_utt = pull_utterance
        self._fps = int(fps)
        self._sr = int(sample_rate)
        self._device = device
        self._dtype = dtype
        self._fallback = fallback
        self._on_block = on_block
        """`on_block(is_real: bool)` —— 每serve 一块就报一次，真音频还是空转。

        下游靠这个把空转块生成的画面扔掉。**不报的话那些画面会照发** ——
        实测 12 s 音频出 57.8 s 视频，音频跟着空转的画面跑，等于又不同步。
        """
        self._max_s = (_float_env("CCA_UTTERANCE_MAX_S", 15.0)
                       if max_utterance_s is None else float(max_utterance_s))
        self._idle_timeout = _float_env("CCA_FEAT_IDLE_TIMEOUT_S", 0.2)
        """没等到下一句就先喂静音、继续转圈的等待上限。**不是调优旋钮，是防死锁。**"""
        self._z = None          # 这句话的全部特征 [T, L, D]
        self._cursor = 0
        self._lock = threading.Lock()
        self._pending_reset = False
        log.info("音频特征：**整句编码**模式（上限 %.1f s）—— "
                 "开口前要等这句音频到齐，换特征跟离线一致", self._max_s)

    def reset(self) -> None:
        """被打断：这句话作废。理由同 `StreamingAudioFeat.reset` —— 只置标志。"""
        with self._lock:
            self._pending_reset = True

    def next_block(self, block_frames: int):
        """⚠️ **绝不能让异常跑出这个函数。**

        它跑在 VAE rank 上，算完要 `dist.send` 给四个 DiT rank。抛出去的话
        那四个永远堵在 `dist.recv`，五张卡一起死。
        """
        try:
            return self._next(block_frames)
        except Exception:                                # noqa: BLE001
            log.warning("整句编码失败，这一块回退上游老实现（口型会差一截）",
                        exc_info=True)
            if self._fallback is None:
                raise
            return self._fallback(np.zeros(0, dtype=np.float32), block_frames)

    def _next(self, block_frames: int):
        import torch

        with self._lock:
            if self._pending_reset:
                self._pending_reset = False
                self._z, self._cursor = None, 0

        if self._z is None or self._cursor >= self._z.shape[0]:
            # ⚠️ **带超时地要，不许干等。** 这个函数跑在 VAE rank 上，它每转
            #    一圈要给四个 DiT rank 发一块。停在这儿 ＝ 五张卡一起死锁。
            #    2026-09-19 第一版就是干等的，py-spy 打出来：VAE rank 停在
            #    pull_utterance，四个 DiT rank 停在 dist.recv。
            #
            #    旧的逐块实现之所以没这个毛病，是因为「等够 0.48 s 音频」
            #    这件事**顺带当了整条流水线的节拍器**。换成整句缓存之后
            #    那个节拍器没了，得自己补一个。
            got = self._pull_utt(self._max_s, self._idle_timeout)
            if got is None:
                return self._silence(block_frames, torch)   # 没人说话，嘴闭着
            pcm = np.asarray(got, dtype=np.float32).reshape(-1)
            if pcm.size == 0:
                return self._silence(block_frames, torch)   # 会话关了
            self._z = self._encode_utterance(pcm, block_frames, torch)
            self._cursor = 0
            log.info("整句编码完成：%.2f s 音频 → %d 帧特征（%d 块）",
                     pcm.size / self._sr, self._z.shape[0],
                     -(-self._z.shape[0] // block_frames))

        self._report(True)
        lo = self._cursor
        hi = min(lo + block_frames, self._z.shape[0])
        blk = self._z[lo:hi]
        self._cursor = hi
        if blk.shape[0] < block_frames:
            # 句尾不足一块：拿最后一帧补齐。补零会让嘴**突然闭死**，
            # 补最后一帧只是多定格一两帧，看起来自然得多。
            pad = blk[-1:].repeat(block_frames - blk.shape[0], 1, 1)
            blk = torch.cat([blk, pad], dim=0)

        # [n, L, D] → [L, D, n] → [1, L, D, n]，上游硬契约（audio_template）
        return blk.permute(1, 2, 0).unsqueeze(0).to(
            self._device, self._dtype).contiguous()

    def _encode_utterance(self, pcm: np.ndarray, block_frames: int, torch):
        """整段跑一次 —— **调的就是离线那两个函数**。"""
        enc = self._enc
        z = enc.extract_audio_feat_from_array(
            pcm, sample_rate=self._sr, return_all_layers=True, dtype=torch.float32)
        eb, _ = enc.get_audio_embed_bucket_fps(
            z, fps=self._fps, batch_frames=block_frames,
            m=getattr(enc, "audio_sample_m", 0))
        return eb.float()                                  # [T, L, D]

    def _report(self, is_real: bool) -> None:
        if self._on_block is None:
            return
        try:
            self._on_block(is_real)
        except Exception:                                # noqa: BLE001
            log.warning("on_block 回调出错，忽略", exc_info=True)

    def _silence(self, block_frames: int, torch):
        self._report(False)
        if self._z is not None and self._z.shape[0]:
            d = self._z.shape[1:]
        else:
            d = (25, 1024)
        blk = torch.zeros((block_frames, *d), dtype=torch.float32)
        return blk.permute(1, 2, 0).unsqueeze(0).to(
            self._device, self._dtype).contiguous()
