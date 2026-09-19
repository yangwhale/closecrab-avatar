"""音频进模型那一侧 —— 几何 + 缓冲。**不依赖 torch、不依赖 asyncio。**

## 这个文件曾经有 200 行，现在只剩这么点

原来这里有一整套「什么时候攒够一轮音频可以开工」的调度算术：
`slot_frame` / `frames_needed_for_repeat` / `repeats_ready` / `gather_slots`…
跟上游 `get_audio_embed_bucket_fps` 逐点对拍过，测试也齐。

**然后发现它全是多余的。**

上游的流式 pipeline 自己按块拉音频、自己编码
（`_streaming_encode_next_audio_block_or_random` → `self.get_audio_callback()`），
调用方**只需要回答一个问题：一块是多少个采样**。那套调度是我在不知道
有这个回调时自己搭的第二套实现。

留个记号是因为这类错误会重犯：**在自己实现一套调度之前，先确认被调用方
是不是已经自己调度了。** 判据很简单 —— 看它向你「要」什么。它要整段
（`audio_path`），你才需要切；它要下一块（callback），切的是它。

（旧实现在 git 历史里，`git log -- worker/audio_stream.py`。）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("closecrab.avatar.audio")


@dataclass(frozen=True)
class BlockGeometry:
    """一块的大小。

    ⚠️ **默认值只是给不带模型的测试用的。** 真跑一律走 `from_model_config()`
    从模型配置读 —— 这两个常数上游有**两棵配置树**（`wan_base` 和 `wan_2_2`），
    同名 `WAN_CONFIGS` 都存在、import 都不报错，拿错那棵 `sample_fps`
    就从 25 变成 16。上次就是这么错的，靠产物 mp4 时长对不上才揪出来。
    """

    fps: int = 25
    frames_per_block: int = 12
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        # 这几个值错了不会报错，只会让口型跟声音差一截 —— 宁可起不来。
        if self.fps <= 0 or self.frames_per_block <= 0 or self.sample_rate <= 0:
            raise ValueError("fps / frames_per_block / sample_rate 都必须为正")

    @classmethod
    def from_model_config(cls, cfg, *, sample_rate: int = 16000) -> "BlockGeometry":
        """从 `WAN_CONFIGS[task]` 读，不写死。

        `frames_per_block = num_frames_per_block × 4` —— 那个 4 是 VAE 的时间
        下采样率。上游在
        `_streaming_encode_next_audio_block_or_random(block_frames=
        self.num_frames_per_block * 4)` 里就是这么算的，照抄它，别自己推。
        """
        return cls(fps=int(cfg.sample_fps),
                   frames_per_block=int(cfg.num_frames_per_block) * 4,
                   sample_rate=sample_rate)

    @property
    def block_seconds(self) -> float:
        """一块对应多少秒视频。12 ÷ 25 = 0.48 s。"""
        return self.frames_per_block / self.fps

    @property
    def block_samples(self) -> int:
        """回调该返回多少个采样。

        ⚠️ 这里用**音频采样率**（16 kHz），不是音频嵌入帧率（30 Hz），
        也不是视频帧率。三个率长得都像「帧率」，拿错了差几个数量级，
        而现象只是「怎么一直不出帧」。
        """
        return int(self.sample_rate * self.block_seconds)


def resample_i16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """int16 单声道重采样。

    整数倍降采样（48k→16k 正好 3:1）走**先平均再抽**，那个平均就是一个最
    简单的低通 —— 直接抽点会把高频折回来变成嘶声，而 wav2vec 对这个敏感。
    非整数倍退回线性插值：够用，也不值得为它引一个 scipy。
    """
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    if src_rate % dst_rate == 0:
        k = src_rate // dst_rate
        n = (pcm.size // k) * k
        if n == 0:
            return pcm[:0]
        return pcm[:n].astype(np.int32).reshape(-1, k).mean(axis=1).astype(np.int16)
    out_n = max(1, int(round(pcm.size * dst_rate / src_rate)))
    idx = np.linspace(0, pcm.size - 1, out_n)
    return np.interp(idx, np.arange(pcm.size), pcm.astype(np.float32)).astype(np.int16)


class PcmInbox:
    """模型来拉音频的地方。**一端事件循环写，一端模型线程读。**

    ## ⚠️ 为什么必须加锁

    `push()` 在 asyncio 事件循环里（LiveKit 推音频进来），`pull_block()` 在
    **模型线程**里（torch 那一坨是同步阻塞的）。两个真线程同时动一个列表。

    `append` / `pop(0)` 各自原子，但 `self._buf[0] = head[take:]` 跟 `pop(0)`
    之间不原子 —— 会丢样本或重复。症状是口型偶尔跟声音错开，**不报错**。

    ## ⭐ 放行的判据是「攒够一整块」，不是「有没有货」

    这条踩过。原来写的是「有一个采样就返回，剩下补静音」，看着挺合理 ——
    直到实测发现三次会话一共 53 秒音频却生成了 210 块（该有 110 块）。

    原因：LiveKit **按 100 ms 一包**推音频，而一块是 480 ms。「有货就放行」
    意味着每来一个小包就触发一整块生成，其中四分之三是补出来的静音。
    生成速率冲到需求的四五倍，出帧队列被灌满、开始丢最旧的帧 ——
    **表现出来是帧率越跑越低**，看起来像 GPU 不够，其实是喂法不对。

    所以攒够 `block_samples` 才放行。句尾那点不足一块的，靠
    `mark_segment_end()` 明确放行 —— LiveKit 协议里本来就有 `AudioSegmentEnd`
    这个信号，它的意思正是「这句说完了」。

    ## 没货的时候会**阻塞**，这也是故意的

    换成「没货就立刻返回静音」的话，整条五卡流水线会以 1.66× 实时的速度
    空转生成闭着嘴的画面 —— 五张 B200 全程满载，没人说话也烧。

    阻塞在这里，上游 DiT rank 自然堵在 `dist.recv` 上，**整组一起停**，
    不需要任何额外的暂停/恢复协调。来音频了回调返回，整组自己接着跑。
    """

    def __init__(self, geom: BlockGeometry):
        self._geom = geom
        self._buf: list[np.ndarray] = []
        self._pending = 0
        self._cv = threading.Condition()
        self._draining = False      # 句子说完了，剩下这点尾巴也要放行
        self._closed = False
        self._was_fed = False
        self.on_drained = None
        """管线抽干时回调一次（在 `pull_block` 真要阻塞之前）。

        **这是整条链上唯一一个能免费拿到「对应关系归零」的地方** ——
        不需要序号、不需要知道流水线多深。"""

    # ── 事件循环那一侧 ──────────────────────────────────────────────

    def push(self, pcm_i16: np.ndarray, *, src_rate: int | None = None) -> None:
        """喂一段 PCM。**给了 `src_rate` 就自动重采样到模型要的那个率。**

        ⚠️ 这一步不能省。房间里的音频是发送方定的率（我们这条是 48 kHz），
        而模型那侧的 wav2vec **要 16 kHz**。不转就是把 48 k 的数据按 16 k 解读 ——
        块长算出来只有实际的三分之一，口型跟声音差三倍，**而且不报错**。
        """
        if src_rate and src_rate != self._geom.sample_rate:
            pcm_i16 = resample_i16(pcm_i16, src_rate, self._geom.sample_rate)
        with self._cv:
            self._buf.append(pcm_i16)
            self._pending += len(pcm_i16)
            self._cv.notify_all()

    def mark_segment_end(self) -> None:
        """这句说完了 —— 不足一块的尾巴也放行。

        ⚠️ **不是打断**，不清缓冲。混为一谈的话每句话结尾都被吞掉一截，
        听起来像「他话没说完」。
        """
        with self._cv:
            self._draining = True
            self._cv.notify_all()

    def clear(self) -> None:
        """打断。**把没念的全扔了。**

        人已经不说话了而屏幕上的嘴还在动，恐怖谷一下就掉进去。
        """
        with self._cv:
            self._buf.clear()
            self._pending = 0
            self._draining = False
            self._cv.notify_all()

    def close(self) -> None:
        """进程要退了。把阻塞中的模型线程放出来，否则 join 不回来。"""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_samples(self) -> int:
        with self._cv:
            return self._pending

    # ── 模型线程那一侧 ──────────────────────────────────────────────

    def _ready(self) -> bool:
        return (self._closed
                or self._pending >= self._geom.block_samples
                or (self._draining and self._pending > 0))

    def pull_block(self, timeout: float | None = None) -> np.ndarray:
        """模型每个 block 调一次。**恒定返回 `block_samples` 个 float32。**

        长度不固定的话模型侧的编码窗口会错位 —— 不报错，只是口型差一截。

        放行条件（见类文档）：攒够一整块，或者 `mark_segment_end()` 说这句
        完了。都不满足就**阻塞**。给了 `timeout` 且等超时则返回整块静音，
        留给「待机也要出画面」那种形态。
        """
        need = self._geom.block_samples
        out = np.zeros(need, dtype=np.float32)
        with self._cv:
            # ⚠️ **只在「从有货到没货」那一次跳变时报，不是每次没货都报。**
            #    「没货」在正常运行里很常见（模型偶尔跑在音频前面）。
            #    每次都报的话，回调里那句「清空出帧队列」会把好帧一路扔光
            #    —— 2026-09-19 实测：连着两场一帧都发不出去。
            #    抽干是个**边沿**，不是个状态。
            if not self._ready() and self._was_fed and self.on_drained is not None:
                self._was_fed = False
                # ⭐ **管线空了的那一刻，就是这里。**
                #    模型回头来要下一块、而我们没货 —— 说明它已经把在途的
                #    全吐出来了。这是唯一一个「不用猜流水线多深」就能确定
                #    对应关系的时刻：从下一块起，出来的帧必定是那一块的。
                #
                #    > Chris 2026-09-19：「你把音频怼进去然后等着，不管等
                #    > 多久，出来的第一帧准是你自己的帧。」
                #
                #    回调在持锁状态下调用，**里面别再回头碰 inbox**，会死锁。
                try:
                    self.on_drained()
                except Exception:                       # noqa: BLE001
                    log.warning("on_drained 回调出错，忽略", exc_info=True)
            self._cv.wait_for(self._ready, timeout)

            filled = 0
            while filled < need and self._buf:
                head = self._buf[0]
                take = min(need - filled, len(head))
                # int16 → float32 [-1,1]，wav2vec 的 processor 要的就是这个量纲。
                # 少了这一步它把整数当振幅，特征完全不对 —— 照样出视频，
                # 只是口型对不上。
                out[filled:filled + take] = head[:take].astype(np.float32) / 32768.0
                filled += take
                self._was_fed = True
                self._pending -= take
                if take == len(head):
                    self._buf.pop(0)
                else:
                    self._buf[0] = head[take:]
            if not self._buf:
                # 尾巴放完了，回到「攒够才放行」。不清的话下一轮又会被
                # 一个 100 ms 的小包触发整块生成。
                self._draining = False
        return out
