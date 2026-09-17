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

import threading
from dataclasses import dataclass

import numpy as np


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


class PcmInbox:
    """模型来拉音频的地方。**一端事件循环写，一端模型线程读。**

    ## ⚠️ 为什么必须加锁

    `push()` 在 asyncio 事件循环里（LiveKit 推音频进来），`pull_block()` 在
    **模型线程**里（torch 那一坨是同步阻塞的）。两个真线程同时动一个列表。

    `append` / `pop(0)` 各自原子，但 `self._buf[0] = head[take:]` 跟 `pop(0)`
    之间不原子 —— 会丢样本或重复。症状是口型偶尔跟声音错开，**不报错**。

    ## 没货的时候会**阻塞**，这也是故意的

    换成「没货就立刻返回静音」的话，整条五卡流水线会以 1.66× 实时的速度
    空转生成闭着嘴的画面 —— 五张 B200 全程满载，没人说话也烧。

    阻塞在这里，上游 DiT rank 自然堵在 `dist.recv` 上，**整组一起停**，
    不需要任何额外的暂停/恢复协调。来音频了回调返回，整组自己接着跑。
    """

    def __init__(self, geom: BlockGeometry):
        self._geom = geom
        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._has_data = threading.Event()
        self._closed = False

    # ── 事件循环那一侧 ──────────────────────────────────────────────

    def push(self, pcm_i16: np.ndarray) -> None:
        with self._lock:
            self._buf.append(pcm_i16)
            self._has_data.set()

    def clear(self) -> None:
        """打断。**把没念的全扔了。**

        人已经不说话了而屏幕上的嘴还在动，恐怖谷一下就掉进去。
        """
        with self._lock:
            self._buf.clear()
            if not self._closed:
                self._has_data.clear()

    def close(self) -> None:
        """进程要退了。把阻塞中的模型线程放出来，否则 join 不回来。"""
        with self._lock:
            self._closed = True
            self._has_data.set()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_samples(self) -> int:
        with self._lock:
            return sum(len(x) for x in self._buf)

    # ── 模型线程那一侧 ──────────────────────────────────────────────

    def pull_block(self, timeout: float | None = None) -> np.ndarray:
        """模型每个 block 调一次。**恒定返回 `block_samples` 个 float32。**

        长度不固定的话模型侧的编码窗口会错位 —— 不报错，只是口型差一截。

        - 一个采样都没有 → **阻塞等**（见类文档，这是省电那一环）
        - 有但不够一块 → 补静音凑满。这是句尾的正常形态，
          补出来的静音正好让嘴闭上
        - 给了 `timeout` 且等超时 → 返回整块静音（留给「待机也要出画面」那种形态）
        """
        self._has_data.wait(timeout)

        need = self._geom.block_samples
        out = np.zeros(need, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < need and self._buf:
                head = self._buf[0]
                take = min(need - filled, len(head))
                # int16 → float32 [-1,1]，wav2vec 的 processor 要的就是这个量纲。
                # 少了这一步它把整数当振幅，特征完全不对 —— 照样出视频，
                # 只是口型对不上。
                out[filled:filled + take] = head[:take].astype(np.float32) / 32768.0
                filled += take
                if take == len(head):
                    self._buf.pop(0)
                else:
                    self._buf[0] = head[take:]
            if not self._buf and not self._closed:
                self._has_data.clear()
        return out
