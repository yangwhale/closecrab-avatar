"""把 LiveAvatar 接成 LiveKit 的 `VideoGenerator`。

协议只有三个方法（`livekit/agents/voice/avatar/_types.py`）：

    push_audio(frame | AudioSegmentEnd)   喂音频
    clear_buffer()                        打断时立刻停
    __aiter__()                           持续吐出视频帧和音频帧

## 分工：**模型自己拉音频，我们只负责有货可拉**

上游流式 pipeline 每个 block 调一次 `self.get_audio_callback()` 要下一块 PCM
（`_streaming_encode_next_audio_block_or_random`）。所以这一层不做任何
「攒够多少才能开工」的调度 —— 那是模型自己的事。

我们只做三件事：

  1. 把 LiveKit 推来的 PCM 攒进缓冲
  2. 模型来要的时候给它**恰好一块**（没货就给静音）
  3. 把模型吐出来的帧转成 `rtc.VideoFrame` 发出去

> 早期版本在这里自己实现了一整套分块调度（还跟上游
> `get_audio_embed_bucket_fps` 逐点对拍过）。**全是多余的** ——
> 在自己写调度之前，先看被调用方向你「要」什么：要整段才需要你切，
> 要下一块就说明切的是它。见 `audio_stream.py` 顶部那段记号。

## ⚠️ clear_buffer 是三个方法里最要命的

人已经不说话了而屏幕上的嘴还在动，恐怖谷一下就掉进去。被打断时
**在途的所有东西都要丢**。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator, Protocol, Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

from .audio_stream import BlockGeometry

log = logging.getLogger("closecrab.avatar.generator")

AVOut = Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


class FrameSource(Protocol):
    """模型那一侧。**只有三个方法** —— 多一个都说明编排漏到模型里去了。"""

    @property
    def size(self) -> tuple[int, int]:
        """(宽, 高)。"""
        ...

    def start(self, audio_cb, out: asyncio.Queue) -> None:
        """开始生成。

        - `audio_cb`：无参可调用，返回一块 PCM（`float32`，长度**恒为**
          `BlockGeometry.block_samples`）。模型每个 block 调一次。
        - `out`：生成出来的 RGBA 帧（`HxWx4`）往这里塞。
          **模型跑在别的线程里**，实现方要用 `loop.call_soon_threadsafe`。
        """
        ...

    def reset(self) -> None:
        """丢掉在途状态。被打断时调。"""
        ...


class LiveAvatarGenerator(VideoGenerator):
    """真数字人。

    音频从 `push_audio` 进来攒着，模型自己来拉；渲出来的帧按视频帧率吐出去。
    原音频**原样**跟着一起吐 —— 数字人只负责脸，声音还是 TTS 那一路的。
    """

    def __init__(self, source: FrameSource, *, geom: BlockGeometry | None = None):
        self._src = source
        self._geom = geom or BlockGeometry()

        # PCM 缓冲。用 `deque` 是因为两头都要动：尾部进、头部出。
        self._pcm: deque[np.ndarray] = deque()

        self._out: asyncio.Queue[rtc.VideoFrame] = asyncio.Queue()
        self._audio_out: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._started = False

    @property
    def size(self) -> tuple[int, int]:
        return self._src.size

    # ── 进 ────────────────────────────────────────────────────────

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            # ⚠️ 段落结束**不清缓冲** —— 剩下那点不足一块的音频还要说完。
            #    真正该清的只有打断（`clear_buffer`）。混为一谈的话每句话
            #    结尾都会被吞掉一截，而且听起来像「他话没说完」。
            await self._audio_out.put(frame)
            return

        # 音频原样转发，**不等视频**。首帧压不下去，卡着等它整句话就晚一拍；
        # 声音先到一点点反而没人察觉。
        await self._audio_out.put(frame)

        self._pcm.append(np.frombuffer(frame.data, dtype=np.int16))
        self._ensure_started()

    def clear_buffer(self) -> None:
        """被打断。**把所有在途的东西一次丢干净。**

        丢四样，少一样都会漏帧：
          1. 还没被模型拉走的 PCM
          2. 已经生成、还没发出去的帧
          3. 已经排队、还没发出去的音频
          4. 模型侧的在途状态（`source.reset()`）
        """
        self._pcm.clear()
        for q in (self._out, self._audio_out):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self._src.reset()

    # ── 模型来拉音频 ───────────────────────────────────────────────

    def _pull_block(self) -> np.ndarray:
        """模型每个 block 调一次。**必须恒定返回 `block_samples` 个采样。**

        长度不固定的话模型侧的编码窗口会错位 —— 而它不会报错，
        只会让口型跟声音差一截。

        没货就给静音。静音对应「人在那儿但没说话」，**不是错误状态** ——
        所以这里一个字都不打日志，否则空闲时一秒刷两行。
        """
        need = self._geom.block_samples
        out = np.zeros(need, dtype=np.float32)
        filled = 0
        while filled < need and self._pcm:
            head = self._pcm[0]
            take = min(need - filled, len(head))
            # int16 → float32 [-1, 1]，wav2vec 要的就是这个量纲。
            # 少了这一步 processor 会把整数当成振幅，特征完全不对。
            out[filled:filled + take] = head[:take].astype(np.float32) / 32768.0
            filled += take
            if take == len(head):
                self._pcm.popleft()
            else:
                self._pcm[0] = head[take:]
        return out

    def _ensure_started(self) -> None:
        """第一次有音频时才启动模型 —— 没人说话就不占卡。"""
        if self._started:
            return
        self._started = True
        self._src.start(self._pull_block, self._out)

    # ── 出 ────────────────────────────────────────────────────────

    async def __aiter__(self) -> AsyncIterator[AVOut]:
        interval = 1.0 / self._geom.fps
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            # 音频优先、有多少发多少：它不能等视频。
            while not self._audio_out.empty():
                yield self._audio_out.get_nowait()
            if not self._out.empty():
                yield self._out.get_nowait()
            next_at += interval
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_at = loop.time()      # 落后就重新对齐，不追债


def to_video_frame(img: np.ndarray) -> rtc.VideoFrame:
    """RGBA `HxWx4` → LiveKit 帧。给 `FrameSource` 的实现方用。"""
    return rtc.VideoFrame(width=img.shape[1], height=img.shape[0],
                          type=rtc.VideoBufferType.RGBA, data=img.tobytes())
