"""把 LiveAvatar 接成 LiveKit 的 `VideoGenerator`。

协议只有三个方法（`livekit/agents/voice/avatar/_types.py`）：

    push_audio(frame | AudioSegmentEnd)   喂音频
    clear_buffer()                        打断时立刻停
    __aiter__()                           持续吐出视频帧和音频帧

## 这一层薄得几乎没有东西，是对的

音频缓冲在 `PcmInbox`（模型线程那一侧），模型在 `LiveAvatarPipelineSource`。
这里只剩**接线**：

  1. LiveKit 推来的 PCM → 转发给音轨 ＋ 塞进 inbox
  2. 源里攒好的 RGBA 帧 → 按帧率吐成 `rtc.VideoFrame`
  3. 打断 → 四个地方一起清

> 早期版本把 PCM 缓冲放在这个类里。**位置就是错的** —— 那个缓冲是
> **模型线程**在读的，放在一个纯 asyncio 的类里连把锁都没有，跨线程
> 撕裂无声无息。挪到 `PcmInbox` 之后锁和阻塞语义都有了自然的归属。

## ⚠️ clear_buffer 是三个方法里最要命的

人已经不说话了而屏幕上的嘴还在动，恐怖谷一下就掉进去。被打断时
**在途的所有东西都要丢**。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol, Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

from .audio_stream import BlockGeometry, PcmInbox

log = logging.getLogger("closecrab.avatar.generator")

AVOut = Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


class FrameSource(Protocol):
    """模型那一侧。**只有四个成员** —— 多一个都说明编排漏到模型里去了。

    注意这里**一个 asyncio 的东西都没有**。模型跑在普通线程里，两边靠
    `PcmInbox`（带锁）和 `next_frame()`（非阻塞轮询）交接，不需要
    `call_soon_threadsafe`，也不需要谁持有事件循环的引用。
    """

    @property
    def size(self) -> tuple[int, int]:
        """(宽, 高)。**模型按 patch 网格取整之后的真实尺寸**，不是请求值。"""
        ...

    @property
    def geometry(self) -> BlockGeometry:
        ...

    @property
    def inbox(self) -> PcmInbox:
        ...

    def next_frame(self) -> np.ndarray | None:
        """取一帧 RGBA `HxWx4`，没有就返回 None。**不能阻塞。**"""
        ...

    def reset(self) -> None:
        """丢掉在途状态（含 inbox）。被打断时调。"""
        ...


class LiveAvatarGenerator(VideoGenerator):
    """真数字人。

    原音频**原样**跟着一起吐 —— 数字人只负责脸，声音还是 TTS 那一路的。
    """

    def __init__(self, source: FrameSource):
        self._src = source
        self._geom = source.geometry
        self._audio_out: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()

    @property
    def size(self) -> tuple[int, int]:
        return self._src.size

    # ── 进 ────────────────────────────────────────────────────────

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            # ⚠️ 段落结束**不清缓冲** —— 剩下那点不足一块的音频还要说完。
            #    真正该清的只有打断（`clear_buffer`）。混为一谈的话每句话
            #    结尾都会被吞掉一截，而且听起来像「他话没说完」。
            #
            # ⭐ 但要**明确放行**那截尾巴：inbox 平时攒够一整块才给模型，
            #    不放行的话最后不足一块的部分会一直卡在缓冲里等下一句。
            self._src.inbox.mark_segment_end()
            await self._audio_out.put(frame)
            return

        # 音频原样转发，**不等视频**。首帧压不下去，卡着等它整句话就晚一拍；
        # 声音先到一点点反而没人察觉。
        await self._audio_out.put(frame)
        self._src.inbox.push(np.frombuffer(frame.data, dtype=np.int16))

    def clear_buffer(self) -> None:
        """被打断。**把所有在途的东西一次丢干净。**

        三样，少一样都会漏：
          1. 已经排队、还没发出去的音频
          2. 还没被模型拉走的 PCM      ┐ 这两样归 source.reset()
          3. 模型侧已生成的在途帧       ┘
        """
        while not self._audio_out.empty():
            try:
                self._audio_out.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._src.reset()

    # ── 出 ────────────────────────────────────────────────────────

    async def __aiter__(self) -> AsyncIterator[AVOut]:
        interval = 1.0 / self._geom.fps
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            # 音频优先、有多少发多少：它不能等视频。
            while not self._audio_out.empty():
                yield self._audio_out.get_nowait()
            img = self._src.next_frame()
            if img is not None:
                yield to_video_frame(img)
            next_at += interval
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_at = loop.time()      # 落后就重新对齐，不追债


def to_video_frame(img: np.ndarray) -> rtc.VideoFrame:
    """RGBA `HxWx4` → LiveKit 帧。

    ⚠️ numpy 是 (高, 宽)，`VideoFrame` 是 (宽, 高)。写反了画面会拉伸撕裂，
    **而且不报错**。
    """
    return rtc.VideoFrame(width=img.shape[1], height=img.shape[0],
                          type=rtc.VideoBufferType.RGBA, data=img.tobytes())
