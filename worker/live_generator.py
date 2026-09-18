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
        # ⭐ 「这一场到底有没有收到音频」必须能从日志一眼看出来。
        #    没有这个计数器的时候，一场没出画面的会话有两种完全不同的解释 ——
        #    音频没进来（上游的事）vs 进来了但没生成（模型的事）—— 而两边
        #    的日志都是**一片空白**，长得一模一样。查这个花了一上午。
        self._audio_frames = 0
        self._audio_seconds = 0.0

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
            log.info("音频段结束：本场累计 %d 帧 / %.1f s",
                     self._audio_frames, self._audio_seconds)
            return

        self._audio_frames += 1
        self._audio_seconds += frame.samples_per_channel / frame.sample_rate
        if self._audio_frames == 1:
            log.info("⭐ 本场第一帧音频到了：%d Hz，%d 声道 —— 音频这条路是通的",
                     frame.sample_rate, frame.num_channels)

        # 两条路要的东西不一样，别合并：
        #   音轨这一份**原样转发** —— 听众听到的是发送方的原始采样率，
        #     而且不等视频（首帧压不下去，卡着等它整句话就晚一拍）
        #   模型这一份**要 16 kHz** —— wav2vec 只吃这个率
        await self._audio_out.put(frame)
        self._src.inbox.push(np.frombuffer(frame.data, dtype=np.int16),
                             src_rate=frame.sample_rate)

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
        """有多少吐多少。**这里不做帧率节流。**

        ⚠️ 踩过：原来这里按 `1/fps` 自己排了一遍节奏。但下游
        `AVSynchronizer` 自带 `_FPSController(expected_fps=video_fps)` ——
        两道闸串在一起，每帧要付两次 40 ms，吞吐直接对半砍。
        实测 25 fps 的目标只跑出 10 fps，看起来像 GPU 不够。

        > 跟音频分块那次是同一个错：**自己实现调度之前，先看被调用方
        > 是不是已经在调度。** 它 `async for` 逐帧拉，节奏就归它管。

        没帧的时候短睡一下，别空转 —— 这是「让出 CPU」不是「排节奏」，
        所以取一个远小于帧间隔的值。

        ⚠️⚠️ **一轮最多各吐一帧，绝不能「把音频抽干再吐视频」。**

        这一条是 2026-09-18 实测出来的，代价很大，写清楚：

        下游 `_forward_video` 是**单个任务**，音频和视频都从这一个生成器里拿，
        拿到音频就 `await audio_source.capture_frame()` —— 而那个调用是**按实时
        阻塞**的（音频源就是整条流水线的时钟，这是 AVSynchronizer 的设计）。

        原来这里写的是「音频有多少吐多少，然后视频有多少吐多少」。TTS 是**成段
        灌进来**的，一句话瞬间就在队列里堆了好几秒音频 —— 于是一进循环就吐出
        几秒钟的音频帧，下游卡在音频 `capture_frame` 里**好几秒一帧视频都收不到**。
        这几秒里模型还在以 25 fps 生产，我们自己的帧队列满了开始丢最旧的。

        实测：模型侧产出 450 帧全部正常，线上只收到 98 帧、5.3 fps，
        其中约 44% 因为丢了参考帧而解不出画面。**表现出来就是「一直黑屏，
        说完话过几秒才蹦出一张脸」** —— 而每一层自己看都是正常的：
        模型说我产出 25 fps，编码器说我在编，客户端说我收到了帧。

        一轮各吐一帧就把这个解开了：音频帧是 10~100 ms、视频帧是 40 ms，
        交替吐出去，下游那次实时阻塞天然变成视频的节拍器。

        > 教训：**「优先级」不等于「先把它抽干」。** 下游是单消费者时，
        > 抽干高优先级的那一路 ＝ 把另一路饿死整整一个批次的时长。
        """
        idle = 1.0 / (self._geom.fps * 4)
        while True:
            sent = False
            # 音频先于视频 —— 它不能等。但**只拿一帧**，理由见上面。
            if not self._audio_out.empty():
                yield self._audio_out.get_nowait()
                sent = True
            if (img := self._src.next_frame()) is not None:
                yield to_video_frame(img)
                sent = True
            if not sent:
                await asyncio.sleep(idle)


def to_video_frame(img: np.ndarray) -> rtc.VideoFrame:
    """RGBA `HxWx4` → LiveKit 帧。

    ⚠️ numpy 是 (高, 宽)，`VideoFrame` 是 (宽, 高)。写反了画面会拉伸撕裂，
    **而且不报错**。
    """
    return rtc.VideoFrame(width=img.shape[1], height=img.shape[0],
                          type=rtc.VideoBufferType.RGBA, data=img.tobytes())
