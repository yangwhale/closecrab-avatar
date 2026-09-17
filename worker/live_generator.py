"""把 LiveAvatar 接成 LiveKit 的 `VideoGenerator`。

协议只有三个方法（`livekit/agents/voice/avatar/_types.py`）：

    push_audio(frame | AudioSegmentEnd)   喂音频
    clear_buffer()                        打断时立刻停
    __aiter__()                           持续吐出视频帧和音频帧

## 这里只做编排，不碰模型

模型那一侧藏在 `FrameSource` 后面（下面那个 Protocol，只有两个方法）。
这么切有一个具体理由：**编排部分能在没有 GPU 的机器上穷举测**，
而模型部分只能在 B200 上验。混在一起的话两边都测不了。

不是为了「可扩展」—— 我们只会有一个实现。

## 调度算术在 `audio_stream.py`

「攒够多少音频才能开下一轮」那套已经跟上游源码逐点对拍过了，这里直接用。

## ⚠️ clear_buffer 是三个方法里最要命的

人已经不说话了而屏幕上的嘴还在动，恐怖谷一下就掉进去。被打断时
**在途的所有东西都要丢**：还没编码的音频、已经生成待发的帧、
以及模型侧那一轮。少丢一样都会漏出几帧。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol, Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

from .audio_stream import BucketGeometry, gather_slots

log = logging.getLogger("liveavatar.worker.generator")

AVOut = Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


class FrameSource(Protocol):
    """模型那一侧。**只有两个方法** —— 多一个都说明编排漏到模型里去了。"""

    async def render(self, embed_frames: np.ndarray) -> list[np.ndarray]:
        """吃一段音频嵌入帧，吐出对应的 RGB 帧（`HxWx4`，RGBA）。

        长度关系由 `BucketGeometry` 定死：一轮 `infer_frames` 个视频帧。
        """
        ...

    def reset(self) -> None:
        """丢掉这一轮的在途状态。被打断时调 —— 见上面 `clear_buffer` 那段。"""
        ...

    @property
    def size(self) -> tuple[int, int]:
        """(宽, 高)。"""
        ...


class LiveAvatarGenerator(VideoGenerator):
    """真数字人。

    音频从 `push_audio` 进来，攒够一轮就交给 `FrameSource` 渲染，
    渲出来的帧按视频帧率吐出去；原音频**原样**跟着一起吐 ——
    数字人只负责脸，声音还是 TTS 那一路的。
    """

    def __init__(self, source: FrameSource, *, geom: BucketGeometry | None = None,
                 encode_audio=None):
        self._src = source
        self._geom = geom or BucketGeometry()
        # 把 PCM 变成音频嵌入帧（30 Hz）。默认 None = 由调用方在真机上注入
        # wav2vec 那一套；离线测时塞一个假的。
        self._encode = encode_audio

        self._pending_pcm: list[np.ndarray] = []      # 还没编码的 PCM
        self._embed: np.ndarray | None = None          # 已编码的嵌入帧
        self._emitted_repeats = 0                      # 已经渲过几轮
        self._ended = False                            # 这一段话说完了吗

        self._out: asyncio.Queue[AVOut] = asyncio.Queue()
        self._audio_out: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._render_task: asyncio.Task | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self._src.size

    # ── 进 ────────────────────────────────────────────────────────

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            self._ended = True
            await self._audio_out.put(frame)
            self._kick()
            return

        # 音频原样转发。**不要等视频** —— 声音先到一点点没人察觉，
        # 而卡着等首帧（实测 1.26 s）会让整句话延迟一秒多。
        await self._audio_out.put(frame)

        pcm = np.frombuffer(frame.data, dtype=np.int16)
        self._pending_pcm.append(pcm)
        self._kick()

    def clear_buffer(self) -> None:
        """被打断。**把所有在途的东西一次丢干净。**

        丢五样，少一样都会漏帧：
          1. 还没编码的 PCM
          2. 已编码但还没渲的嵌入帧
          3. 轮次计数（不清零的话下一句会从第 N 轮的槽位取音频，
             取到越界或别人的帧，口型跟内容完全对不上）
          4. 队列里已经生成、还没发出去的帧
          5. 在途那一轮：`cancel()` 渲染任务 + `source.reset()`

        > 曾经这里还有一个「世代号」机制：每次打断加一，在途任务回来时
        > 比对，对不上就丢结果。**变异测试证明它一行都没起作用** ——
        > 任务挂在 `await render()` 上，`cancel()` 会在那儿抛
        > `CancelledError`，根本走不到比对那一步。已删。
        """
        self._pending_pcm.clear()
        self._embed = None
        self._emitted_repeats = 0
        self._ended = False
        for q in (self._out, self._audio_out):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self._render_task and not self._render_task.done():
            self._render_task.cancel()
        self._src.reset()

    # ── 渲 ────────────────────────────────────────────────────────

    def _kick(self) -> None:
        """有新音频了，看看够不够开下一轮。**同一时刻只允许一轮在渲。**"""
        if self._render_task and not self._render_task.done():
            return
        self._render_task = asyncio.create_task(self._render_ready())

    async def _render_ready(self) -> None:
        try:
            while True:
                if self._pending_pcm and self._encode is not None:
                    pcm = np.concatenate(self._pending_pcm)
                    self._pending_pcm.clear()
                    new = await self._encode(pcm)
                    self._embed = new if self._embed is None else np.concatenate(
                        [self._embed, new], axis=-2)

                have = 0 if self._embed is None else self._embed.shape[-2]
                ready = self._geom.repeats_ready(have, ended=self._ended)
                if ready <= self._emitted_repeats:
                    return                       # 还不够，等下一块音频

                # 超出已有音频的槽位由 gather_slots 填零（上游语义），
                # 冲句尾那一轮一定会用到。
                slots = gather_slots(self._embed, self._emitted_repeats, self._geom)
                # 被打断的话，CancelledError 会在上面这个 await 处抛出来，
                # 下面的入队根本不会执行 —— 这就是「在途那一轮不漏帧」的全部机制。
                frames = await self._src.render(slots)

                self._emitted_repeats += 1
                for img in frames:
                    await self._out.put(rtc.VideoFrame(
                        width=img.shape[1], height=img.shape[0],
                        type=rtc.VideoBufferType.RGBA, data=img.tobytes()))
        except Exception:
            # 渲染炸了不能把整条会话带走 —— 掉回「只出声」比整路断掉好。
            #
            # 不用再写一句 `except CancelledError: raise` —— 它继承自
            # `BaseException`（3.8 起），**`except Exception` 本来就捕不到**。
            # 写了等于没写，只是看着像做了防护。
            log.exception("渲染出错，这一轮跳过")

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
