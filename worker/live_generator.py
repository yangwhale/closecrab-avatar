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
from collections import deque
from typing import AsyncIterator, Protocol, Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

from .audio_stream import BlockGeometry, PcmInbox
from .av_dump import AVDump

log = logging.getLogger("closecrab.avatar.generator")

AVOut = Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


def _int_env(name: str, default: int) -> int:
    """读一个整数旋钮。**读不出来就用默认值，绝不抛。**

    这几个旋钮是给真机 A/B 用的（「看着顺不顺」我这边量不出来），
    一个手滑的值不该让整场会话起不来。
    """
    import os
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r 不是整数，用默认 %d", name, raw, default)
        return default


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

    @property
    def pending(self) -> int:
        """还有多少帧攒着没被取走。"""
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
        # ⚠️ 用 deque 不用 `asyncio.Queue`：我们从不 `await` 取（只 `popleft`），
        #    而 Queue 唯一多出来的能力就是那个 await。用它反而要靠 `_queue[0]`
        #    这种私有属性才能看一眼队头 —— 那是随时会碎的。
        self._audio_out: deque[rtc.AudioFrame | AudioSegmentEnd] = deque()
        # ⭐ 「这一场到底有没有收到音频」必须能从日志一眼看出来。
        #    没有这个计数器的时候，一场没出画面的会话有两种完全不同的解释 ——
        #    音频没进来（上游的事）vs 进来了但没生成（模型的事）—— 而两边
        #    的日志都是**一片空白**，长得一模一样。查这个花了一上午。
        self._audio_frames = 0
        self._audio_seconds = 0.0

        # ── 抖动缓冲与音画对齐（2026-09-18 加，Chris 真机反馈）────────
        #
        # 症状两个，根是同一个：
        #
        #   跳帧     模型是**一块 12 帧 / 0.48 秒**地出，而下游
        #            `AVSynchronizer` 的视频队列只有 `int(25×100/1000)=2` 帧
        #            —— 80 毫秒缓冲。生产刚好是实时速度时没有任何余量，
        #            块边界一抖消费端就空手，一次断一到三帧。
        #   音画差   音频这一路**原样直发不等视频**（上面 `push_audio` 里
        #            那条注释），而视频必须等整块算完才存在。于是声音恒定
        #            早半拍 —— 实测 0.5 秒，跟一块的 0.48 秒对得上。
        #
        # 所以做两件事：**先攒够一块再开播**（给块边界留余量），
        # 以及**把音频拴在视频上**（领先超过阈值就等一等）。
        #
        # 代价是开口晚 0.5 秒左右。这个取舍是明确的：这一路是播报，
        # 不是实时对话，「稳且对齐」比「早半秒开口」值钱得多。
        self._preroll = _int_env("CCA_PREROLL_FRAMES", self._geom.fps // 2)
        """开播前先攒几帧。默认半秒 —— 略多于一块（0.48 s）。"""

        self._max_lead_s = _int_env(
            "CCA_MAX_AUDIO_LEAD_MS", round(1000 / self._geom.fps)) / 1000.0
        """音频最多领先视频多少。**默认就是一帧的时长**（25 fps ＝ 40 ms）。

        ## 为什么这里能严格对齐，而且不用知道生成延迟

        > Chris 2026-09-18：「真正对得齐的是音频第一帧和视频第一帧，它俩
        > 100% 一样。对齐以后后续的帧逐渐累加就行……不能靠猜。」

        对的，而且这条在我们这儿成立得很干净：模型**一块吃 0.48 秒音频、
        吐 12 帧**，而 12 ÷ 25 fps 正好也是 0.48 秒。两条媒体时间轴
        **天生等长，不会漂**。

        所以对齐只需要两件事，一件都不用猜：

            锚点   攒够之后第一次循环里，音频第一帧和视频第一帧一起出去
            步进   之后各按自己的时长累加，谁跑快了谁等

        生成延迟是 0.5 秒还是 0.9 秒、换张卡快了慢了 —— **完全不参与**。
        它只影响「锚点什么时候到」，不影响锚上之后的对齐。这也是为什么
        预缓冲是「等帧真的出现」而不是「等一个估出来的毫秒数」。

        额度取一帧而不是 0 —— 但理由要说准：**0 也能走**（实测过，两边会
        严格交替，一帧换一帧）。取一帧是为了让「同一时刻的音频和视频」能在
        **同一轮**里一起出去，而不是音频永远晚半帧跟在视频屁股后面。

        ⚠️ 它**不是给生成延迟留的余量**。延迟由预缓冲吸收，跟这个数无关。
        """

        self._max_lead = 0.0
        self._max_lag = 0.0
        self._audio_live = False
        """实测的对齐误差，**分正负两个方向记**。

        ⚠️ 只记一个绝对值是**错的**，2026-09-18 第一版就这么写，量出来
        660~740 ms，看着像对齐彻底失效 —— 其实绝大部分是**句尾那截**：
        音频说完了，缓冲里剩下的视频还在往外放，视频时间轴自然超过音频。
        那是正常的收尾，不是失步。

        所以分开记：`_max_lead` 是音频跑在前面（这个才该被额度管住），
        `_max_lag` 是视频跑在前面（句尾必然出现，参考意义小）。
        而且只在**音频还在流**的时候记 —— 段落一结束就停止统计。

        两个数都会在段落结束时打进日志。没有它们的话，「对齐做对了没有」
        只能靠眼睛看嘴型 —— 而那正是我之前拍一个 120 毫秒出来的原因。
        """

        # 排障用的离线录制。默认不开，开关见 `av_dump.AVDump`。
        self._dump = AVDump(self._geom.fps, source.size)

        self._rolling = False          # 攒够了没有
        self._video_out_s = 0.0        # 已吐出去的视频时长
        self._audio_out_s = 0.0        # 已吐出去的音频时长

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
            self._audio_live = False       # 句尾那截视频不计入统计
            self._src.inbox.mark_segment_end()
            self._audio_out.append(frame)
            log.info("音频段结束：本场累计 %d 帧 / %.1f s；"
                     "音频最多领先 %.0f ms（额度 %.0f ms）、最多落后 %.0f ms",
                     self._audio_frames, self._audio_seconds,
                     self._max_lead * 1000, self._max_lead_s * 1000,
                     self._max_lag * 1000)
            return

        self._audio_live = True
        self._audio_frames += 1
        self._audio_seconds += frame.samples_per_channel / frame.sample_rate
        if self._audio_frames == 1:
            log.info("⭐ 本场第一帧音频到了：%d Hz，%d 声道 —— 音频这条路是通的",
                     frame.sample_rate, frame.num_channels)

        # 两条路要的东西不一样，别合并：
        #   音轨这一份**原样转发** —— 听众听到的是发送方的原始采样率，
        #     而且不等视频（首帧压不下去，卡着等它整句话就晚一拍）
        #   模型这一份**要 16 kHz** —— wav2vec 只吃这个率
        self._audio_out.append(frame)
        self._src.inbox.push(np.frombuffer(frame.data, dtype=np.int16),
                             src_rate=frame.sample_rate)

    def clear_buffer(self) -> None:
        """被打断。**把所有在途的东西一次丢干净。**

        三样，少一样都会漏：
          1. 已经排队、还没发出去的音频
          2. 还没被模型拉走的 PCM      ┐ 这两样归 source.reset()
          3. 模型侧已生成的在途帧       ┘
        """
        self._audio_out.clear()
        self._src.reset()
        # ⚠️ 录制**不在这里关** —— 打断之后还会接着说，整场只录一份。
        #    2026-09-18 踩过：原来挂在「段落结束」上，结果一场会话录了 0.5 秒
        #    就自己停了，拿到的文件里视频 93 帧、音频 0 字节。
        #    「段落」和「会话」是两个粒度，这个工具要的是后者。
        # ⚠️ 抖动缓冲的状态也要清。不清的话打断之后 `_rolling` 还是 True，
        #    下一句**不重新攒**就直接开播 —— 那正好退回打断前的毛病，
        #    而且只在「被打断过」的那几句上出现，最难复现。
        self._rolling = False
        self._video_out_s = 0.0
        self._audio_out_s = 0.0
        self._max_drift = 0.0

    def close_recording(self) -> None:
        """会话结束：把录制收尾，写出 `meta.json`。

        ## 为什么单独一个方法，而不是挂在已有的钩子上

        录制的粒度是**一场会话**，而这个类身上另外两个「结束」都是更细的粒度：

            clear_buffer()      被打断 —— 后面还会接着说
            AudioSegmentEnd     一句说完 —— 后面还会接着说

        2026-09-18 挂在段落上，结果一场录了 0.5 秒就停了。改成只在打断/拆除时
        关，**又把唯一的调用点弄没了** —— 于是文件照写、`meta.json` 永远不写，
        而 `mux-av-dump.sh` 认准 `meta.json`（尺寸/帧率/采样率任一猜错，
        合出来的东西照样能播、只是错的，整场判断作废）。

        录了一堆裸流却合不出来，是个**沉默**的失败：日志里一行异常都没有。
        所以现在有一个明确的、由 runner 在 `finally` 里调的收尾点。
        """
        self._dump.close()

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
        frame_s = 1.0 / self._geom.fps
        while True:
            sent = False
            # ⚠️ 在**循环开头**量，不是末尾。末尾量的话每次 `yield` 之后
            #    生成器就挂起了，那一行常常根本跑不到 —— 于是偏差永远是 0，
            #    一个「测量」变成了一句安慰。
            if self._audio_live:
                d = self._audio_out_s - self._video_out_s
                if d > self._max_lead:
                    self._max_lead = d
                elif -d > self._max_lag:
                    self._max_lag = -d

            # ── 攒够了才开播 ──
            # 攒的时候两路都不吐：只拦视频的话音频会先跑掉半秒，
            # 那是在用「不同步」换「不卡顿」，两个毛病换一个。
            if not self._rolling:
                if self._src.pending >= self._preroll:
                    self._rolling = True
                    log.info("抖动缓冲攒够 %d 帧（%.0f ms），开播",
                             self._src.pending, self._preroll * frame_s * 1000)
                else:
                    await asyncio.sleep(idle)
                    continue

            # ── 音频：领先太多就等视频 ──
            # `AudioSegmentEnd` 不受这条管 —— 它是个标记不是声音，
            # 压着它只会让「这段说完了」这个信号迟到。
            if self._audio_out:
                is_mark = not isinstance(self._audio_out[0], rtc.AudioFrame)
                if is_mark or self._audio_out_s - self._video_out_s <= self._max_lead_s:
                    item = self._audio_out.popleft()
                    if not is_mark:
                        self._audio_out_s += (item.samples_per_channel
                                              / item.sample_rate)
                        self._dump.audio(bytes(item.data),
                                         sample_rate=item.sample_rate,
                                         num_channels=item.num_channels)
                    yield item
                    sent = True

            if (img := self._src.next_frame()) is not None:
                self._video_out_s += frame_s
                vf = to_video_frame(img)
                self._dump.video(bytes(vf.data))
                yield vf
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
