"""编排层 —— 用假模型，不碰 GPU。

这一层现在薄得只剩接线：音频转发 ＋ 塞 inbox、出帧、打断。
缓冲本身的行为在 `test_pcm_inbox.py`，不在这里重复。
"""
import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from worker.audio_stream import BlockGeometry, PcmInbox
from worker.live_generator import LiveAvatarGenerator, to_video_frame

G = BlockGeometry()


class FakeSource:
    """假模型。真 inbox（就那一个类，没必要再假一个），帧靠手工塞。"""

    def __init__(self, *, w=64, h=48):
        self.w, self.h = w, h
        self._inbox = PcmInbox(G)
        self._frames: list[np.ndarray] = []
        self.resets = 0

    @property
    def size(self):
        return (self.w, self.h)

    @property
    def geometry(self):
        return G

    @property
    def inbox(self):
        return self._inbox

    @property
    def pending(self):
        return len(self._frames)

    def next_frame(self):
        return self._frames.pop(0) if self._frames else None

    def reset(self):
        self.resets += 1
        self._inbox.clear()
        self._frames.clear()

    def emit(self):
        self._frames.append(np.zeros((self.h, self.w, 4), dtype=np.uint8))


def pcm_frame(seconds: float, value: int = 1000) -> rtc.AudioFrame:
    n = int(G.sample_rate * seconds)
    return rtc.AudioFrame(data=np.full(n, value, dtype=np.int16).tobytes(),
                          sample_rate=G.sample_rate, num_channels=1,
                          samples_per_channel=n)


def make(*, preroll=0, max_lead_ms=None):
    """造一对。

    ⚠️ **默认把抖动缓冲关掉（preroll=0）。** 下面那批用例测的是
    「交替吐」「不自己节流」「打断清干净」这些**逐帧语义**，它们只塞一两帧；
    带着默认的半秒门槛会全部卡在「还没攒够」上 —— 红的原因跟它们要测的
    东西毫无关系。缓冲本身另有专门的用例。
    """
    src = FakeSource()
    gen = LiveAvatarGenerator(src)
    gen._preroll = preroll
    if max_lead_ms is not None:
        gen._max_lead_s = max_lead_ms / 1000.0
    else:
        # 老用例假定音频**不等视频**（那时候就是这么设计的）。给一个大到
        # 不会触发的领先额度，让它们继续验各自的东西。
        gen._max_lead_s = 1e9
    return src, gen


async def drain(gen, n, timeout=2.0):
    got = []

    async def _run():
        async for item in gen:
            got.append(item)
            if len(got) >= n:
                return
    try:
        await asyncio.wait_for(_run(), timeout)
    except asyncio.TimeoutError:
        pass
    return got


# ── 音频不能等视频 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_passes_through_immediately():
    """⭐ 音频原样转发，**不等视频**。卡着等首帧整句话就晚一拍。"""
    src, gen = make()
    await gen.push_audio(pcm_frame(0.1))
    got = await drain(gen, 1, timeout=0.5)
    assert got and isinstance(got[0], rtc.AudioFrame)


@pytest.mark.asyncio
async def test_pushed_audio_reaches_the_model():
    """推进来的 PCM 必须进 inbox —— 漏了的症状是「有声音但嘴不动」。"""
    src, gen = make()
    await gen.push_audio(pcm_frame(0.1))
    assert src.inbox.pending_samples == int(G.sample_rate * 0.1)


@pytest.mark.asyncio
async def test_segment_end_does_not_clear_the_buffer():
    """⭐ 段落结束**不能清缓冲** —— 剩下那点不足一块的音频还要说完。

    跟打断混为一谈的话，每句话结尾都被吞掉一截，听起来像「他话没说完」，
    而且不报错。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(0.1, value=1000))     # 远不足一块
    await gen.push_audio(AudioSegmentEnd())
    assert src.inbox.pending_samples > 0, "段落结束把没说完的音频清掉了"
    assert src.resets == 0, "段落结束不该通知模型丢状态"


# ── 出帧 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_frames_are_emitted_as_video_frames():
    src, gen = make()
    src.emit()
    got = await drain(gen, 1, timeout=1.0)
    assert got and isinstance(got[0], rtc.VideoFrame)
    assert (got[0].width, got[0].height) == src.size


@pytest.mark.asyncio
async def test_generator_does_not_pace_frames():
    """⭐ **这一层不做帧率节流**，攒了多少一次吐多少。

    下游 `AVSynchronizer` 自带 `_FPSController`。两道闸串起来每帧付两次
    间隔，吞吐对半砍 —— 实测 25 fps 的目标只跑出 10 fps，看起来像 GPU
    不够，其实是自己多排了一道节奏。

    判据：一整块 12 帧要在**远小于 12/fps（0.48 s）**的时间里吐完。
    """
    src, gen = make()
    for _ in range(12):
        src.emit()
    t0 = asyncio.get_running_loop().time()
    got = await drain(gen, 12, timeout=2.0)
    elapsed = asyncio.get_running_loop().time() - t0
    assert len(got) == 12, f"只吐出 {len(got)} 帧"
    assert elapsed < 12 / G.fps / 2, \
        f"吐 12 帧花了 {elapsed:.3f}s —— 这一层还在按帧率节流"


# ── ⭐ 打断 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_buffer_drops_everything():
    src, gen = make()
    await gen.push_audio(pcm_frame(3.0))
    src.emit()

    gen.clear_buffer()
    assert not gen._audio_out, "音频队列没清"
    assert src.resets == 1, "没通知模型丢在途状态"
    assert src.inbox.pending_samples == 0, "PCM 没清 —— 打断后还在念上一句"
    assert src.next_frame() is None, "已生成的帧没丢 —— 打断后嘴还在动"


@pytest.mark.asyncio
async def test_can_speak_again_after_interrupt():
    src, gen = make()
    await gen.push_audio(pcm_frame(3.0))
    gen.clear_buffer()
    await gen.push_audio(pcm_frame(3.0, value=500))
    assert src.inbox.pending_samples > 0, "打断之后新的话进不来"


# ── 帧格式 ────────────────────────────────────────────────────────

def test_frame_conversion_keeps_dimensions():
    """⚠️ numpy 是 (高, 宽)，VideoFrame 是 (宽, 高)。写反了画面会拉伸撕裂，
    而且不报错。
    """
    f = to_video_frame(np.zeros((48, 64, 4), dtype=np.uint8))
    assert (f.width, f.height) == (64, 48)


def test_size_comes_from_the_model():
    """⭐ 尺寸取模型真实出帧尺寸，不取请求值 —— 模型按 64 的网格取整。"""
    src, gen = make()
    assert gen.size == src.size


# ── ⭐ 音视频交错：把视频饿死的那个 bug ────────────────────────────

@pytest.mark.asyncio
async def test_audio_and_video_interleave_one_by_one():
    """⭐ 一轮最多各吐一帧。**不能把音频抽干再吐视频。**

    下游 `_forward_video` 是单个任务，音频视频都从这一个生成器拿，拿到音频
    就 `await audio_source.capture_frame()` —— 那个调用按实时阻塞。

    「音频有多少吐多少」的写法遇上成段灌进来的 TTS：一句话瞬间在队列里堆几秒
    音频，于是下游卡在音频里几秒钟一帧视频都收不到，而模型还在 25 fps 生产，
    帧队列满了开始丢。实测模型产出 450 帧全好，线上只到 98 帧、5.3 fps、
    44% 解不出画面 —— 表现成「一直黑屏，说完话几秒后蹦出一张脸」。

    判据就是吐出来的顺序必须是 AVAVAV，不能是 AAAA…VVVV。
    """
    src, gen = make()
    for _ in range(6):
        await gen.push_audio(pcm_frame(0.1))     # 音频先堆满，模拟 TTS 成段灌
        src.emit()
    kinds = ["A" if isinstance(x, rtc.AudioFrame) else "V"
             for x in await drain(gen, 8)]
    assert "".join(kinds) == "AVAVAVAV", \
        f"没有交错，视频会被饿死：{''.join(kinds)}"


@pytest.mark.asyncio
async def test_video_still_flows_while_audio_is_backlogged():
    """音频积压时视频**不能停**。

    上一条测顺序，这条测后果：音频堆了 20 帧，视频只有 3 帧 —— 这 3 帧必须
    在前 6 次产出里就出来，而不是排在 20 帧音频后面。
    """
    src, gen = make()
    for _ in range(20):
        await gen.push_audio(pcm_frame(0.1))
    for _ in range(3):
        src.emit()
    first6 = await drain(gen, 6)
    videos = sum(1 for x in first6 if isinstance(x, rtc.VideoFrame))
    assert videos == 3, f"前 6 帧里只有 {videos} 帧视频 —— 视频被音频挡住了"


# ── ⭐ 抖动缓冲：攒够一块再开播 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_preroll_holds_both_paths_until_buffered():
    """攒的时候**音频和视频都不吐**。

    只拦视频的话音频会先跑掉半秒 —— 那是拿「不同步」换「不卡顿」，
    两个毛病换一个。Chris 2026-09-18 报的正是这两个症状同时出现。
    """
    src, gen = make(preroll=5)
    await gen.push_audio(pcm_frame(0.1))
    src.emit(); src.emit()                     # 只有 2 帧，不够 5
    got = await drain(gen, 1, timeout=0.3)
    assert got == [], f"没攒够就开播了：{got}"

    for _ in range(3):
        src.emit()                             # 凑到 5
    got = await drain(gen, 2, timeout=1.0)
    assert got, "攒够了还不开播"


@pytest.mark.asyncio
async def test_preroll_rearms_after_interrupt():
    """⭐ 打断之后必须**重新攒**。

    不重置 `_rolling` 的话，下一句直接开播、没有任何缓冲 —— 正好退回
    打断前的卡顿，而且**只在「被打断过」的那几句上出现**，最难复现。
    """
    src, gen = make(preroll=3)
    for _ in range(3):
        src.emit()
    await drain(gen, 1, timeout=1.0)
    assert gen._rolling is True

    gen.clear_buffer()
    assert gen._rolling is False, "打断后没重新武装，下一句会裸奔"

    src.emit()                                 # 只有 1 帧，不够 3
    assert await drain(gen, 1, timeout=0.3) == [], "打断后没攒够就又开播了"


# ── ⭐ 音画对齐：音频不许领先视频太多 ────────────────────────────────


@pytest.mark.asyncio
async def test_audio_waits_for_video_when_too_far_ahead():
    """音频领先超过阈值就等一等。

    根因：视频必须等整块算完才存在（一块 12 帧 / 0.48 s），而音频原样直发。
    不拴住的话声音恒定早半拍 —— Chris 实测 0.5 秒，跟一块的时长对得上。
    """
    src, gen = make(preroll=0, max_lead_ms=50)
    for _ in range(6):                         # 0.6 秒音频，一帧视频都没有
        await gen.push_audio(pcm_frame(0.1))
    got = await drain(gen, 6, timeout=0.5)
    audios = [g for g in got if isinstance(g, rtc.AudioFrame)]
    # 50 ms 额度 + 一帧 100 ms：最多吐一帧就该停下等视频。
    assert len(audios) <= 1, f"音频没被拴住，一口气吐了 {len(audios)} 帧"


@pytest.mark.asyncio
async def test_audio_resumes_once_video_catches_up():
    """反向验证：视频跟上来，音频必须接着走。**只测「会等」不够** ——
    一个永远不放行的实现也能通过上一条，而那是彻底哑掉。"""
    src, gen = make(preroll=0, max_lead_ms=50)
    for _ in range(6):
        await gen.push_audio(pcm_frame(0.1))
    await drain(gen, 2, timeout=0.5)
    for _ in range(30):                        # 1.2 秒视频（25 fps）
        src.emit()
    got = await drain(gen, 20, timeout=1.5)
    assert any(isinstance(g, rtc.AudioFrame) for g in got), "视频跟上了音频还不走"


@pytest.mark.asyncio
async def test_segment_end_is_never_held_back():
    """`AudioSegmentEnd` 是**标记不是声音**，不受领先额度管。

    压着它只会让「这段说完了」这个信号迟到 —— 而下游靠它收尾。
    """
    src, gen = make(preroll=0, max_lead_ms=0)
    await gen.push_audio(AudioSegmentEnd())
    got = await drain(gen, 1, timeout=0.5)
    assert got and isinstance(got[0], AudioSegmentEnd), f"段结束被压住了：{got}"
