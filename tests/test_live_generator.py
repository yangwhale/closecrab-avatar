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


def make():
    src = FakeSource()
    return src, LiveAvatarGenerator(src)


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


# ── ⭐ 打断 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_buffer_drops_everything():
    src, gen = make()
    await gen.push_audio(pcm_frame(3.0))
    src.emit()

    gen.clear_buffer()
    assert gen._audio_out.empty(), "音频队列没清"
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
