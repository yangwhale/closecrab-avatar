"""编排层 —— 用假模型，不碰 GPU。

这一层只做三件事：攒 PCM、模型来要时给**恰好一块**、把帧发出去。
所以测试也只盯这三件，外加最要命的打断。
"""
import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from worker.audio_stream import BlockGeometry
from worker.live_generator import LiveAvatarGenerator, to_video_frame

G = BlockGeometry()


class FakeSource:
    """假模型。记下回调、按需拉块、记 reset 次数。"""

    def __init__(self, *, w=64, h=48):
        self.w, self.h = w, h
        self.cb = None
        self.out = None
        self.resets = 0
        self.starts = 0

    @property
    def size(self):
        return (self.w, self.h)

    def start(self, audio_cb, out):
        self.starts += 1
        self.cb, self.out = audio_cb, out

    def reset(self):
        self.resets += 1

    def tick(self):
        """模拟模型拉一块并吐一帧。"""
        chunk = self.cb()
        self.out.put_nowait(to_video_frame(
            np.zeros((self.h, self.w, 4), dtype=np.uint8)))
        return chunk


def pcm_frame(seconds: float, value: int = 1000) -> rtc.AudioFrame:
    n = int(G.sample_rate * seconds)
    return rtc.AudioFrame(data=np.full(n, value, dtype=np.int16).tobytes(),
                          sample_rate=G.sample_rate, num_channels=1,
                          samples_per_channel=n)


def make():
    src = FakeSource()
    return src, LiveAvatarGenerator(src, geom=G)


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
async def test_segment_end_does_not_clear_the_buffer():
    """⭐ 段落结束**不能清缓冲** —— 剩下那点不足一块的音频还要说完。

    跟打断混为一谈的话，每句话结尾都被吞掉一截，听起来像「他话没说完」，
    而且不报错。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(0.1, value=1000))     # 远不足一块
    await gen.push_audio(AudioSegmentEnd())
    chunk = src.cb()
    assert len(chunk) == G.block_samples
    assert np.count_nonzero(chunk) > 0, "段落结束把没说完的音频清掉了"


# ── 模型来拉音频：长度恒定 + 静音兜底 ──────────────────────────────

@pytest.mark.asyncio
async def test_block_length_is_always_exact():
    """⭐ 无论缓冲里有多少，返回长度**恒为** block_samples。

    长度不固定的话模型侧编码窗口会错位 —— 不报错，只是口型跟声音差一截。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(0.01))           # 远不足
    assert len(src.cb()) == G.block_samples
    await gen.push_audio(pcm_frame(5.0))            # 远超
    for _ in range(5):
        assert len(src.cb()) == G.block_samples


@pytest.mark.asyncio
async def test_silence_when_empty():
    """没货给静音 —— 对应「人在那儿但没说话」，不是错误状态。"""
    src, gen = make()
    await gen.push_audio(pcm_frame(0.01))
    src.cb()                                        # 把那点货吃掉
    assert not np.any(src.cb()), "缓冲空了却没给静音"


@pytest.mark.asyncio
async def test_int16_is_scaled_to_unit_float():
    """⭐ int16 必须转成 [-1,1] 的 float32。

    少这一步 wav2vec 的 processor 会把整数当振幅，特征完全不对 ——
    而它照样出视频，只是口型对不上。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(1.0, value=32767))
    chunk = src.cb()
    assert chunk.dtype == np.float32
    assert 0.9 < chunk.max() <= 1.0, f"没归一化：max={chunk.max()}"


@pytest.mark.asyncio
async def test_pcm_is_consumed_in_order_without_gaps():
    """跨多个 AudioFrame 拼接时不能丢样本、不能重复。"""
    src, gen = make()
    await gen.push_audio(pcm_frame(0.3, value=100))      # 4800
    await gen.push_audio(pcm_frame(0.3, value=200))      # 4800，合计 9600
    first = src.cb()
    assert np.count_nonzero(first) == G.block_samples, "第一块就出现空洞"
    second = src.cb()
    assert np.count_nonzero(second) == 9600 - G.block_samples, \
        f"第二块真样本数不对：{np.count_nonzero(second)}"


# ── 模型什么时候启动 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_starts_only_on_first_audio():
    """⭐ 没人说话就不启动模型 —— 不说话不占卡。"""
    src, gen = make()
    assert src.starts == 0
    await gen.push_audio(pcm_frame(0.1))
    assert src.starts == 1
    await gen.push_audio(pcm_frame(0.1))
    assert src.starts == 1, "重复启动了"


# ── ⭐ 打断 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_buffer_drops_everything():
    src, gen = make()
    await gen.push_audio(pcm_frame(3.0))
    src.tick()                                      # 队列里放一帧
    assert not gen._out.empty()

    gen.clear_buffer()
    assert gen._out.empty(), "帧队列没清"
    assert gen._audio_out.empty(), "音频队列没清"
    assert src.resets == 1, "没通知模型丢在途状态"
    assert not np.any(src.cb()), "PCM 缓冲没清 —— 打断后还在念上一句"


@pytest.mark.asyncio
async def test_can_speak_again_after_interrupt():
    src, gen = make()
    await gen.push_audio(pcm_frame(3.0))
    gen.clear_buffer()
    await gen.push_audio(pcm_frame(3.0, value=500))
    assert np.count_nonzero(src.cb()) == G.block_samples, "打断之后新的话进不来"


# ── 帧格式 ────────────────────────────────────────────────────────

def test_frame_conversion_keeps_dimensions():
    """⚠️ numpy 是 (高, 宽)，VideoFrame 是 (宽, 高)。写反了画面会拉伸撕裂，
    而且不报错。
    """
    f = to_video_frame(np.zeros((48, 64, 4), dtype=np.uint8))
    assert (f.width, f.height) == (64, 48)


def test_size_comes_from_the_model():
    src, gen = make()
    assert gen.size == src.size
