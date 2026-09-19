"""句首抽干 —— **残留帧不许跨到下一句头上。**

> Chris 2026-09-19：「你现在要输入一个新的音频帧，在这之前你先把视频的
> 队列清空。这样的话从此以后再生成的视频帧，就是这个音频触发的。」

这组测试钉的是**判据**，不是实现：抽干必须挂在「一句话开始」这个**边界**
上。之前两版挂在「音频输入队列变空」这个**状态**上 —— 实时推流下那个状态
一句话中间要出现几十次，等于边说话边清自己的帧（实测同一段音频两次会话，
一次收到 0 帧、一次 140 帧）。`test_no_drain_mid_utterance` 就是拦这个的。
"""
import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from worker.audio_stream import BlockGeometry
from worker.live_generator import LiveAvatarGenerator

from test_live_generator import FakeSource, pcm_frame

G = BlockGeometry()


def _gen():
    src = FakeSource()
    return src, LiveAvatarGenerator(src)


@pytest.mark.asyncio
async def test_residual_dropped_before_first_audio():
    """预热留下的帧，必须在第一帧音频进模型之前就被扔掉。"""
    src, gen = _gen()
    for _ in range(24):            # 预热那 24 帧「不说话的脸」
        src.emit()
    assert src.pending == 24

    await gen.push_audio(pcm_frame(0.1))

    assert src.drops == 1
    assert src.dropped_frames == 24
    assert src.pending == 0, "残留没清干净，它们会顶在这句话的帧前面"


@pytest.mark.asyncio
async def test_no_drain_mid_utterance():
    """⚠️ **一句话中间一次都不许抽。** 这是之前两版翻车的地方。"""
    src, gen = _gen()
    await gen.push_audio(pcm_frame(0.1))
    assert src.drops == 1

    for _ in range(50):            # 实时推流：一句话里几十小块
        src.emit()                 # 模型同时在产帧
        await gen.push_audio(pcm_frame(0.04))

    assert src.drops == 1, f"句子中间又抽了 {src.drops - 1} 次，帧会被清光"
    assert src.pending == 50, "这句话自己的帧被清掉了"


@pytest.mark.asyncio
async def test_each_utterance_drains_once():
    """每句一次，不多不少。"""
    src, gen = _gen()
    for _ in range(3):
        await gen.push_audio(pcm_frame(0.1))
        await gen.push_audio(pcm_frame(0.1))
        await gen.push_audio(AudioSegmentEnd())
    assert src.drops == 3


@pytest.mark.asyncio
async def test_interrupt_reopens_the_boundary():
    """被打断之后那句不作数，下一句还得抽。

    不重置 `_utt_open` 的话下一句走不到抽干那条路 —— 被打断那句留下的
    在途帧会顶到下一句头上，而这正是「打断过之后才偏」那种最难复现的毛病。
    """
    src, gen = _gen()
    await gen.push_audio(pcm_frame(0.1))
    assert src.drops == 1

    gen.clear_buffer()
    src.emit(); src.emit()         # 打断之后模型又吐了两帧在途的
    await gen.push_audio(pcm_frame(0.1))

    assert src.drops == 2
    assert src.pending == 0


@pytest.mark.asyncio
async def test_expected_frame_count_rounds_up(caplog):
    """「这句话该出多少帧」是**算出来的**：块数向上取整 × 每块帧数。

    向上取整是因为段尾不足一块的那截由 `mark_segment_end()` 补零放行，
    它照样产一整块的帧。
    """
    src, gen = _gen()
    blk, fpb = G.block_samples, G.frames_per_block
    # 1.5 块 → 2 块 → 2×fpb 帧
    n = int(blk * 1.5)
    await gen.push_audio(rtc.AudioFrame(
        data=np.zeros(n, dtype=np.int16).tobytes(),
        sample_rate=G.sample_rate, num_channels=1, samples_per_channel=n))
    with caplog.at_level("INFO"):
        await gen.push_audio(AudioSegmentEnd())
    line = next(r.getMessage() for r in caplog.records if "应出" in r.getMessage())
    assert f"应出 {2 * fpb} 帧" in line, line


@pytest.mark.asyncio
async def test_drain_waits_for_inflight_pcm():
    """输入里还有没消化的 PCM 时，不能立刻抽 —— 那几帧会落在清空之后。

    这里把超时压到很短，验证的是「它确实在等」而不是「它等对了多久」：
    塞一块没人消费的 PCM，静止条件永远不成立，于是必须走超时那条路。
    """
    src, gen = _gen()
    gen._quiesce_s = 0.2
    src.inbox.push(np.zeros(G.block_samples, dtype=np.int16))

    t0 = asyncio.get_running_loop().time()
    await gen.push_audio(pcm_frame(0.1))
    waited = asyncio.get_running_loop().time() - t0

    assert waited >= 0.2, f"根本没等，只花了 {waited * 1000:.0f} ms"
    assert src.drops == 1, "等超时之后还是要抽的 —— 卡死比错半秒更糟"
