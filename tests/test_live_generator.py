"""编排层的测试 —— 用假模型，不碰 GPU。

最要紧的一条是**打断**：人不说话了而屏幕上嘴还在动，恐怖谷一下就掉进去。
而它最难测，因为「漏几帧」在功能上完全看不出来，只有把在途那一轮
故意卡住、打断、再放它回来，才会现形。
"""
import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from worker.audio_stream import BucketGeometry
from worker.live_generator import LiveAvatarGenerator

G = BucketGeometry()
EMBED_DIM = 8


class FakeSource:
    """假模型。记调用、可控延迟、可控卡住。"""

    def __init__(self, *, w=64, h=48):
        self.w, self.h = w, h
        self.calls = 0
        self.resets = 0
        self.gate: asyncio.Event | None = None      # 设了就卡在这儿

    @property
    def size(self):
        return (self.w, self.h)

    def reset(self):
        self.resets += 1

    async def render(self, embed_frames):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        n = G.infer_frames
        return [np.zeros((self.h, self.w, 4), dtype=np.uint8) for _ in range(n)]


async def fake_encode(pcm):
    """假编码器：每 `sr/30` 个采样出一个嵌入帧。返回 (1, T, D)。"""
    n = max(1, len(pcm) // (16000 // G.video_rate))
    return np.zeros((1, n, EMBED_DIM), dtype=np.float32)


def pcm_frame(seconds: float) -> rtc.AudioFrame:
    n = int(16000 * seconds)
    return rtc.AudioFrame(data=np.zeros(n, dtype=np.int16).tobytes(),
                          sample_rate=16000, num_channels=1, samples_per_channel=n)


def make():
    src = FakeSource()
    return src, LiveAvatarGenerator(src, geom=G, encode_audio=fake_encode)


async def drain(gen, n, timeout=2.0):
    """从 __aiter__ 取 n 个东西出来。"""
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
    """⭐ 音频原样转发，**不等视频**。

    首帧实测 1.26 秒压不下去。卡着等它，整句话就晚一秒多；
    声音先到一点点反而没人察觉。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(0.1))
    got = await drain(gen, 1, timeout=0.5)
    assert got and isinstance(got[0], rtc.AudioFrame)
    assert src.calls == 0, "才 0.1 秒音频就去渲染了"


@pytest.mark.asyncio
async def test_segment_end_forwarded():
    src, gen = make()
    await gen.push_audio(AudioSegmentEnd())
    got = await drain(gen, 1, timeout=0.5)
    assert any(isinstance(g, AudioSegmentEnd) for g in got)


# ── 攒够一轮才渲 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_renders_only_when_a_full_repeat_is_ready():
    """一轮要 149 个嵌入帧（≈4.97 s）。差一点都不能开 ——
    开了就得拿裁剪值凑，口型会跟声音错开，而且不报错。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(4.0))          # 120 帧，不够
    await asyncio.sleep(0.05)
    assert src.calls == 0, "不够一轮就渲了"

    await gen.push_audio(pcm_frame(1.5))          # 累计 165 帧，够了
    await asyncio.sleep(0.15)
    assert src.calls == 1, f"够了却没渲（calls={src.calls}）"


@pytest.mark.asyncio
async def test_segment_end_flushes_the_tail():
    """⭐ 说完了，尾巴那点不足一轮的也要出 —— 否则句尾被吞，
    用户听到话没说完、画面先停。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(1.0))          # 远不够一轮
    await asyncio.sleep(0.05)
    assert src.calls == 0
    await gen.push_audio(AudioSegmentEnd())
    await asyncio.sleep(0.15)
    assert src.calls == 1, "说完了尾巴没出"


@pytest.mark.asyncio
async def test_frames_have_the_right_shape():
    src, gen = make()
    await gen.push_audio(pcm_frame(6.0))
    got = await drain(gen, 12, timeout=2.0)
    vids = [g for g in got if isinstance(g, rtc.VideoFrame)]
    assert vids, "一帧视频都没出"
    assert (vids[0].width, vids[0].height) == src.size


# ── ⭐ 打断：这一节是整个文件的理由 ────────────────────────────────

@pytest.mark.asyncio
async def test_clear_buffer_drops_everything_in_flight():
    src, gen = make()
    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.2)
    assert not gen._out.empty(), "前置条件不成立：队列里本该有帧"

    gen.clear_buffer()
    assert gen._out.empty() and gen._audio_out.empty(), "打断后队列没清干净"
    assert src.resets == 1, "没通知模型丢掉这一轮"


@pytest.mark.asyncio
async def test_frames_rendered_before_the_interrupt_never_leak_out():
    """⭐ 这条抓的是最难看见的漏：**在途那一轮回来时，帧不能再进队列**。

    用户打断了，屏幕上却还在把上一句话的口型演完 —— 几帧而已，
    但正是恐怖谷的来源，而且日志里什么都看不到。

    做法：把假模型卡在渲染中间 → 打断 → 放它回来 → 队列必须仍然是空的。

    机制是 `clear_buffer()` 里那句 `cancel()`：任务挂在 `await render()` 上，
    取消会在那儿抛 `CancelledError`，入队那几行根本不执行。
    （曾经还叠了一个「世代号」比对，**变异测试证明它一行都没起作用** ——
    因为永远走不到。已删，见 `clear_buffer` 的注释。）
    """
    src, gen = make()
    src.gate = asyncio.Event()                    # 卡住这一轮
    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.1)
    assert src.calls == 1 and gen._out.empty(), "前置条件不成立"

    gen.clear_buffer()                            # 打断（模型还卡着）
    src.gate.set()                                # 模型现在回来了
    await asyncio.sleep(0.2)

    assert gen._out.empty(), "打断之后，在途那一轮的帧还是漏出来了"


@pytest.mark.asyncio
async def test_can_speak_again_after_an_interrupt():
    """打断不能把生成器弄成半死不活 —— 下一句话必须照常渲。"""
    src, gen = make()
    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.2)
    gen.clear_buffer()

    before = src.calls
    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.25)
    assert src.calls > before, "打断之后再也不渲了"


@pytest.mark.asyncio
async def test_interrupt_resets_the_repeat_counter():
    """⭐ 计数不清零的话，下一句话会从第 N 轮的槽位开始取音频 ——
    取到的是**越界或者别人的**嵌入帧，口型跟内容完全对不上。
    """
    src, gen = make()
    await gen.push_audio(pcm_frame(11.0))         # 两轮
    await asyncio.sleep(0.3)
    assert gen._emitted_repeats >= 1
    gen.clear_buffer()
    assert gen._emitted_repeats == 0


# ── 别的 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_render_at_a_time():
    """⭐ 同一时刻只许一轮在渲。并发两轮会抢同一张卡，
    而且第二轮拿到的 `_emitted_repeats` 是旧值 → 两轮取同一段音频。
    """
    src, gen = make()
    src.gate = asyncio.Event()
    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.05)
    await gen.push_audio(pcm_frame(6.0))          # 再喂，不该起第二个
    await asyncio.sleep(0.05)
    assert src.calls == 1, f"并发起了 {src.calls} 轮"
    src.gate.set()


@pytest.mark.asyncio
async def test_render_failure_does_not_kill_the_session():
    """渲染炸了只能掉回「只出声」，不能把整路会话带走。"""
    src, gen = make()

    async def boom(_):
        raise RuntimeError("模型炸了")
    src.render = boom

    await gen.push_audio(pcm_frame(6.0))
    await asyncio.sleep(0.2)
    # 音频照常出得来
    got = await drain(gen, 1, timeout=0.5)
    assert got and isinstance(got[0], rtc.AudioFrame)


def test_size_comes_from_the_model():
    src, gen = make()
    assert gen.size == src.size
