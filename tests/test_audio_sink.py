"""数字人音频出口的生命周期 —— 什么时候该换一条流。

钉的是 2026-09-18 那个静默 bug 的语义：`clear_buffer()` 只通知对端丢缓冲，
**不换流**；换流只能靠 `flush()`。两个都调、而且顺序不能反。

跑法：`python3 -m pytest tests/test_audio_sink.py -q`
"""
import asyncio

import pytest

from closecrab_avatar.audio_sink import AvatarAudioSink


class FakeRaw:
    """记下被调了什么、什么顺序。"""

    def __init__(self):
        self.calls: list[str] = []
        self.frames = 0
        self.clear_raises = False

    async def capture_frame(self, frame):
        self.frames += 1
        if "capture_frame" not in self.calls:
            self.calls.append("capture_frame")

    def flush(self):
        self.calls.append("flush")

    def clear_buffer(self):
        self.calls.append("clear_buffer")
        if self.clear_raises:
            raise RuntimeError("对端不认这个 RPC")


def sink():
    raw = FakeRaw()
    return raw, AvatarAudioSink(raw, label="test")


# ── 打断 / 重播 ──────────────────────────────────────────────────────


def test_interrupt_does_both():
    """⭐ 整个文件最值钱的一条。少了 flush，重播之后对端永远收不到音频。"""
    raw, s = sink()
    asyncio.run(s.capture_frame(object()))
    s.interrupt()
    assert raw.calls == ["capture_frame", "clear_buffer", "flush"]


def test_interrupt_order_matters():
    """flush 先跑的话流已经没了，clear_buffer 那道 `_started` 门会直接 return，
    对端手里那几百毫秒就留下来 —— 嘴型接着念上一句。"""
    raw, s = sink()
    asyncio.run(s.capture_frame(object()))
    s.interrupt()
    assert raw.calls.index("clear_buffer") < raw.calls.index("flush")


def test_interrupt_flushes_even_if_clear_raises():
    """缓冲没清掉已经够糟，再把流焊死就彻底没救。"""
    raw, s = sink()
    raw.clear_raises = True
    asyncio.run(s.capture_frame(object()))
    s.interrupt()                              # 不该抛
    assert raw.calls == ["capture_frame", "clear_buffer", "flush"]


def test_interrupt_without_audio_still_notifies():
    """还没写过音频也要通知对端 —— 它可能还攥着上一句的尾巴。

    但流本身没开过，`_rotate` 会跳过 flush。
    """
    raw, s = sink()
    s.interrupt()
    assert raw.calls == ["clear_buffer"]


# ── 一句说完 ─────────────────────────────────────────────────────────


def test_end_utterance_rotates():
    raw, s = sink()
    asyncio.run(s.capture_frame(object()))
    s.end_utterance()
    assert raw.calls == ["capture_frame", "flush"]
    assert s.dirty is False


def test_end_utterance_is_idempotent():
    """调用方通常在空闲循环里每秒调一次。不挡的话会对着空流反复开关流任务。"""
    raw, s = sink()
    asyncio.run(s.capture_frame(object()))
    s.end_utterance()
    s.end_utterance()
    s.end_utterance()
    assert raw.calls.count("flush") == 1, raw.calls


def test_end_utterance_before_any_audio_does_nothing():
    raw, s = sink()
    s.end_utterance()
    assert raw.calls == []


def test_new_audio_after_end_starts_a_fresh_cycle():
    """⭐ 这一条是「隔一会儿再说下一句」那个场景的核心：
    换流之后再写，必须能再换一次 —— 否则第二句之后就永远不换了。"""
    raw, s = sink()
    asyncio.run(s.capture_frame(object()))
    s.end_utterance()
    asyncio.run(s.capture_frame(object()))
    assert s.dirty is True
    s.end_utterance()
    assert raw.calls.count("flush") == 2, raw.calls


# ── 收摊 ─────────────────────────────────────────────────────────────


def test_aclose_flushes_even_when_clean():
    """⚠️ 跟 `end_utterance` 不同：**没写过也要关**。

    流可能是上一轮留下的；不关的话对端 reader 等不到结束标记，
    网关要等空闲回收才还槽位 —— 而槽位是稀缺的。
    """
    raw, s = sink()
    s.aclose()
    assert raw.calls == ["flush"]


def test_aclose_swallows_errors():
    class Boom(FakeRaw):
        def flush(self):
            self.calls.append("flush")
            raise RuntimeError("连接已经没了")

    raw = Boom()
    AvatarAudioSink(raw).aclose()              # 不该抛
    assert raw.calls == ["flush"]


# ── 计数 ─────────────────────────────────────────────────────────────


def test_frames_go_through_untouched():
    """这一层不缓冲、不改帧，只是记一个脏标记。"""
    raw, s = sink()
    for _ in range(5):
        asyncio.run(s.capture_frame(object()))
    assert raw.frames == 5


def test_raw_is_reachable():
    raw, s = sink()
    assert s.raw is raw


@pytest.mark.parametrize("method", ["interrupt", "end_utterance", "aclose"])
def test_no_method_raises_on_a_dead_sink(method):
    """出口这一路挂了，正确的降级是「这段没对上口型」，不是整轮播放崩掉。"""
    class Dead(FakeRaw):
        def flush(self):
            raise RuntimeError("流已经断了")

        def clear_buffer(self):
            raise RuntimeError("RPC 发不出去")

    s = AvatarAudioSink(Dead())
    s._dirty = True
    getattr(s, method)()                       # 三个都不该抛
