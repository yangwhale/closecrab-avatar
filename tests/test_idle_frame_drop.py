"""空转帧不许发 —— **纯逻辑，不碰 GPU、不 import torch 之外的东西。**

背景：utterance 模式下没人说话时上游照样要音频块（不给就五卡死锁），
那些「空转块」照样会生成画面。发出去的话视频凭空变长（实测 12 s 音频
出 57.8 s 视频），而音频是按视频节奏放的，于是又不同步。

这里只测归类逻辑：第 n 块对应第 n 组 frames_per_block 帧。
"""
import collections

import pytest


class _Classifier:
    """把 `LiveAvatarPipelineSource` 里那三行单拎出来测。

    ⚠️ 这是**抄一份**不是调真的 —— 真的那个要 torch/livekit。抄的代价是
    两边可能走散，所以两处都写了同一句话当锚点：「第 n 块对应第 n 组
    frames_per_block 帧」。改一处要搜另一处。
    """

    def __init__(self, fpb):
        self.fpb = fpb
        self._blocks = collections.deque()
        self._kind = True
        self._left = 0

    def on_block(self, is_real):
        self._blocks.append(is_real)

    def is_real_frame(self):
        if self._left <= 0:
            self._kind = self._blocks.popleft() if self._blocks else True
            self._left = self.fpb
        self._left -= 1
        return self._kind


def _run(kinds, frames, fpb=12):
    c = _Classifier(fpb)
    for k in kinds:
        c.on_block(k)
    return [c.is_real_frame() for _ in range(frames)]


def test_real_blocks_all_pass():
    assert _run([True, True], 24) == [True] * 24


def test_idle_blocks_all_dropped():
    assert _run([False, False], 24) == [False] * 24


def test_boundary_is_exactly_frames_per_block():
    """一块管 12 帧，第 13 帧归下一块。**错一格这里就红。**"""
    got = _run([True, False], 24)
    assert got[:12] == [True] * 12
    assert got[12:] == [False] * 12


def test_alternating_blocks():
    got = _run([True, False, True], 36)
    assert got[0:12] == [True] * 12
    assert got[12:24] == [False] * 12
    assert got[24:36] == [True] * 12


def test_prewarm_frames_pass_when_no_block_reported_yet():
    """预热那两轮不调回调，队列是空的 —— 那几帧按「真」放行。

    它们随后会被句首抽干清掉。**不能按「假」丢**，否则真有音频那几块
    的帧会被这批预热帧顶掉一部分，对应关系整体前移。
    """
    c = _Classifier(12)
    assert all(c.is_real_frame() for _ in range(24))     # 预热 24 帧
    c.on_block(True)
    assert all(c.is_real_frame() for _ in range(12))


def test_idle_then_speech_recovers_alignment():
    """长时间空转之后来一句话：这句话的帧必须一帧不少地放行。"""
    got = _run([False] * 50 + [True] * 3, 50 * 12 + 36)
    assert got[:600] == [False] * 600
    assert got[600:] == [True] * 36


@pytest.mark.parametrize("fpb", [1, 3, 12, 48])
def test_works_at_any_block_size(fpb):
    got = _run([True, False], fpb * 2, fpb=fpb)
    assert got[:fpb] == [True] * fpb and got[fpb:] == [False] * fpb
