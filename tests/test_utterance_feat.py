"""整句编码 —— **不碰 GPU，假编码器。**

钉的是「切片和边界」，不是「特征准不准」。特征准不准由上游那两个函数决定，
我们只负责原封不动地调它们、把结果按块切对（见 `UtteranceAudioFeat` 的
类文档：不要重新实现帧映射，上一版就是栽在这儿）。
"""
import threading
import time

import numpy as np
import pytest
import torch

from worker.audio_stream import BlockGeometry, PcmInbox
from worker.audio_feat import UtteranceAudioFeat

G = BlockGeometry()
L, D = 25, 8            # 假编码器的层数 / 维度，小一点跑得快


class FakeEnc:
    """每 1/fps 秒一帧，第 i 帧的值恒等于 i —— 这样「切错块」立刻看得出来。"""

    def __init__(self):
        self.calls = []
        self.audio_sample_m = 0

    def extract_audio_feat_from_array(self, pcm, sample_rate=16000,
                                      return_all_layers=False, dtype=None):
        self.calls.append(len(pcm))
        return pcm                      # 原样带过去，下一步才生成特征

    def get_audio_embed_bucket_fps(self, z, fps=25, batch_frames=12, m=0):
        n = int(len(z) / 16000 * fps)
        eb = torch.arange(n, dtype=torch.float32).view(n, 1, 1).repeat(1, L, D)
        return eb, 1


def _pcm(seconds, value=1000):
    return np.full(int(G.sample_rate * seconds), value, dtype=np.int16)


# ── PcmInbox.pull_utterance ────────────────────────────────────────

def test_pull_utterance_waits_for_segment_end():
    box = PcmInbox(G)
    got = {}

    def worker():
        got["pcm"] = box.pull_utterance(max_s=30)

    t = threading.Thread(target=worker, daemon=True); t.start()
    box.push(_pcm(2.0)); time.sleep(0.15)
    assert "pcm" not in got, "整句还没说完就放行了 —— 那就退回边来边翻了"

    box.push(_pcm(1.0)); box.mark_segment_end()
    t.join(timeout=3)
    assert got["pcm"].shape[0] == int(G.sample_rate * 3.0)
    assert box.pending_samples == 0, "取走之后缓冲要清空，否则下一句会带上这一句"


def test_pull_utterance_hits_cap_without_segment_end():
    """上游忘了发段落结束时**不能挂死** —— 攒到上限就先翻这一截。"""
    box = PcmInbox(G)
    box.push(_pcm(2.0))
    out = box.pull_utterance(max_s=1.0)
    assert out.shape[0] == int(G.sample_rate * 2.0)


def test_pull_utterance_returns_float_in_range():
    box = PcmInbox(G)
    box.push(_pcm(0.5, value=32767)); box.mark_segment_end()
    out = box.pull_utterance()
    assert out.dtype == np.float32 and 0.9 < out.max() <= 1.0


def test_pull_utterance_unblocks_on_close():
    box = PcmInbox(G)
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("x", box.pull_utterance()),
                         daemon=True)
    t.start(); time.sleep(0.1); box.close(); t.join(timeout=3)
    assert got["x"].size == 0, "关了要放行，否则模型线程 join 不回来"


def test_second_utterance_is_independent():
    box = PcmInbox(G)
    box.push(_pcm(1.0)); box.mark_segment_end()
    a = box.pull_utterance()
    box.push(_pcm(2.0)); box.mark_segment_end()
    b = box.pull_utterance()
    assert a.shape[0] == int(G.sample_rate * 1.0)
    assert b.shape[0] == int(G.sample_rate * 2.0), "第二句带上了第一句的尾巴"


# ── UtteranceAudioFeat ─────────────────────────────────────────────

def _feat(utterances):
    """把几段 PCM 排成队，一次 pull 给一段。"""
    q = list(utterances)
    enc = FakeEnc()
    f = UtteranceAudioFeat(enc, pull_utterance=lambda *_: (
        q.pop(0) if q else np.zeros(0, dtype=np.float32)),
        fps=G.fps, sample_rate=G.sample_rate)
    return enc, f


def test_blocks_come_out_in_order():
    """第 k 块必须是第 k*12 .. k*12+11 帧。错位一格这里就红。"""
    enc, f = _feat([np.zeros(G.sample_rate * 2, dtype=np.float32)])   # 2 s → 50 帧
    for k in range(4):
        out = f.next_block(12)
        assert out.shape == (1, L, D, 12), out.shape
        want = list(range(k * 12, k * 12 + 12))
        assert out[0, 0, 0].tolist() == [float(v) for v in want], f"第 {k} 块错位"


def test_encodes_once_per_utterance_not_once_per_block():
    """整句只过一次编码器 —— 每块过一次就退回原样了（还慢 N 倍）。"""
    enc, f = _feat([np.zeros(G.sample_rate * 2, dtype=np.float32)])
    for _ in range(4):
        f.next_block(12)
    assert len(enc.calls) == 1, f"编码了 {len(enc.calls)} 次，应该只有 1 次"


def test_tail_block_pads_with_last_frame_not_zeros():
    """句尾不足一块：补最后一帧。补零会让嘴**突然闭死**。"""
    enc, f = _feat([np.zeros(int(G.sample_rate * 0.8), dtype=np.float32)])  # 20 帧
    f.next_block(12)
    out = f.next_block(12)              # 只剩 8 帧，要补 4 帧
    vals = out[0, 0, 0].tolist()
    assert vals[:8] == [float(v) for v in range(12, 20)]
    assert vals[8:] == [19.0] * 4, f"尾巴补的是 {vals[8:]}，应该是重复最后一帧"


def test_next_utterance_pulled_after_exhaustion():
    enc, f = _feat([np.zeros(int(G.sample_rate * 0.48), dtype=np.float32),
                    np.zeros(G.sample_rate * 2, dtype=np.float32)])
    f.next_block(12)                    # 第一句正好一块
    f.next_block(12)                    # 该去要第二句了
    assert len(enc.calls) == 2


def test_reset_drops_the_cached_utterance():
    """被打断：缓存作废，下一块必须重新去要音频。"""
    enc, f = _feat([np.zeros(G.sample_rate * 2, dtype=np.float32),
                    np.zeros(G.sample_rate * 2, dtype=np.float32)])
    f.next_block(12)
    f.reset()
    f.next_block(12)
    assert len(enc.calls) == 2, "打断之后还在用上一句的特征"


def test_empty_utterance_does_not_hang_or_throw():
    """拿到空音频（会话关了）也得返回一块东西 —— 抛出去五张卡一起死。"""
    enc, f = _feat([])
    out = f.next_block(12)
    assert out.shape == (1, 25, 1024, 12)
    assert float(out.abs().sum()) == 0.0
