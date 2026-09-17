"""PCM 缓冲 —— 纯 numpy，不碰 torch / livekit / asyncio。

这一层是**跨线程**的：事件循环写，模型线程读。所以除了「给的块对不对」，
还得测并发下不丢样本。
"""
import threading
import time

import numpy as np
import pytest

from worker.audio_stream import BlockGeometry, PcmInbox

G = BlockGeometry()


def i16(n: int, value: int = 1000) -> np.ndarray:
    return np.full(n, value, dtype=np.int16)


def make() -> PcmInbox:
    return PcmInbox(G)


# ── 块几何 ────────────────────────────────────────────────────────

def test_geometry_from_model_config_beats_hardcoding():
    """⭐ 常数从模型配置读。上游有两棵配置树，拿错那棵 fps 就是 16 不是 25。

    这里用一个最小的假 config 验证换算，重点是 `×4`（VAE 时间下采样率）
    别丢 —— 丢了每块从 12 帧变 3 帧，音频窗口整体错位。
    """
    class Cfg:
        sample_fps = 25
        num_frames_per_block = 3

    g = BlockGeometry.from_model_config(Cfg())
    assert (g.fps, g.frames_per_block) == (25, 12)
    assert g.block_seconds == pytest.approx(0.48)
    assert g.block_samples == 7680


def test_bad_geometry_refuses_to_construct():
    """错了不会报错、只会让口型差一截 —— 宁可起不来。"""
    with pytest.raises(ValueError):
        BlockGeometry(fps=0)


# ── 拉块 ──────────────────────────────────────────────────────────

def test_block_length_is_always_exact():
    """⭐ 无论缓冲里有多少，返回长度**恒为** block_samples。

    长度不固定的话模型侧编码窗口会错位 —— 不报错，只是口型跟声音差一截。
    """
    box = make()
    box.push(i16(160))                       # 远不足
    assert len(box.pull_block()) == G.block_samples
    box.push(i16(G.sample_rate * 5))         # 远超
    for _ in range(5):
        assert len(box.pull_block()) == G.block_samples


def test_int16_is_scaled_to_unit_float():
    """⭐ int16 必须转成 [-1,1] 的 float32。

    少这一步 wav2vec 的 processor 会把整数当振幅，特征完全不对 ——
    而它照样出视频，只是口型对不上。
    """
    box = make()
    box.push(i16(G.block_samples, value=32767))
    chunk = box.pull_block()
    assert chunk.dtype == np.float32
    assert 0.9 < chunk.max() <= 1.0, f"没归一化：max={chunk.max()}"


def test_partial_tail_is_padded_not_dropped():
    """句尾不足一块要补静音说完，不能吞掉。

    吞掉的话每句话结尾都缺一截，听起来像「他话没说完」。
    """
    box = make()
    box.push(i16(1000))
    chunk = box.pull_block()
    assert np.count_nonzero(chunk) == 1000
    assert len(chunk) == G.block_samples


def test_pcm_is_consumed_in_order_without_gaps():
    """跨多个 push 拼接时不能丢样本、不能重复。"""
    box = make()
    box.push(i16(4800, value=100))
    box.push(i16(4800, value=200))           # 合计 9600
    first = box.pull_block()
    assert np.count_nonzero(first) == G.block_samples, "第一块就出现空洞"
    second = box.pull_block()
    assert np.count_nonzero(second) == 9600 - G.block_samples, \
        f"第二块真样本数不对：{np.count_nonzero(second)}"


# ── ⭐ 阻塞：省电那一环 ────────────────────────────────────────────

def test_pull_blocks_while_silent():
    """⭐ 没货**必须阻塞**。

    返回静音的话整条五卡流水线会以 1.66× 实时空转生成闭嘴画面 ——
    五张 B200 全程满载，没人说话也烧。这条测的就是那个开销不存在。
    """
    box = make()
    done = threading.Event()
    threading.Thread(target=lambda: (box.pull_block(), done.set()),
                     daemon=True).start()
    assert not done.wait(0.3), "没货却立刻返回了 —— 会空转烧卡"
    box.push(i16(G.block_samples))
    assert done.wait(2.0), "来音频了却没被唤醒"


def test_timeout_falls_back_to_silence():
    """显式给 timeout 才返回静音 —— 留给「待机也要出画面」那种形态。"""
    t0 = time.monotonic()
    chunk = make().pull_block(timeout=0.2)
    assert not np.any(chunk)
    assert time.monotonic() - t0 >= 0.15


def test_close_releases_a_blocked_puller():
    """进程要退了得能收摊，否则卡在 join 上。"""
    box = make()
    done = threading.Event()
    threading.Thread(target=lambda: (box.pull_block(), done.set()),
                     daemon=True).start()
    time.sleep(0.1)
    box.close()
    assert done.wait(2.0), "close() 没把阻塞的拉取放出来"


# ── ⭐ 打断 ───────────────────────────────────────────────────────

def test_clear_drops_everything_and_reparks():
    """打断之后既不能还念上一句，也不能变成空转。"""
    box = make()
    box.push(i16(G.sample_rate * 3))
    box.clear()
    assert box.pending_samples == 0

    done = threading.Event()
    threading.Thread(target=lambda: (box.pull_block(), done.set()),
                     daemon=True).start()
    assert not done.wait(0.3), "清空后没有重新阻塞 —— 会空转"
    box.push(i16(G.block_samples, value=500))
    assert done.wait(2.0), "打断之后新的话进不来"


# ── ⭐ 跨线程 ─────────────────────────────────────────────────────

def test_no_samples_lost_under_concurrent_push_and_pull():
    """⭐ 一端事件循环写、一端模型线程读，同时动同一个缓冲。

    没锁的话 `pop(0)` 和 `buf[0] = head[take:]` 之间会撕裂 —— 丢样本或重复，
    **不报错**，只是口型偶尔跟声音错开。这条按总能量对账。
    """
    box = make()
    chunks, per = 200, 400
    total_pushed = chunks * per

    def writer():
        for _ in range(chunks):
            box.push(i16(per, value=1))
            time.sleep(0)                     # 让出，制造交错

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    seen = 0
    deadline = time.monotonic() + 10
    while seen < total_pushed and time.monotonic() < deadline:
        seen += int(np.count_nonzero(box.pull_block(timeout=0.5)))
    t.join(5)

    assert seen == total_pushed, \
        f"并发下样本对不上：推了 {total_pushed}，拉到 {seen}（差 {total_pushed - seen}）"
