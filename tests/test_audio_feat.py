"""音频特征窗口的**时间轴**对不对。不需要 torch、不需要模型。

## 这份测试守的是什么

整个 `audio_feat.py` 只为一件事存在：让流式那条路上「第 i 个视频帧」对上
「第 i/25 秒的声音」。做错了不会报错、不会掉帧、时长还完全对得上 ——
只会让嘴型差几十毫秒，而那个只有人眼能判。**所以判据必须是时间，不是形状。**

三条性质，任何一条破了口型就歪：

    ① 窗口落在合法格子上   → 插值出来的帧间距**精确**是 1/30 s
    ② 块内 12 个索引全在界内 → 没有哪一帧被喂「静音」
    ③ 每个索引的实际时刻贴着理想时刻 → 误差只剩训练时就有的取整量化

外加一条反向的：**用上游那种「一块自己编一次」的窗口，① 必然破** ——
不钉住这条，有人「简化」回去的时候三条测试可能还是全绿。
"""
import numpy as np
import pytest

from worker.audio_feat import (W2V_FIELD, W2V_FPS, block_indices, choose_window,  # noqa: F401
                               w2v_frames, window_grid)

SR = 16000
VIDEO_RATE = 30
FPS = 25
BLOCK_FRAMES = 12
BLOCK_SAMPLES = SR * BLOCK_FRAMES // FPS      # 7680


def spacing_ms(n_samples: int, out_len: int) -> float:
    """插值后相邻两个嵌入帧真实差多少毫秒。

    `align_corners=True`：输出第 0 帧钉输入第 0 帧、最后一帧钉最后一帧，
    所以每个输出帧跨 `(T-1)/(out_len-1)` 个 wav2vec 帧。
    """
    T = w2v_frames(n_samples)
    return (T - 1) / (out_len - 1) / W2V_FPS * 1000


# ── 锚点：卷积长度不是我推的，是跟真 config 对过的 ────────────────────

@pytest.mark.parametrize("n,expect", [(7680, 23), (16000, 49), (39680, 123),
                                      (279680, 873)])
def test_w2v_frames_matches_real_model(n, expect):
    """这四个数是在 b2 上用真的 `Wav2Vec2Config` 跑出来的。

    化简公式 `(n-400)//320+1` 如果哪天不成立（换了模型 / 换了卷积栈），
    整个窗口推导都塌了 —— 所以拿外部锚点钉住，不要只测公式自洽。
    """
    assert w2v_frames(n) == expect


# ── ① 帧间距必须精确 ─────────────────────────────────────────────────

@pytest.mark.parametrize("want_s", [0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
def test_chosen_window_gives_exact_frame_spacing(want_s):
    """⭐ 核心性质：选出来的窗口，帧间距**精确**等于 1/30 s。

    「差不多」不行 —— 0.2% 的偏差在 2 秒窗口上就是 4 ms，而块与块之间
    这个误差会重新累积成锯齿。用 exact 比较（整除关系），不用 approx。
    """
    W, out_len = choose_window(int(want_s * SR), VIDEO_RATE)
    T = w2v_frames(W)
    # (T-1)/(out_len-1) == 50/30，写成乘法避免浮点
    assert (T - 1) * VIDEO_RATE == (out_len - 1) * W2V_FPS
    assert spacing_ms(W, out_len) == pytest.approx(1000 / VIDEO_RATE, abs=1e-9)
    assert W >= want_s * SR


def test_chosen_window_is_the_smallest_valid_one():
    """别选过头 —— 窗口每长一点，wav2vec 每块就多算一点。"""
    step_w, _ = window_grid(VIDEO_RATE)
    for want in range(SR // 2, 3 * SR, 997):
        W, _ = choose_window(want, VIDEO_RATE)
        assert W >= want
        assert W - step_w < want           # 再小一格就不够了
        assert (W - W2V_FIELD) % step_w == 0


def block_drift_ms(n_samples: int, out_len: int) -> float:
    """帧间距偏差在**一块之内**累积成多少毫秒 —— 这才是人眼看到的东西。

    一块 0.48 s 跨约 `0.48×30 = 14.4` 个嵌入帧，每帧偏一点，块尾攒成一截；
    下一块又归零，于是是锯齿不是恒定偏移。恒定偏移人是看不出来的，锯齿能。
    """
    err = spacing_ms(n_samples, out_len) - 1000 / VIDEO_RATE
    return abs(err) * (BLOCK_SAMPLES / SR * VIDEO_RATE)


def test_round_seconds_window_is_wrong():
    """⭐ 反向钉子：**整秒窗口是错的**，别「顺手取整」。

    16000 个采样看着最自然，出来 49 帧，(49-1) 不是 5 的倍数 ——
    判据不看「帧间距差了几毫秒」（看着都很小），看**一块之内攒成多少**：
    合法窗口是 0，整秒窗口是好几毫秒到几十毫秒。
    这一条存在是因为「取个整数好看」是个非常容易犯的改动。
    """
    for bad in (SR // 2, SR, 2 * SR):
        T = w2v_frames(bad)
        out_len = int(T / W2V_FPS * VIDEO_RATE)
        assert (T - 1) * VIDEO_RATE != (out_len - 1) * W2V_FPS, f"{bad} 居然合法？"
        assert block_drift_ms(bad, out_len) > 5.0, (bad, block_drift_ms(bad, out_len))
    # 而合法窗口是**精确** 0，不是「小”
    for want in (0.5, 1.0, 1.5, 2.0):
        W, out_len = choose_window(int(want * SR), VIDEO_RATE)
        assert block_drift_ms(W, out_len) == pytest.approx(0.0, abs=1e-9)


def test_upstream_single_block_window_is_broken():
    """⭐ 把上游那个 bug 钉成测试：一块自己编一次，间距 36.667 ms。

    这不是在测我们的代码，是在**记住我们为什么要写这个模块**。
    哪天有人觉得「直接编一块不就行了」，这条会告诉他会发生什么。
    """
    T = w2v_frames(BLOCK_SAMPLES)                      # 23
    out_len = int(T / W2V_FPS * VIDEO_RATE)            # 13
    assert (T, out_len) == (23, 13)
    assert spacing_ms(BLOCK_SAMPLES, out_len) == pytest.approx(36.667, abs=0.01)
    # 时间轴被拉伸 10%
    assert spacing_ms(BLOCK_SAMPLES, out_len) / (1000 / VIDEO_RATE) == \
        pytest.approx(1.10, abs=1e-6)


# ── ② 没有哪一帧被喂静音 ─────────────────────────────────────────────

@pytest.mark.parametrize("look_s", [0.0, 0.3, 1.5, 3.0])
def test_no_index_falls_out_of_range(look_s):
    """越界的那一帧上游会填 `torch.zeros` —— 模型被告知「这一帧没声音」。

    上游每 12 帧就中 1 次，25 fps 下是 2 Hz 的嘴部抽动。
    """
    W, out_len = choose_window(BLOCK_SAMPLES + int(look_s * SR), VIDEO_RATE)
    idx = block_indices(W, BLOCK_SAMPLES, BLOCK_FRAMES, fps=FPS, sample_rate=SR,
                        video_rate=VIDEO_RATE)
    assert len(idx) == BLOCK_FRAMES
    assert all(0 <= i <= out_len - 1 for i in idx), (idx, out_len)


def test_upstream_single_block_does_fall_out_of_range():
    """⭐ 同样反向钉住：上游那条第 12 帧确实越界（索引 13 / 只有 0–12）。"""
    idx = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]     # b2 上从上游函数直接取的
    assert sum(1 for i in idx if i > 12) == 1


# ── ③ 每一帧对上的时刻要贴着理想时刻 ─────────────────────────────────

def test_indices_land_within_quantisation_of_ideal_time():
    """⭐ 真正的判据：第 i 帧实际对上的声音，跟 i/25 秒差多少。

    「实际时刻」按**离线的约定**算：第 j 个嵌入帧就在 j/30 秒
    （见 `block_indices` 里那段 —— 不补感受野的 12.5 ms，实测过）。

    嵌入帧只有 30 Hz，所以最好也只能做到 ±1/60 s ＝ ±16.7 ms ——
    **这跟离线是同一个量化**，目标是追平离线，不是超过它。
    超出这个界就说明索引算错了，不是量化的锅。
    """
    W, out_len = choose_window(BLOCK_SAMPLES + int(1.5 * SR), VIDEO_RATE)
    idx = block_indices(W, BLOCK_SAMPLES, BLOCK_FRAMES, fps=FPS, sample_rate=SR,
                        video_rate=VIDEO_RATE)
    base = (W - BLOCK_SAMPLES) / SR
    errs = [((j / VIDEO_RATE) - (base + i / FPS)) * 1000
            for i, j in enumerate(idx)]
    assert max(abs(e) for e in errs) <= 1000 / VIDEO_RATE / 2 + 1e-9, errs
    # 而且**不能有系统性漂移** —— 锯齿的特征就是首尾差一大截。
    assert abs(errs[-1] - errs[0]) <= 1000 / VIDEO_RATE, errs


def test_no_drift_across_many_blocks():
    """⭐ 跨 50 块看**绝对时间轴**：误差不许累积，也不许有锯齿。

    这是整份测试里唯一一条能抓到「块内拉伸 / 块间归零」的 —— 单看一块
    永远是对的，锯齿只有连着看才现形。上游那条在这里会从 +6 ms 一路爬到
    +59 ms、然后掉回 +6 ms，周期 0.48 s。

    界取 ±1/(2·30) ＝ ±16.7 ms：嵌入帧只有 30 Hz，**离线自己也有这个量化**
    （`round(1.2g)`），追平它就够，不追求比它更准。
    """
    W, _ = choose_window(BLOCK_SAMPLES + int(1.5 * SR), VIDEO_RATE)
    idx = block_indices(W, BLOCK_SAMPLES, BLOCK_FRAMES, fps=FPS, sample_rate=SR,
                        video_rate=VIDEO_RATE)
    block_s = BLOCK_SAMPLES / SR
    q = 1000 / VIDEO_RATE / 2
    errs = []
    for n in range(50):
        win_start = (n + 1) * block_s - W / SR
        for i, j in enumerate(idx):
            g = n * BLOCK_FRAMES + i
            errs.append(((win_start + j / VIDEO_RATE) - g / FPS) * 1000)
    assert max(abs(e) for e in errs) <= q + 1e-9, (min(errs), max(errs))
    # 第 1 块和第 50 块的误差分布必须一样 —— 有累积的话后面会整体偏走
    head, tail = errs[:BLOCK_FRAMES * 3], errs[-BLOCK_FRAMES * 3:]
    assert abs(np.mean(head) - np.mean(tail)) < 1.0, (np.mean(head), np.mean(tail))


def test_offline_has_the_same_quantisation_bound():
    """对照组：离线 `round(1.2g)` 的误差界跟我们一样。

    放这条是为了防止把量化误差当成我们的锅去「优化」—— 真去插值抹平它，
    等于喂给模型一种训练时没见过的平滑特征，是拿一个可量化的小问题
    换一个不可量化的大问题。
    """
    q = 1000 / VIDEO_RATE / 2
    errs = [(round(g * VIDEO_RATE / FPS) / VIDEO_RATE - g / FPS) * 1000
            for g in range(600)]
    assert max(abs(e) for e in errs) <= q + 1e-9, (min(errs), max(errs))


def test_indices_are_monotonic_and_evenly_stepped():
    """相邻两帧只能差 1 或 2 个嵌入帧（30/25 = 1.2）。

    差 0 意味着两帧看同一刻声音（口型卡住），差 ≥3 意味着跳过一段。
    """
    W, _ = choose_window(BLOCK_SAMPLES + int(1.5 * SR), VIDEO_RATE)
    idx = block_indices(W, BLOCK_SAMPLES, BLOCK_FRAMES, fps=FPS, sample_rate=SR,
                        video_rate=VIDEO_RATE)
    d = np.diff(idx)
    assert set(np.unique(d)).issubset({1, 2}), idx
    assert d.mean() == pytest.approx(VIDEO_RATE / FPS, abs=0.1)


def test_consecutive_blocks_are_continuous():
    """⭐ 块与块之间不能有缝，也不能重叠。

    上游那条的锯齿正是「块内累积、块间归零」—— 所以这里要跨块看：
    把第 n 块的最后一帧和第 n+1 块的第一帧放在**同一条绝对时间轴**上，
    间隔必须还是 1/25 s（±量化）。
    """
    W, _ = choose_window(BLOCK_SAMPLES + int(1.5 * SR), VIDEO_RATE)
    idx = block_indices(W, BLOCK_SAMPLES, BLOCK_FRAMES, fps=FPS, sample_rate=SR,
                        video_rate=VIDEO_RATE)
    block_s = BLOCK_SAMPLES / SR

    # 第 n 块窗口末端在绝对时刻 (n+1)*block_s，窗口起点因此是 (n+1)*block_s - W/SR
    def abs_time(n, i):
        return (n + 1) * block_s - W / SR + idx[i] / VIDEO_RATE

    gap = abs_time(1, 0) - abs_time(0, BLOCK_FRAMES - 1)
    assert gap == pytest.approx(1 / FPS, abs=1 / VIDEO_RATE), gap
