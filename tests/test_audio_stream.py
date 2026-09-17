"""音频流式化调度的测试。

最有价值的一条不是「我算的对不对」，而是**跟上游对拍**：
把上游 `get_sample_indices(fixed_start=0)` 用 numpy 原样复刻一遍，
逐点比对 `slot_frame(i)`。对上了，就说明「第 i 个槽位取哪一帧只跟 i 有关」
这条推论是真的 —— 而整个流式改造都架在它上面。

其余的钉两件事：
  · 没到齐的音频**绝不能**提前开一轮（开了就要拿裁剪值凑，口型静默错位）
  · 那处**有意偏离上游**的 ceil，得有测试拦着别被人「修」回去
"""
import math

import numpy as np
import pytest

from worker.audio_stream import BucketGeometry, seconds_to_embed_frames

G = BucketGeometry()          # 默认即生产值


# ── 先把常数钉住：它们全是从上游源码抄的，改了要有人知道 ────────────────

def test_constants_match_upstream():
    assert (G.video_rate, G.fps, G.infer_frames, G.audio_sample_m) == (30, 16, 80, 0)
    assert G.scale == 30 / 16 == 1.875
    assert G.frames_per_repeat == 150


def test_external_anchor_five_seconds_in_five_seconds_out():
    """⭐ 外部锚点：一轮吃的音频时长 == 一轮出的视频时长。

    音频侧 150 帧 ÷ 30 Hz = 5 s；视频侧 80 帧 ÷ 16 fps = 5 s。
    这两个数走的是完全不同的公式，能对上说明整套几何没算歪。
    **孤立的数字最危险，这一条就是它的解药。**
    """
    audio_seconds = G.frames_per_repeat / G.video_rate
    video_seconds = G.infer_frames / G.fps
    assert audio_seconds == video_seconds == 5.0


def test_stride_is_truncated_not_rounded():
    # 上游写的是 int(video_rate/fps) —— 30/16=1.875 截断成 1。
    # 改成 round 会变 2，m>0 时取的上下文窗口整个错位。
    assert G.stride == 1
    assert BucketGeometry(video_rate=30, fps=8).stride == 3


# ── ⭐ 跟上游对拍：复刻 get_sample_indices，逐点比 ──────────────────────

def _upstream_batch_idx(geom: BucketGeometry, audio_frame_num: int) -> np.ndarray:
    """原样复刻 audio_encoder.py 里那两段（:176-:186），只固定 fixed_start=0。"""
    scale = geom.video_rate / geom.fps
    min_batch_num = int(audio_frame_num / (geom.infer_frames * scale)) + 1
    bucket_num = min_batch_num * geom.infer_frames
    padd = math.ceil(min_batch_num * geom.infer_frames / geom.fps
                     * geom.video_rate) - audio_frame_num
    total_frames = audio_frame_num + padd

    # get_sample_indices，fixed_start=0 分支
    required_duration = bucket_num / geom.fps
    assert required_duration <= total_frames / geom.video_rate, "上游会在这里抛"
    time_points = np.linspace(0, required_duration, bucket_num, endpoint=False)
    idx = np.round(np.array(time_points) * geom.video_rate).astype(int)
    return np.clip(idx, 0, total_frames - 1)


@pytest.mark.parametrize("audio_frame_num", [1, 30, 149, 150, 151, 300, 451, 3000])
def test_slot_frame_matches_upstream_pointwise(audio_frame_num):
    """⭐ 整个流式改造的立足点：槽位 → 帧 的映射与总长无关。

    对拍方式是把上游那段算出来的 batch_idx 整个数组拿来，逐点跟
    `slot_frame(i)` 比。注意**上游会裁剪到 total_frames-1**，而我们不裁 ——
    所以只比较没被裁到的那部分；裁剪本身由 repeats_ready 负责。
    """
    want = _upstream_batch_idx(G, audio_frame_num)
    total = audio_frame_num + (
        math.ceil(len(want) / G.fps * G.video_rate) - audio_frame_num)
    for i, w in enumerate(want):
        if w >= total - 1:
            continue                     # 被上游裁掉的尾巴，不比
        assert G.slot_frame(i) == w, f"槽位 {i}: 我们 {G.slot_frame(i)} vs 上游 {w}"


def test_slot_frame_does_not_depend_on_total_length():
    """同一个槽位，在不同总长下必须取同一帧 —— 这是流式成立的充要条件。"""
    a = _upstream_batch_idx(G, 200)
    b = _upstream_batch_idx(G, 5000)
    n = min(len(a), len(b))
    assert list(a[:n]) == list(b[:n])


# ── 什么时候可以开一轮 ────────────────────────────────────────────────

def test_frames_needed_for_first_repeat():
    # 第 0 轮用槽位 0..79，最后一个取 round(79*1.875)=round(148.125)=148，
    # 所以要 149 帧。**不是 150** —— 差这一帧就会白等 1/30 秒。
    assert G.slot_frame(79) == 148
    assert G.frames_needed_for_repeat(0) == 149


def test_frames_needed_is_strictly_increasing():
    need = [G.frames_needed_for_repeat(r) for r in range(20)]
    assert need == sorted(need) and len(set(need)) == len(need)


def test_m_context_widens_the_requirement():
    """m>0 时每个槽位还要往后看 m*stride 帧，门槛要跟着抬。

    生产值是 m=0，但上游签名里它是参数 —— 哪天调了而这里没跟上，
    现象是「偶尔一两帧口型对不上」，几乎不可能被发现。
    """
    g2 = BucketGeometry(audio_sample_m=2)
    assert g2.frames_needed_for_repeat(0) == G.frames_needed_for_repeat(0) + 2 * G.stride


def test_never_starts_a_repeat_one_frame_early():
    """⭐ 少一帧都不许开。

    提前开的话，那个槽位要么拿到裁剪后的旧帧、要么拿到零 —— 两种都是
    **静默错误**：视频照出，口型跟声音错开，没有任何报错。
    """
    need = G.frames_needed_for_repeat(0)
    assert G.repeats_ready(need - 1, ended=False) == 0
    assert G.repeats_ready(need, ended=False) == 1


def test_repeats_ready_is_monotonic_in_arrival():
    prev = 0
    for n in range(0, 1200, 7):
        cur = G.repeats_ready(n, ended=False)
        assert cur >= prev
        prev = cur


def test_nothing_ready_before_any_audio():
    assert G.repeats_ready(0, ended=False) == 0
    # ⭐ 流结束但一帧音频都没有 → 0 轮，不是 1 轮静音。
    #    上游公式在这里会给 1（int(0/150)+1），照抄的话数字人会对着空气动 5 秒。
    assert G.repeats_ready(0, ended=True) == 0


# ── 有意偏离上游的那一处 ──────────────────────────────────────────────

def test_deliberate_divergence_ceil_vs_plus_one():
    """⭐ 这条测试的作用是**拦住「修复」**。

    N 正好是一轮的整数倍时，上游多给一轮纯静音。直播里那是数字人
    对着空气动 5 秒嘴，所以我们用 ceil。差异是有意的。
    """
    assert G.upstream_num_repeat(150) == 2          # 上游：第 2 轮全是零填充
    assert G.repeats_for_finished_stream(150) == 1  # 我们：到此为止
    assert G.repeats_for_finished_stream(151) == 2  # 多一帧就真的要第 2 轮


@pytest.mark.parametrize("n", [1, 2, 149, 150, 151, 299, 300, 301, 1000, 4501])
def test_divergence_is_never_more_than_one_repeat(n):
    """偏离必须是「恰好少 0 或 1 轮」。差出别的数说明哪里算歪了，不是设计。"""
    diff = G.upstream_num_repeat(n) - G.repeats_for_finished_stream(n)
    assert diff in (0, 1), f"N={n} 差了 {diff} 轮"


@pytest.mark.parametrize("n", [1, 75, 149, 150, 151, 450, 3000])
def test_finished_stream_covers_all_audio(n):
    """⭐ 偏离不能偏出「漏掉音频」—— 下一轮的第一个槽位必须已经越过音频尾。

    判据写对很关键。第一版我写的是「最后取到的帧 ≥ N-1」，**错的**：
    桶是以 16 Hz 从 30 Hz 的嵌入流里抽样的，**本来就不是每一帧都会被取到**。
    N=3000 时最后一个槽位落在第 2998 帧，2999 帧根本不在任何槽位上 ——
    那是降采样的固有行为，不是漏掉了音频。

    正确的判据是「再多一轮的话，它的第一个槽位已经在音频之外」：

        slot_frame(r * infer_frames) >= N

    而 `slot_frame(r*80) = round(r*150) = r*150`（整数，不会有舍入歧义），
    所以这条等价于 `r*150 >= N` —— `ceil` 恰好是满足它的最小值。
    """
    r = G.repeats_for_finished_stream(n)
    next_slot_frame = G.slot_frame(r * G.infer_frames)
    assert next_slot_frame >= n, f"N={n} 只转了 {r} 轮，还有音频没被取样"


@pytest.mark.parametrize("n", [1, 75, 149, 150, 151, 450, 3000])
def test_finished_stream_is_minimal(n):
    """⭐ 也不能多转。少一轮就必须盖不住 —— 否则就是白烧一轮 GPU。

    跟上面那条合起来，把 `ceil` 夹成唯一解：多一轮浪费，少一轮吞句尾。
    """
    r = G.repeats_for_finished_stream(n)
    if r > 1:
        assert G.slot_frame((r - 1) * G.infer_frames) < n, f"N={n} 转 {r-1} 轮就够了"


def test_ended_flushes_the_tail():
    """流结束时，那些「还差几帧」的轮次也要放出来（尾巴按上游语义填零）。"""
    n = G.frames_needed_for_repeat(0) - 1        # 差一帧
    assert G.repeats_ready(n, ended=False) == 0
    assert G.repeats_ready(n, ended=True) == 1


# ── 输入校验：宁可炸，不要算出一个错的数 ───────────────────────────────

@pytest.mark.parametrize("kw", [{"fps": 0}, {"video_rate": 0},
                                {"infer_frames": 0}, {"audio_sample_m": -1}])
def test_bad_geometry_raises(kw):
    with pytest.raises(ValueError):
        BucketGeometry(**kw)


@pytest.mark.parametrize("call", [
    lambda: G.slot_frame(-1),
    lambda: G.frames_needed_for_repeat(-1),
    lambda: G.repeats_ready(-1, ended=False),
    lambda: G.repeats_for_finished_stream(-1),
    lambda: G.upstream_num_repeat(-1),
    lambda: seconds_to_embed_frames(-0.1, G),
])
def test_negative_inputs_raise(call):
    # ⭐ 这类参数错了如果只是算出个负数往下传，最终表现是「视频少了几帧」——
    #    没人会把它追回到这里。炸掉才有人查。
    with pytest.raises(ValueError):
        call()


def test_seconds_conversion_uses_embed_rate_not_pcm_rate():
    """秒 → 嵌入帧走 30 Hz。**拿 16000 或 48000 去换算会差三个数量级**，
    而现象只是「怎么一直不出帧」。
    """
    assert seconds_to_embed_frames(1.0, G) == 30
    assert seconds_to_embed_frames(5.0, G) == 150      # 正好一轮


# ── 取槽位：超出已有音频的部分要填零 ──────────────────────────────

def _embed(n, dim=4):
    """(1, n, dim)，第 i 帧全是 i+1，方便看取到了哪一帧。"""
    a = np.zeros((1, n, dim), dtype=np.float32)
    for i in range(n):
        a[0, i, :] = i + 1
    return a


def test_gather_takes_the_right_frames():
    from worker.audio_stream import gather_slots
    out = gather_slots(_embed(300), 0, G)
    assert out.shape == (1, G.infer_frames, 4)
    # 第 0 个槽位取第 0 帧（值 1），第 79 个取第 148 帧（值 149）
    assert out[0, 0, 0] == 1
    assert out[0, 79, 0] == 149 == G.slot_frame(79) + 1


def test_gather_zero_fills_past_the_end():
    """⭐ 超出已有长度的槽位填零 —— 上游 audio_encoder.py:209-212 的 else 分支。

    这一半最容易漏：我第一版只做了「什么时候能开一轮」，取槽位直接拿索引，
    冲句尾那一轮立刻 IndexError。两者是同一个契约的两半。
    """
    from worker.audio_stream import gather_slots
    out = gather_slots(_embed(30), 0, G)          # 只有 30 帧，远不够一轮
    # slot_frame(i) < 30 的那些槽位有值，其余全零
    filled = [k for k in range(G.infer_frames) if out[0, k, 0] != 0]
    assert filled == [k for k in range(G.infer_frames) if G.slot_frame(k) < 30]
    assert out[0, G.infer_frames - 1, 0] == 0, "尾巴没填零"


def test_gather_does_not_clamp_to_the_last_frame():
    """⭐ **不能改成裁剪到最后一帧。**

    裁剪的话句尾最后一个音素会被拖长成半秒，嘴型定在那儿不动 ——
    比填零（闭嘴）难看得多，而且同样不报错。
    """
    from worker.audio_stream import gather_slots
    out = gather_slots(_embed(30), 0, G)
    assert out[0, G.infer_frames - 1, 0] != 30, "裁剪到了最后一帧，不是填零"


def test_gather_second_repeat_offsets_correctly():
    from worker.audio_stream import gather_slots
    out = gather_slots(_embed(400), 1, G)
    assert out[0, 0, 0] == G.slot_frame(G.infer_frames) + 1 == 151
