"""音画配对账本的单测。**不要 GPU、不要 torch、不要 livekit。**

这一层的全部错误都是**计数错误**，而计数错误在带 GPU 的环境里最难复现：
要占卡、等模型加载，还得凑出「队列正好满」这种时序。剥成纯逻辑之后，
那些情况就是几行 `for` 循环。

钉死的是**失败模式**，不是「跑通了没有」：

    错位不许静默        多吐/少吐一帧必须留下 ERROR
    丢就成对丢          帧和它那段音频一起消失，对账仍然平
    发布顺序 == 拉取顺序  中间那块先填满也不许插队
    每一帧都有下落      seen == 预热 + 丢掉 + 已发 + 在途
"""
from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from worker.av_ledger import AVLedger, Unit  # noqa: E402

FPB = 12


def feed(led: AVLedger, blocks: int, frames_per_block: int | None = None) -> None:
    """喂 `blocks` 块，每块跟着 n 帧（默认正好填满）。"""
    n = FPB if frames_per_block is None else frames_per_block
    for b in range(blocks):
        led.on_block_pulled(f"pcm{b}")
        for i in range(n):
            led.on_frame(f"f{b}.{i}")


def drain(led: AVLedger) -> list[Unit]:
    out = []
    while (u := led.pop_ready()) is not None:
        out.append(u)
    return out


# ── 正常流 ────────────────────────────────────────────────────────

def test_pairs_are_correct():
    led = AVLedger(frames_per_block=FPB)
    feed(led, 3)
    units = drain(led)
    assert [u.blk for u in units] == [0, 1, 2]
    for b, u in enumerate(units):
        assert u.pcm == f"pcm{b}"
        assert u.frames == [f"f{b}.{i}" for i in range(FPB)]
    assert led.reconciled and led.errors == 0


def test_publish_order_equals_pull_order_even_if_later_block_fills_first():
    """⭐ 中间那块先填满**也不许插队** —— 插队就是声音串到别的句子上。

    构造：先拉两块，再把帧全喂进去。按拉取顺序归属，块 0 先满，
    所以即便喂的过程中块 1 也满了，出来的顺序仍然必须是 0、1。
    """
    led = AVLedger(frames_per_block=FPB)
    led.on_block_pulled("A")
    led.on_block_pulled("B")
    for i in range(FPB * 2):
        led.on_frame(i)
    units = drain(led)
    assert [u.pcm for u in units] == ["A", "B"]
    assert units[0].frames == list(range(FPB))
    assert units[1].frames == list(range(FPB, FPB * 2))


def test_partially_filled_block_is_not_published():
    """帧没齐就不许发 —— 发了就等于把下一块的帧算给它。"""
    led = AVLedger(frames_per_block=FPB)
    led.on_block_pulled("A")
    for i in range(FPB - 1):
        led.on_frame(i)
    assert led.pop_ready() is None
    led.on_frame("last")
    assert led.pop_ready() is not None


# ── 预热帧 ────────────────────────────────────────────────────────

def test_prewarm_frames_are_discarded_not_attributed():
    """⭐ 一块都还没拉就吐出来的帧 = 上游用预置音频预热的，**必须扔掉**。

    混进第一块的话，从第一帧起就错位，而且之后每一块都错 ——
    这正是「冷启动那句嘴慢半拍」的成因之一。
    """
    led = AVLedger(frames_per_block=FPB)
    for i in range(30):                       # 预热吐了 30 帧
        led.on_frame(f"warm{i}")
    feed(led, 1)
    u = led.pop_ready()
    assert u is not None
    assert all(not str(f).startswith("warm") for f in u.frames), u.frames
    assert led.frames_prewarm == 30
    assert led.errors == 0                    # 预热是预期内的，不算错
    assert led.reconciled


# ── 错位必须吵 ────────────────────────────────────────────────────

def test_extra_frame_logs_error_and_is_not_silently_absorbed(caplog=None):
    """⭐⭐ 多吐一帧 —— **绝不能默默塞进下一块**。

    塞进去之后从此每块错开一帧，而且再也查不出来。这是整个方案唯一的
    脆弱点（「一块恰好 N 帧」是个假设），所以它必须留下 ERROR。
    """
    led = AVLedger(frames_per_block=FPB)
    feed(led, 1)                              # 块 0 填满
    before = led.frames_attributed
    led.on_frame("extra")                     # 多的一帧，此时没有块在等
    assert led.errors == 1, "多出来的帧必须记 ERROR"
    assert led.frames_attributed == before, "多出来的帧不许被归属给任何块"


def test_short_block_is_caught_at_segment_end():
    """⭐ 少吐帧要在句尾被抓住，并且**丢掉**。

    留着它，下一句的头几帧会被它吃进来 —— 那就从下一句开始全错，
    而且错的是「上一句的嘴配下一句的话」，最难看的那种。
    """
    led = AVLedger(frames_per_block=FPB)
    led.on_block_pulled("A")
    for i in range(FPB - 3):                  # 少 3 帧
        led.on_frame(i)
    stale = led.on_segment_end()
    assert stale == 1 and led.errors == 1
    assert led.pop_ready() is None            # 半块不许流出去
    # 下一句必须从干净状态开始
    feed(led, 1)
    u = led.pop_ready()
    assert u is not None and len(u.frames) == FPB
    assert all(isinstance(f, str) for f in u.frames), "上一句的残帧串进来了"
    assert led.reconciled


def test_segment_end_on_clean_state_is_silent():
    """对齐的时候句尾不许报错 —— 会叫的监控只有在真出事时叫才有用。"""
    led = AVLedger(frames_per_block=FPB)
    feed(led, 2)
    drain(led)
    assert led.on_segment_end() == 0
    assert led.errors == 0


# ── 背压：丢就成对丢 ──────────────────────────────────────────────

def test_backpressure_drops_whole_blocks_and_keeps_books_balanced():
    """⭐⭐ 积压时丢的是**整块**（帧 + 它那段音频），不是单帧。

    只丢视频不丢音频的话，每丢一帧对应关系就永久错开一帧，
    而按秒数配平会自动「重新平衡」，**把错位藏起来**。
    """
    led = AVLedger(frames_per_block=FPB, max_pending_blocks=3)
    feed(led, 10)                             # 远超上限，不取走
    assert led.blocks_dropped > 0
    assert led.frames_dropped % FPB == 0, "丢的不是整块 —— 边界没落在块上"

    units = drain(led)
    # ⚠️ 这一条不能省：按单帧丢的实现会让队头永远填不满，于是**一块都发不出来**，
    #    上面那些 for 循环全部空转、测试假绿。问过「有没有东西活下来」才算数。
    assert units, "一块都没发出来 —— 队头被半块卡住了"
    assert len(units) <= 3
    for u in units:
        b = u.pcm.removeprefix("pcm")
        assert len(u.frames) == FPB, f"块 {u.pcm} 只剩 {len(u.frames)} 帧 —— 被按帧丢过"
        assert u.frames == [f"f{b}.{i}" for i in range(FPB)], \
            f"块 {u.pcm} 的帧不是它自己的：{u.frames[:3]}"
    assert led.reconciled


def test_dropped_blocks_never_break_pairing_of_survivors():
    """丢掉中间的块之后，剩下的**仍然各配各的**，不许错位一格。"""
    led = AVLedger(frames_per_block=FPB, max_pending_blocks=2)
    feed(led, 6)
    units = drain(led)
    assert units, "一块都没发出来"
    for u in units:
        b = u.pcm.removeprefix("pcm")
        assert len(u.frames) == FPB
        assert u.frames[0] == f"f{b}.0", f"块 {u.pcm} 配到了别人的帧：{u.frames[0]}"

    # 边取边喂：这才是线上的样子（消费者在跑，偶尔跟不上）。
    # 按单帧丢的实现在这里会露馅 —— 某一块会少几帧还照样被发出去。
    led2 = AVLedger(frames_per_block=FPB, max_pending_blocks=2)
    got = []
    for b in range(12):
        led2.on_block_pulled(f"pcm{b}")
        for i in range(FPB):
            led2.on_frame(f"f{b}.{i}")
        if b % 3 == 0:
            got += drain(led2)
    got += drain(led2)
    assert got, "边取边喂一块都没出来"
    for u in got:
        b = u.pcm.removeprefix("pcm")
        assert u.frames == [f"f{b}.{i}" for i in range(FPB)], \
            f"块 {u.pcm} 的帧被动过：{u.frames[:3]}…（{len(u.frames)} 帧）"


# ── 打断 ──────────────────────────────────────────────────────────

def test_clear_drops_everything_and_keeps_block_numbers_monotonic():
    """打断要清干净；**块号不许重置** —— 重置后打断前后同号，日志对不上。"""
    led = AVLedger(frames_per_block=FPB)
    feed(led, 2)
    led.on_block_pulled("half")
    led.on_frame("x")
    last = led._next_blk
    led.clear()
    assert led.pop_ready() is None
    assert led.on_segment_end() == 0          # 清完之后没有残留
    blk = led.on_block_pulled("after")
    assert blk >= last, "块号被重置了 —— 打断前后会同号"
    assert led.reconciled


# ── 对账 ──────────────────────────────────────────────────────────

def test_every_frame_is_accounted_for_in_a_messy_run():
    """⭐ 一趟乱七八糟的流程走完，**每一帧都要有下落**。

    对账等式是这套方案能被信任的唯一依据 —— 它一旦不成立，
    「配对正确」就没人能证明。
    """
    led = AVLedger(frames_per_block=FPB, max_pending_blocks=3)
    for _ in range(15):                       # 预热
        led.on_frame("warm")
    feed(led, 4)
    drain(led)
    feed(led, 8)                              # 触发背压
    led.on_block_pulled("short")
    for i in range(5):
        led.on_frame(i)
    led.on_segment_end()
    feed(led, 2)
    drain(led)
    led.clear()
    assert led.reconciled, led.stats()


def test_a_vanished_frame_makes_the_books_not_balance():
    """⭐⭐ 账本挡不住**外部**丢帧，但对账等式必须能发现它。

    这正是旧实现的病：帧队列满了偷偷丢最旧的一帧，音频一帧不丢 ——
    对应关系永久错开，而按秒数配平会自动「重新平衡」把它藏起来。
    账本不承诺阻止，只承诺**不可能悄悄发生**。
    """
    led = AVLedger(frames_per_block=FPB)
    led.on_block_pulled("A")
    for i in range(FPB - 2):
        led.on_frame(i)
    assert led.reconciled
    led._pending[0].frames.pop()              # 模拟：一帧凭空没了
    assert not led.reconciled, "帧丢了而账还是平的 —— 对账等式失效，整套配对不可信"


def test_stats_exposes_the_thing_that_matters():
    """指标必须能回答「配对还对不对」，不是「发了几秒」。

    现有那个「音频最多领先 0 ms」的监控就是反面教材：它数的是秒数，
    相等就报 0，**量错了对象，所以永远是绿的**。
    """
    led = AVLedger(frames_per_block=FPB)
    feed(led, 1)
    s = led.stats()
    for k in ("blocks_pulled", "frames_seen", "frames_dropped", "errors", "reconciled"):
        assert k in s, f"缺了 {k}"
    assert s["reconciled"] is True


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)         # 预期内的 ERROR 不刷屏
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  ✅", fn.__name__)
        except AssertionError as e:
            bad += 1
            print("  ❌", fn.__name__, "—", e)
    print(f"\n{'=' * 56}\n通过 {len(fns) - bad} 条，失败 {bad} 条")
    sys.exit(1 if bad else 0)
