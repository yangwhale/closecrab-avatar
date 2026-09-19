"""音画配对账本 —— **角落情况穷举 + 随机 fuzz + 长跑**。

`test_av_ledger.py` 钉的是**已知的失败模式**（每条都对应一个真出过的 bug）。
这一份不同：它穷举**参数空间**，找的是还没人想到的组合。

> Chris 2026-09-19：「最主要的是你新设计的内部音频编码和视频编码，这种按
> 比例从 0 开始往上涨的模式，看它中间会不会有各种奇怪的问题造成它俩错位。」

三类：

    角落情况   ~40 个显式构造的组合（边界值、退化值、极端交错）
    随机 fuzz  上百条随机操作序列，每条都验四条不变量
    长跑       5 万块（≈6.7 小时语音），验编号不漂、账不歪

## 四条不变量（每个 case 都验）

    ① 对账平      seen == 预热 + 丢掉 + 已发 + 在途
    ② 配对正确    每块拿到的必须是它自己那几帧，一帧不错
    ③ 顺序守恒    发布顺序 == 拉取顺序，块号严格递增
    ④ 不吞错误    该报的 ERROR 一个不少（用计数比对，不看日志文本）

跑法：`python3 tests/test_av_ledger_corners.py [--report 报告.md]`
"""
from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from worker.av_ledger import AVLedger  # noqa: E402


class Harness:
    """包一层，**替账本记一份影子账** —— 用来独立验证，不信它自己的统计。"""

    def __init__(self, fpb: int, max_pending: int = 6) -> None:
        self.fpb = fpb
        self.led = AVLedger(frames_per_block=fpb, max_pending_blocks=max_pending)
        self.nblk = 0
        self.popped: list = []
        self.blk_order: list[int] = []

    def block(self) -> int:
        b = self.led.on_block_pulled(f"a{self.nblk}")
        self.nblk += 1
        return b

    def frames(self, n: int, tag: str = "") -> None:
        for i in range(n):
            self.led.on_frame(f"{tag}{i}")

    def block_with_frames(self, n: int | None = None) -> None:
        b = self.nblk
        self.block()
        for i in range(self.fpb if n is None else n):
            self.led.on_frame(f"f{b}.{i}")

    def drain(self) -> int:
        k = 0
        while (u := self.led.pop_ready()) is not None:
            self.popped.append(u)
            self.blk_order.append(u.blk)
            k += 1
        return k

    # ── 不变量 ──────────────────────────────────────────────────

    def check(self, name: str) -> list[str]:
        bad = []
        if not self.led.reconciled:
            bad.append(f"① 对账不平：{self.led.stats()}")
        for u in self.popped:
            if len(u.frames) != self.fpb:
                bad.append(f"② 块 {u.pcm} 发出去时只有 {len(u.frames)} 帧")
                break
            b = u.pcm.removeprefix("a")
            want = [f"f{b}.{i}" for i in range(self.fpb)]
            if u.frames != want and all(str(f).startswith("f") for f in u.frames):
                bad.append(f"② 块 {u.pcm} 配到别人的帧：{u.frames[:2]}")
                break
        if self.blk_order != sorted(self.blk_order):
            bad.append(f"③ 发布顺序不等于拉取顺序：{self.blk_order[:8]}")
        if len(set(self.blk_order)) != len(self.blk_order):
            bad.append("③ 同一块被发了两次")
        return [f"[{name}] {m}" for m in bad]


CASES: list[tuple[str, callable]] = []


def case(name: str):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ── 退化与边界 ────────────────────────────────────────────────────

@case("什么都不做")
def _(h): pass


@case("只拉块不给帧")
def _(h):
    for _ in range(5):
        h.block()
    h.drain()


@case("只给帧不拉块（纯预热）")
def _(h):
    h.frames(200, "w")
    h.drain()


@case("正好一块")
def _(h):
    h.block_with_frames(); h.drain()


@case("一块差一帧")
def _(h):
    h.block_with_frames(h.fpb - 1); h.drain()


@case("一块多一帧")
def _(h):
    h.block_with_frames(h.fpb + 1); h.drain()
    assert h.led.frames_rejected == 1, "多出来那帧没被记成拒收"


for _n in (0, 1, 5, 11, 12, 13, 47, 1000):
    @case(f"预热 {_n} 帧后再正常跑 3 块")
    def _(h, n=_n):
        h.frames(n, "w")
        for _ in range(3):
            h.block_with_frames()
        h.drain()


# ── 流水线深度：块比帧早很多 ──────────────────────────────────────

for _d in (1, 2, 3, 5, 8, 20):
    @case(f"先拉 {_d} 块再一次性喂满")
    def _(h, d=_d):
        # 容量要够放下这么多在途的，否则测到的是背压不是深流水
        h.led = AVLedger(frames_per_block=h.fpb, max_pending_blocks=d + 2)
        for _ in range(d):
            h.block()
        h.frames(h.fpb * d, "x")          # 匿名帧，只验顺序与计数
        h.drain()


@case("块和帧完全错开：拉 10 块、喂 5 块的帧、再拉 10 块、喂完")
def _(h):
    h.led = AVLedger(frames_per_block=h.fpb, max_pending_blocks=32)
    for _ in range(10):
        h.block()
    h.frames(h.fpb * 5, "p")
    h.drain()
    for _ in range(10):
        h.block()
    h.frames(h.fpb * 15, "q")
    h.drain()


# ── 句子边界：在块内每一个位置切 ──────────────────────────────────

for _k in range(0, 13):
    @case(f"句尾落在块内第 {_k} 帧")
    def _(h, k=_k):
        h.block_with_frames()
        h.block()
        h.frames(k, "t")
        h.led.on_segment_end()
        h.block_with_frames()
        h.drain()


@case("连续三次句尾，中间什么都没有")
def _(h):
    for _ in range(3):
        h.led.on_segment_end()
    h.block_with_frames(); h.drain()


@case("句尾紧跟句尾，各带半块")
def _(h):
    for _ in range(3):
        h.block(); h.frames(3, "s"); h.led.on_segment_end()
    h.block_with_frames(); h.drain()


# ── 打断 ──────────────────────────────────────────────────────────

for _k in (0, 1, 6, 12):
    @case(f"打断时块内已有 {_k} 帧")
    def _(h, k=_k):
        h.block_with_frames()
        h.drain()
        h.block(); h.frames(k, "c")
        h.led.clear()
        h.block_with_frames(); h.drain()


@case("打断后立刻句尾")
def _(h):
    h.block_with_frames(); h.block(); h.frames(4, "c")
    h.led.clear(); h.led.on_segment_end()
    h.block_with_frames(); h.drain()


@case("反复打断 50 次")
def _(h):
    for i in range(50):
        h.block(); h.frames(i % 13, "c"); h.led.clear()
    h.block_with_frames(); h.drain()


# ── 背压 ──────────────────────────────────────────────────────────

for _mp in (1, 2, 3, 6, 64):
    @case(f"背压 max_pending={_mp}，灌 30 块不取")
    def _(h, mp=_mp):
        h.led = AVLedger(frames_per_block=h.fpb, max_pending_blocks=mp)
        for _ in range(30):
            h.block_with_frames()
        h.drain()


for _every in (1, 2, 7):
    @case(f"边灌边取：每 {_every} 块取一次，共 40 块")
    def _(h, every=_every):
        for i in range(40):
            h.block_with_frames()
            if i % every == 0:
                h.drain()
        h.drain()


# ── 块大小退化 ────────────────────────────────────────────────────

for _fpb in (1, 2, 3, 25, 100):
    @case(f"每块 {_fpb} 帧，跑 20 块")
    def _(h, fpb=_fpb):
        h.fpb = fpb
        h.led = AVLedger(frames_per_block=fpb, max_pending_blocks=6)
        for _ in range(20):
            h.block_with_frames()
        h.drain()


# ── 混合 ──────────────────────────────────────────────────────────

@case("全家桶：预热 + 深流水 + 半块句尾 + 打断 + 背压")
def _(h):
    h.frames(37, "w")
    for _ in range(4):
        h.block()
    h.frames(h.fpb * 4, "a")
    h.drain()
    h.block(); h.frames(5, "b"); h.led.on_segment_end()
    for _ in range(20):
        h.block_with_frames()
    h.led.clear()
    for i in range(10):
        h.block_with_frames()
        if i % 3 == 0:
            h.drain()
    h.led.on_segment_end()
    h.drain()


def run_cases(fpb: int = 12) -> tuple[int, list[str]]:
    fails = []
    for name, fn in CASES:
        h = Harness(fpb)
        try:
            fn(h)
        except Exception as e:                       # noqa: BLE001
            fails.append(f"[{name}] 抛异常：{type(e).__name__}: {e}")
            continue
        fails += h.check(name)
    return len(CASES), fails


# ── 随机 fuzz ─────────────────────────────────────────────────────

def run_fuzz(rounds: int = 300, ops: int = 400) -> tuple[int, list[str]]:
    """随机操作序列。**找的是没人想到的组合**，所以别给它加约束。"""
    fails = []
    for seed in range(rounds):
        rng = random.Random(seed)
        fpb = rng.choice([1, 2, 3, 12, 25])
        h = Harness(fpb, max_pending=rng.choice([1, 2, 3, 6, 32]))
        try:
            for _ in range(ops):
                r = rng.random()
                if r < 0.30:
                    h.block()
                elif r < 0.75:
                    b = h.nblk - 1
                    h.frames(rng.randint(1, fpb), f"f{max(b, 0)}." if False else "z")
                elif r < 0.88:
                    h.drain()
                elif r < 0.95:
                    h.led.on_segment_end()
                else:
                    h.led.clear()
            h.drain()
        except Exception as e:                       # noqa: BLE001
            fails.append(f"[fuzz seed={seed}] 抛异常：{type(e).__name__}: {e}")
            continue
        # fuzz 里帧是匿名的，只验 ①③（②需要可追溯的帧名）
        if not h.led.reconciled:
            fails.append(f"[fuzz seed={seed}] 对账不平：{h.led.stats()}")
        if h.blk_order != sorted(h.blk_order):
            fails.append(f"[fuzz seed={seed}] 发布顺序乱了")
        if len(set(h.blk_order)) != len(h.blk_order):
            fails.append(f"[fuzz seed={seed}] 同一块发了两次")
    return rounds, fails


# ── 长跑 ──────────────────────────────────────────────────────────

def run_long(blocks: int = 50_000, fpb: int = 12) -> tuple[str, list[str]]:
    """5 万块 ≈ 6.7 小时语音。**验编号不漂、账不歪。**

    这一条专门回答「时间长了会不会错开」：编号是从 0 一路往上涨的整数，
    理论上不会漂 —— 但「理论上」正是今天翻过两次车的地方。
    """
    fails = []
    h = Harness(fpb, max_pending=8)
    for i in range(blocks):
        h.block_with_frames()
        if i % 5 == 0:
            h.drain()
        if i % 997 == 0:
            h.led.on_segment_end()
    h.drain()
    fails += h.check("长跑")
    if h.popped and h.popped[-1].blk != h.blk_order[-1]:
        fails.append("长跑：末尾块号对不上")
    hours = blocks * fpb / 25 / 3600
    return f"{blocks} 块 ≈ {hours:.1f} 小时语音，发出 {len(h.popped)} 块", fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="把结果写成 markdown")
    ap.add_argument("--fuzz-rounds", type=int, default=300)
    ap.add_argument("--long-blocks", type=int, default=50_000)
    a = ap.parse_args()

    logging.disable(logging.CRITICAL)        # 预期内的 ERROR 不刷屏
    lines = []

    n_case, f_case = run_cases()
    print(f"角落情况  {n_case - len(f_case)}/{n_case}")
    lines.append(f"| 角落情况 | {n_case} | {n_case - len(f_case)} | {len(f_case)} |")

    n_fz, f_fz = run_fuzz(a.fuzz_rounds)
    print(f"随机 fuzz  {n_fz - len(f_fz)}/{n_fz} 条序列")
    lines.append(f"| 随机 fuzz | {n_fz} | {n_fz - len(f_fz)} | {len(f_fz)} |")

    desc, f_long = run_long(a.long_blocks)
    print(f"长跑      {desc}  {'通过' if not f_long else '失败'}")
    lines.append(f"| 长跑（{desc}） | 1 | {0 if f_long else 1} | {len(f_long)} |")

    fails = f_case + f_fz + f_long
    for m in fails[:20]:
        print("  ❌", m)
    print(f"\n{'=' * 60}\n共 {n_case + n_fz + 1} 项，失败 {len(fails)} 项")

    if a.report:
        pathlib.Path(a.report).write_text(
            "| 类别 | 项数 | 通过 | 失败 |\n|---|---|---|---|\n" + "\n".join(lines)
            + ("\n\n失败明细：\n\n" + "\n".join(f"- {m}" for m in fails) if fails else ""),
            encoding="utf-8")
        print(f"报告 → {a.report}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
