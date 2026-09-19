"""音画配对账本 —— **记住每一帧是哪一块音频生成的。**

设计与取舍见 `docs/av-pairing-design.md`，这里只讲代码怎么用、为什么这么写。

## 它解决什么

上游生成器吐的是裸张量，没有任何标识。发布侧原来按「已发出几秒音频 /
几秒视频」配平 —— **秒数可以完全对得上而对应关系完全错，并且不报错**。
实测同机同码跑两遍，同一句的音画偏移一次 −20 ms、一次 +1940 ms。

这个账本把对应关系建起来：块进来时登记，帧出来时按**拉取顺序**归属，
一块的帧齐了才允许发布。发布顺序就是配对关系 ——
LiveKit 的协议本来就是靠交错顺序表达配对的，不靠序号。

## 为什么纯逻辑、不碰 torch 和 livekit

这一层的全部错误都是**计数错误**，而计数错误在有 GPU 的环境里最难复现：
要占一张卡、要等模型加载、还得凑出"队列正好满"这种时序。
剥成纯逻辑之后，那些情况变成三行 `for` 循环就能构造的单测。

## 用法

    led = AVLedger(frames_per_block=12, max_pending_blocks=6)

    blk = led.on_block_pulled(pcm)     # 模型要走一块音频
    led.on_frame(img)                  # 上游吐出一帧
    while (u := led.pop_ready()):      # 帧齐了的块，成对拿走
        publish(u.pcm, u.frames)
    led.on_segment_end()               # 一句说完，顺便对账

## 三条不肯让步的地方

1. **不许静默修正。** 数对不上就 `ERROR`，宁可吵。这条链上所有的坑，
   共同点都是「不报错地把错误往下传」。
2. **丢就整块丢**（帧和它那段音频一起），边界永远落在块上，
   对账是整数运算，没有半块状态。
3. **预热帧直接扔掉。** 上游开头那几轮用预置音频生成的帧，
   不对应我们任何一段声音 —— 混进去就是从第一帧起就错位。
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("closecrab.avatar.ledger")


@dataclass
class Unit:
    """一块音频和它那几帧 —— **配好对的一组，发布的最小单位。**"""

    utt: int
    blk: int
    pcm: Any
    want: int                                   # 这块该有几帧
    frames: list[Any] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.want


class AVLedger:
    """块进、帧出、成对走。

    线程模型：**全部调用必须在同一个线程里**。`on_frame` 来自模型线程、
    `on_block_pulled` 也来自模型线程（那个回调就是模型调的），
    `pop_ready` 在同一个消费循环里 —— 现有代码里这三者本来就同线程。
    真要跨线程，外面加锁，别在这里面加：**这里加了锁，调用方就会以为
    整个序列是原子的**，而它不是。
    """

    def __init__(self, *, frames_per_block: int, max_pending_blocks: int = 6,
                 lead_blocks: int = 0) -> None:
        if frames_per_block <= 0:
            raise ValueError("frames_per_block 必须为正")
        self.fpb = frames_per_block
        self.max_pending = max(1, max_pending_blocks)
        self.lead_blocks = max(0, lead_blocks)
        """**流水线深度：拉到第 k 块时，正在吐的是第 k−lead 块的帧。**

        「先进先出归属」有个没写下来的前提：模型拉一块、马上吐这一块的帧。
        这条管线不满足 —— 它开头两轮用预置音频预热，那两轮的帧在我们
        第一次拉块**之后**才出来，于是被算成第 0、1 块的；再加上在途的，
        整体错开约 4 块。

        表现是「声音先出来，2 秒后嘴才动」，而账本**照样报对账平、零错误**
        —— 因为数量完全对得上，错的是对应关系。这正是设计文档里写的
        「唯一的脆弱点」，而我先前只验了「一次回调恰好 12 帧」那一半。

        ⚠️ 它是**块数**（整数、结构性的），不是毫秒。所以跟「量一个延迟去补」
        不是一回事：墙上时钟会随负载漂，块数不会。"""
        self._lead_left = self.lead_blocks * frames_per_block

        self._pending: deque[Unit] = deque()   # 已登记、帧还没齐
        self._ready: deque[Unit] = deque()     # 帧齐了、等着被取走
        self._utt = 0
        self._next_blk = 0
        self._awaiting_warmup = True
        """现在吐出来的帧算不算「预热」。

        ⚠️ 判据**不能**写成 `blocks_pulled == 0` —— 换形象会让上游从头
        再预热一遍，那时 `blocks_pulled` 早就不是 0 了，于是新一轮的预热帧
        会被当成「多出来的帧」狂报 ERROR，而它们完全是预期内的。
        Chris 2026-09-19 提出换图这个 case 时才发现的。"""

        # 对账用。**每一项都要能对上**，见 `reconciled`。
        self.frames_seen = 0
        self.frames_attributed = 0
        self.frames_prewarm = 0        # 还没有任何块时吐出来的，直接扔
        self.frames_dropped = 0        # 背压丢掉的（成块丢）
        self.frames_rejected = 0       # 无处可归、被拒收的（见 on_frame ②）
        self.blocks_pulled = 0
        self.blocks_dropped = 0
        self.frames_published = 0
        self.errors = 0
        self.drains = 0

    # ── 进 ────────────────────────────────────────────────────────

    def on_block_pulled(self, pcm: Any) -> int:
        """模型要走了一块音频。返回它的块号。"""
        self._awaiting_warmup = False        # 开始要真音频了，预热结束
        u = Unit(utt=self._utt, blk=self._next_blk, pcm=pcm, want=self.fpb)
        self._next_blk += 1
        self.blocks_pulled += 1
        self._pending.append(u)
        self._evict_if_needed()
        return u.blk

    def on_frame(self, img: Any) -> None:
        """上游吐出一帧。按**拉取顺序**归给最早那个还没填满的块。"""
        self.frames_seen += 1
        # 开头这 lead_blocks 块的帧是预热/在途的产物，不属于我们拉的任何一块。
        if self._lead_left > 0:
            self._lead_left -= 1
            self.frames_prewarm += 1
            return
        for u in self._pending:
            if not u.full:
                u.frames.append(img)
                self.frames_attributed += 1
                if u.full:
                    self._promote()
                return

        # 没有块在等帧。两种情况，处理不同：
        if self._awaiting_warmup:
            #   ① 还没拉过块（或刚重启过）—— 上游预热阶段用预置音频生成的帧。
            #      **扔掉**：它不对应我们任何一段声音，混进去就是从第一帧
            #      起就错位。这是预期内的，不算错。
            self.frames_prewarm += 1
            return
        #   ② 已经拉过块，却还多吐 —— 「一块恰好 N 帧」这个假设破了。
        #      **这是整个方案唯一的脆弱点**，必须吵，绝不能默默塞进下一块：
        #      默默塞进去之后，从此每一块都错开一帧，而且再也查不出来。
        self.frames_rejected += 1
        self.errors += 1
        log.error("多出来的帧：已拉 %d 块、每块应 %d 帧，却又来一帧（累计 %d 次）"
                  "—— 「一块恰好 N 帧」的假设不成立，配对已不可信",
                  self.blocks_pulled, self.fpb, self.errors)

    def on_segment_end(self) -> int:
        """一句说完。**顺便对账**，返回这句里没填满就被扔掉的块数。

        句尾是天然的对账点：在途的本该清空。没清空说明上游对这一句
        少吐了帧 —— 留着它们会把下一句的帧吃进来，那才是灾难。
        """
        stale = len(self._pending)
        if stale:
            self.errors += 1
            missing = sum(self.fpb - len(u.frames) for u in self._pending)
            log.error("句尾还有 %d 块没填满（共缺 %d 帧）—— 丢掉，"
                      "不然下一句的帧会被它们吃掉", stale, missing)
            for u in self._pending:
                self.frames_dropped += len(u.frames)
                self.blocks_dropped += 1
            self._pending.clear()
        self._utt += 1
        return stale

    # ── 出 ────────────────────────────────────────────────────────

    def pop_ready(self) -> Unit | None:
        """取走下一组配好对的。没有就返回 None。

        **「已发布」记在这里，不记在搬进待发队列的时候** —— 待发队列里的
        东西还可能因为积压被丢掉，提前记成已发布，对账就永远平不了。
        """
        if not self._ready:
            return None
        u = self._ready.popleft()
        self.frames_published += len(u.frames)
        return u

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    # ── 打断 ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """被打断：在途和待发的全扔。

        ⚠️ **不重置块号**。块号只需单调，不需要连续 —— 重置的话打断前后
        的块会同号，日志里两段对不上，排障时最需要它的时候它是错的。
        """
        for u in list(self._pending) + list(self._ready):
            self.frames_dropped += len(u.frames)
            self.blocks_dropped += 1
        self._pending.clear()
        self._ready.clear()

    def on_pipeline_drained(self) -> None:
        """管线抽干了 —— **对应关系在这一刻归零。**

        模型回头来要下一块而我们没货，说明它已经把在途的全吐完了。
        这是唯一一个不用猜流水线深度就能确定对应关系的时刻。

        > Chris 2026-09-19：「你把音频怼进去然后等着，不管等多久，
        > 出来的第一帧准是你自己的帧。」

        比 `lead_blocks` 那条路好在：**它把一个假设变成了可强制的状态**。
        猜深度要求「我猜对了」，抽干只要求「我等到了」。

        在途的必须扔 —— 那些帧属于上一段，留着会被算进下一块。
        """
        n = len(self._pending) + len(self._ready)
        if n:
            for u in list(self._pending) + list(self._ready):
                self.frames_dropped += len(u.frames)
                self.blocks_dropped += 1
            self._pending.clear(); self._ready.clear()
        self._lead_left = 0          # 抽干之后不需要再吃预热
        self.drains += 1

    def on_generator_restart(self, why: str = "换形象") -> None:
        """上游生成器从头开始了（**换形象是唯一会触发的情形**）。

        > Chris 2026-09-19：「换图片这个事情就把 generate 停了，不能再续上，
        > 直接从头开始，不要任何 workaround —— 省得上下文污染。」

        对账本来说要做两件事，缺一不可：

        1. **在途的全扔。** 那些帧是旧形象画的，配新音频就是张冠李戴。
        2. **重新进入预热态。** 上游重启后会再用预置音频跑几轮，
           那批帧必须当预热扔掉 —— 不重新置位的话它们会被判成
           「多出来的帧」，一路狂报 ERROR，而那是**预期内**的东西。
           报错要留给真异常，假警报会把真警报淹掉。

        块号**照旧不重置**（见 `clear`）。
        """
        n = len(self._pending) + len(self._ready)
        self.clear()
        self._awaiting_warmup = True
        self._lead_left = self.lead_blocks * self.fpb   # 重启要重新吃一遍预热
        log.info("生成器重启（%s）：扔掉在途 %d 块，重新进入预热态", why, n)

    # ── 对账 ──────────────────────────────────────────────────────

    @property
    def reconciled(self) -> bool:
        """每一帧都有下落吗？

        ⚠️ **被拒收的帧也要有位置。** 第一版漏了 `frames_rejected` ——
        于是只要报过一次「多出来的帧」，对账就永远平不了，而对账正是
        判断「配对还可不可信」的唯一依据。**一个报过错就再也不会变绿的
        指标，跟一个永远绿的指标一样没用。**
        fuzz 第一轮就撞出来了（300 条序列全红）。
        """
        in_flight = sum(len(u.frames) for u in self._pending) \
            + sum(len(u.frames) for u in self._ready)
        return self.frames_seen == (self.frames_prewarm + self.frames_rejected
                                    + self.frames_dropped + self.frames_published
                                    + in_flight)

    def stats(self) -> dict[str, int | bool]:
        return {
            "utt": self._utt,
            "blocks_pulled": self.blocks_pulled,
            "blocks_dropped": self.blocks_dropped,
            "frames_seen": self.frames_seen,
            "frames_prewarm": self.frames_prewarm,
            "frames_dropped": self.frames_dropped,
            "frames_rejected": self.frames_rejected,
            "frames_published": self.frames_published,
            "pending_blocks": len(self._pending),
            "ready_blocks": len(self._ready),
            "errors": self.errors,
            "drains": self.drains,
            "reconciled": self.reconciled,
        }

    # ── 内部 ──────────────────────────────────────────────────────

    def _promote(self) -> None:
        """把队头**连续**填满的块挪到待发队列。

        只挪队头：中间那块先填满不代表它能先发 —— 发布顺序必须等于
        拉取顺序，否则声音会串。
        """
        while self._pending and self._pending[0].full:
            self._ready.append(self._pending.popleft())

    def _evict_if_needed(self) -> None:
        """积压太多就丢**最旧的一整块**（帧 + 它那段音频一起）。

        丢帧本身是对的 —— 数字人只有「现在」有意义，补播两秒前的嘴型比
        丢帧更糟。**错的是只丢视频不丢音频**：那样每丢一帧对应关系就永久
        错开一帧，而秒数配平会自动"重新平衡"，把错位藏起来。

        丢了要吼。这是持续性退化不是偶发，**不去重** —— 压成一条会让人
        以为只发生过一次。
        """
        while len(self._pending) + len(self._ready) > self.max_pending:
            # 优先丢**最旧的待发**；待发空了才动在途的。
            u = self._ready.popleft() if self._ready else self._pending.popleft()
            self.frames_dropped += len(u.frames)
            self.blocks_dropped += 1
            log.warning("积压超过 %d 块，丢掉整块 #%d（%d 帧 + 对应音频）——"
                        "累计丢 %d 块", self.max_pending, u.blk,
                        len(u.frames), self.blocks_dropped)
