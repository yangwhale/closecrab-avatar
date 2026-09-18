"""把音频喂给数字人 —— **这条字节流的生命周期归这里管，别在调用方各写一遍。**

## 为什么要有这个文件

`DataStreamAudioOutput` 有两个长得很像、干的完全不是一件事的方法：

    clear_buffer()  只发一条 `lk.clear_buffer` RPC 通知对端把缓冲丢掉，
                    **`_stream_writer` 一个字都不碰**
    flush()         关掉当前这条字节流、置空，下一次 `capture_frame`
                    才会 `stream_bytes()` 重开一条

对端收到 clear_buffer 就把那条流当作作废了。调用方这边流还开着，于是之后写进去
的每一帧都**掉进黑洞：不报错、不抛异常、不断线**。

2026-09-18 线上就是这么坏的。Chris 在真机上测出四条现象：

    1. 第一次播放好使（流是新的）
    2. 开着 Avatar 点重播 → 卡住
    3. 把 Avatar 关掉 → 语音模式接着播（出口切回本地音轨，绕开那条死流）
    4. 再打开 Avatar → 又好了（重建会话 ＝ 新 sink ＝ 新流）

第 3 条是决定性线索：**坏的不是音频源，是那一条出口。**

## 所以它必须住在产品仓库里

这套语义有两个调用方，形态完全不同：

    语音助手进程      走 livekit-agents 的 AgentSession，插件帮它接好
    bot 播报旁路      没有 AgentSession，是一个裸的 PCM 泵，只能自己接

第二个当初因为接口对不上而**手写了一遍**，于是同一个语义在两处各实现一次 ——
写错的那一处就是上面那个 bug，而另一处是对的，所以一直没人发现。

> Chris 2026-09-18：「这种复杂度……不要暴露出来。」

这个类就是那层收口：调用方只说**发生了什么**（说了一段 / 被打断了 / 说完了 /
不用了），什么时候该换流是这里的事。

## 状态机

    capture_frame ──► 有流在写，dirty=True
         │
         ├── interrupt()      打断 / 重播：通知对端丢缓冲 ＋ 换新流
         ├── end_utterance()  一句说完：换新流（dirty 才做）
         └── aclose()         不用了：把流关上，别让对端干等
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger("closecrab.avatar.audio_sink")


class _RawSink(Protocol):
    """我们用到的那几个方法。**用 Protocol 不 import 具体类** ——
    这个模块因此能被没装 livekit-agents 的地方 import（比如离线测试）。"""

    async def capture_frame(self, frame: Any) -> None: ...
    def flush(self) -> None: ...
    def clear_buffer(self) -> None: ...


class AvatarAudioSink:
    """数字人音频出口。**包住那条字节流该什么时候换。**

    每个方法都吞异常只记日志：这条路上任何一步失败，正确的降级都是
    「这一段没对上口型」，而不是「整轮播放崩掉」。
    """

    def __init__(self, raw: _RawSink, *, label: str = "avatar") -> None:
        self._raw = raw
        self._label = label
        self._dirty = False
        """自上次换流以来写过东西没有。

        只为了**别对着一条空流反复 flush** —— 调用方通常在空闲循环里调
        `end_utterance()`，不记这个状态就会每一轮都开一次关流任务。
        """

    @property
    def raw(self) -> _RawSink:
        """底下那个真出口。给必须直接操作它的地方用（少用）。"""
        return self._raw

    @property
    def dirty(self) -> bool:
        """当前这条流写过东西没有。主要给测试和排障看。"""
        return self._dirty

    async def capture_frame(self, frame: Any) -> None:
        """喂一帧。没有流就会自动开一条（`DataStreamAudioOutput` 的行为）。"""
        await self._raw.capture_frame(frame)
        self._dirty = True

    def interrupt(self) -> None:
        """打断 / 重播 / 拖进度条 —— 队列里那些话已经不作数了。

        **两步都要做，而且顺序不能反：**

        1. `clear_buffer()` 让对端把手里那几百毫秒丢掉。不做的话它会接着
           对完口型再停 —— 「我已经在讲新的了，屏幕上那张脸还在念上一句」。
        2. `flush()` 把这条已经被对端判死的流关掉，下一句开新的。

        反过来的话：流先没了，`clear_buffer()` 里那道 `_started` 门会让它
        直接 return，对端手里那段就留下来了。

        `clear_buffer()` 抛了也要继续 `flush()` —— 缓冲没清掉已经够糟，
        再把流焊死就彻底没救。
        """
        try:
            self._raw.clear_buffer()
        except Exception:                                # noqa: BLE001
            logger.debug("[%s] 通知对端清缓冲失败", self._label, exc_info=True)
        self._rotate("打断")

    def end_utterance(self) -> None:
        """一句说完了（调用方判断，通常是「静默超过某个阈值」）。

        ## 为什么说完要主动换流

        不换也能用：下一句写进同一条开着的流里，对端接着读。实测一段连续
        107.8 秒的音频就是好几句拼起来的。

        但那让「隔一会儿再说下一句」依赖**一条已经开了很久的流还活着** ——
        而上面那个 bug 正是「以为流还活着，其实对端早判死了」，同一类，
        静默、不报错。

        换掉之后两者结构上一样了：**每一句都是一条新流**，而「第一句总是
        好使」是反复验证过的路径。代价只有一次 `stream_bytes()`。
        """
        self._rotate("说完一句")

    def aclose(self) -> None:
        """不用这个出口了（摘数字人 / 换出口 / 收摊）。

        ⚠️ 不关的话流一直开着，对端那条 reader 等不到结束标记。表现是
        「数字人已经走了，网关那边的会话却要等空闲回收才还槽位」——
        而槽位是稀缺的。
        """
        self._rotate("收摊", force=True)

    # ── 内部 ──────────────────────────────────────────────────────

    def _rotate(self, why: str, *, force: bool = False) -> None:
        """关掉当前这条流，下次写的时候自动开新的。

        `force=False` 时 `dirty` 为假就跳过 —— 见 `_dirty` 的说明。
        收摊那一路用 `force=True`：哪怕没写过东西，也要把可能开着的流关上。
        """
        if not force and not self._dirty:
            return
        self._dirty = False
        try:
            self._raw.flush()
        except Exception:                                # noqa: BLE001
            logger.debug("[%s] 关流失败（%s）", self._label, why, exc_info=True)
