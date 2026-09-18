"""接线测试 —— 只看「命令行参数有没有一路传到模型」。

## 为什么需要这么一个怪测试

`worker/runner.py` 加过一次 `--ckpt-dir` / `--training-config`：argparse 加了、
构造函数签名加了、检测逻辑也用上了，**就是 `main()` 里的构造调用点没跟着改**。

结果是最难查的那种故障：注册、心跳、建会话、出帧，一路全绿，日志一条
错误都没有 —— 只是画面不动，因为路径是空的，检测如实判定「没有模型」，
安静地退回静帧兜底。**兜底越体面，漏越难发现。**

普通单测抓不到它：测试自己把参数传得好好的。**被测对象正确 ≠ 组装正确。**
能抓的只有对调用点本身下断言。

## 为什么用 AST 不用 grep

`grep 'ckpt_dir=a.ckpt_dir'` 会被换行、空格、参数顺序、改成 `**vars(a)`
全部骗过。AST 看的是「这个 Call 节点有没有这个关键字参数」，跟源码长相无关。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = pathlib.Path(__file__).resolve().parent.parent / "worker" / "runner.py"

# 命令行给了、模型要用，就必须在构造点传过去。加新参数时往这里补一条。
# 键 = 构造 `LiveAvatarPipelineSource` 时的 kwarg，值 = 对应的命令行开关。
REQUIRED = {
    "ckpt_dir": "--ckpt-dir",
    "training_config": "--training-config",
    "ref_image_path": "--image",
    "warmup_audio": "--warmup-audio",
    "prompt": "--prompt",
    "size": "--size",
    "lora_path": "--lora-path",
}

SOURCE_CLS = "LiveAvatarPipelineSource"


def _src() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _calls(name: str) -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(_src()))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def test_every_source_call_site_passes_the_model_args():
    """⭐ 就是这条抓的那个 bug。"""
    calls = _calls(SOURCE_CLS)
    assert calls, f"runner.py 里找不到 {SOURCE_CLS}(...) 调用点 —— 是不是改名了？"
    for c in calls:
        given = {k.arg for k in c.keywords if k.arg}
        missing = set(REQUIRED) - given
        assert not missing, (
            f"runner.py:{c.lineno} 的 {SOURCE_CLS}(...) 漏传 {sorted(missing)}。\n"
            "漏了不会报错，只会让 worker 安静地退回静帧 —— 画面不动但日志全绿。")


@pytest.mark.parametrize("kwarg,flag", sorted(REQUIRED.items()))
def test_cli_exposes_each_arg(kwarg, flag):
    """反向那一半：模型要的，命令行得能配。

    只测调用点的话，有人把 `ckpt_dir=""` 硬编码进去照样绿。
    """
    src = _src()
    assert f'"{flag}"' in src or f"'{flag}'" in src, \
        f"模型要 {kwarg}，但命令行没有 {flag} —— 生产上没法配"


@pytest.mark.parametrize("kwarg", sorted(REQUIRED))
def test_the_arg_comes_from_argparse_not_a_literal(kwarg):
    """⭐ 传了，但传的得是**命令行那个值**。

    `ckpt_dir="/tmp/x"` 一样能让上一条通过 —— 那种「传了个假的」比漏传
    更隐蔽，因为连 grep 都看得到这个 kwarg 在。
    """
    for c in _calls(SOURCE_CLS):
        for k in c.keywords:
            if k.arg != kwarg:
                continue
            assert isinstance(k.value, ast.Attribute), (
                f"runner.py:{c.lineno} 的 {kwarg}= 不是取自 argparse 的结果，"
                f"而是 {ast.dump(k.value)[:60]}…")


def test_worker_gets_the_source_not_the_paths():
    """`Worker` 只该知道「有没有 source」。

    路径判定留在 `main()` 一处 —— 两处各判一次的话，迟早一处改了另一处没改，
    而症状还是那个「日志全绿画面不动」。
    """
    calls = _calls("Worker")
    assert calls, "找不到 Worker(...) 调用点"
    for c in calls:
        given = {k.arg for k in c.keywords if k.arg}
        assert "source" in given, f"runner.py:{c.lineno} 的 Worker 没拿到 source"
        leaked = given & {"ckpt_dir", "training_config"}
        assert not leaked, f"路径又漏进 Worker 了：{sorted(leaked)}"


def test_distributed_is_actually_initialised():
    """⭐ torchrun **只设环境变量**，进程组得自己建。

    漏了 `init_process_group` 的症状跟漏传参数一模一样：一路全绿，
    画面不动。这条守住那一行别在重构里蒸发。
    """
    src = _src()
    assert "init_process_group" in src, \
        "没有 init_process_group —— torchrun 下 dist 永远没初始化，模型起不来"
    assert "set_device" in src, \
        "没有 torch.cuda.set_device —— 5 个 rank 会挤在 0 号卡上 OOM"


def test_dit_ranks_do_not_register():
    """只有 VAE rank 该注册。

    5 个 rank 都注册的话控制面会多出 4 个幽灵 worker，按它们的容量派活，
    派过去没人接 —— 表现是「有时候能用有时候不能用」。
    """
    tree = ast.parse(_src())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    # main() 里必须有一条在 Worker(...) 之前 return 的分支（DiT rank 那条）
    returns_before_worker = [
        n for n in ast.walk(main) if isinstance(n, ast.Return)
        and n.lineno < min(c.lineno for c in _calls("Worker"))]
    assert returns_before_worker, \
        "main() 里没有「DiT rank 提前返回」那条路 —— 5 个 rank 都会去注册"


def test_process_group_timeout_is_not_the_default():
    """⭐ NCCL watchdog 默认 10 分钟，会把「空闲阻塞省电」那条设计打死。

    实测：最后一次说话之后整十分钟，五个进程一起 SIGABRT ——
      Watchdog caught collective operation timeout: WorkNCCL(OpType=RECV)
      ran for 600087 ms before timing out

    现象是「worker 好好的突然不在册了」，原因藏在日志最底下一大段 C++ 栈里。
    这条守住那个显式 timeout 别在重构里蒸发。
    """
    src = _src()
    assert "timeout=" in src and "timedelta" in src, \
        "init_process_group 没有显式 timeout —— 空闲 10 分钟后整组会被 NCCL 打掉"


# ── ⭐ 会话收尾必须关掉 AvatarRunner ─────────────────────────────────


def _run_session_ast():
    """拿 `_run_session` 的语法树。

    这一条只能用结构断言：真跑一遍要起 LiveKit 房间、网关 HTTP 和五卡模型。
    而要钉的东西恰恰是结构性的 —— **`finally` 里有没有那一句**。
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "worker" / "runner.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_session":
            return ast, node
    raise AssertionError("找不到 _run_session")


def test_session_teardown_closes_the_runner():
    """⭐ 漏掉这一句的后果不是「多占点内存」，是**画面严重卡顿、一场比一场重**。

    `AvatarRunner` 自己起三条常驻任务，不关就永远活着；而每一场的 generator
    包的都是**同一个** pipeline source —— 漏下来的消费者会跟当前这一场抢帧，
    真正在播的那路只拿到 1/N。

    日志指纹：同一毫秒出现 2～3 条**一模一样**的
    `Frame capture was behind schedule for 4188.51 ms`。一个同步器不可能
    重复报同一个值，重数就是当时活着的 runner 数。
    """
    ast, fn = _run_session_ast()
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "_run_session 连 finally 都没有"
    closed = any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "aclose"
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "runner"
        for t in tries for stmt in t.finalbody for c in ast.walk(stmt)
    )
    assert closed, "会话收尾没有 await runner.aclose() —— 会漏 AvatarRunner"


def test_runner_is_predeclared_so_finally_cannot_nameerror():
    """`runner` 在 `try` 里才赋值。**进 try 之前就抛的话 finally 会 NameError** ——
    而那会把真正的异常盖掉，日志里只剩一句莫名其妙的 NameError。"""
    ast, fn = _run_session_ast()
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    first_try_line = min(t.lineno for t in tries)
    pre = [n for n in ast.walk(fn)
           if isinstance(n, ast.Assign) and n.lineno < first_try_line
           and any(isinstance(t, ast.Name) and t.id == "runner" for t in n.targets)]
    assert pre, "runner 没有在 try 之前预先置空"
