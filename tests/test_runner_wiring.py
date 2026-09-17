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
