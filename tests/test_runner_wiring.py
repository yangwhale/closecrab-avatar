"""接线测试 —— 只看「参数有没有从命令行一路传到构造函数」。

## 为什么需要这么一个怪测试

`worker/runner.py` 加过一次 `--ckpt-dir` / `--training-config`：argparse 加了、
构造函数签名加了、`_detect_model()` 也用上了，**就是 `main()` 里的
`Worker(...)` 调用点没跟着改**。

结果是最难查的那种故障：注册、心跳、建会话、出帧，一路全绿，日志一条
错误都没有 —— 只是画面不动。因为 `self._ckpt_dir` 是空串，`_detect_model()`
如实判定「没有模型」，安静地退回静帧兜底。**兜底越体面，漏越难发现。**

普通单测抓不到它：测试自己 `Worker(ckpt_dir=..., ...)` 构造，参数传得好好的。
**被测对象正确 ≠ 组装正确。** 能抓的只有对调用点本身下断言。

## 为什么用 AST 不用 grep

`grep 'ckpt_dir=a.ckpt_dir'` 会被换行、空格、参数顺序、改成 `**vars(a)` 全部骗过。
AST 看的是「这个 Call 节点有没有这个关键字参数」，跟源码长相无关。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = pathlib.Path(__file__).resolve().parent.parent / "worker" / "runner.py"

# 命令行给了、构造函数收了，就必须在调用点传过去。加新参数时往这里补一条。
REQUIRED_KWARGS = {"ckpt_dir", "training_config"}


def _tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def _worker_calls(tree: ast.Module) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "Worker"]


def test_every_worker_call_site_passes_the_model_args():
    """⭐ 就是这条抓的那个 bug。"""
    calls = _worker_calls(_tree())
    assert calls, "runner.py 里找不到 Worker(...) 调用点 —— 是不是改名了？"
    for c in calls:
        given = {k.arg for k in c.keywords if k.arg}
        missing = REQUIRED_KWARGS - given
        assert not missing, (
            f"runner.py:{c.lineno} 的 Worker(...) 漏传 {sorted(missing)}。\n"
            "漏了不会报错，只会让 worker 安静地退回静帧 —— 画面不动但日志全绿。")


def test_cli_exposes_the_model_args():
    """反向那一半：构造函数要的，命令行得能给。

    只测调用点的话，有人把 `ckpt_dir=""` 硬编码进去照样绿。
    """
    src = RUNNER.read_text(encoding="utf-8")
    for name in REQUIRED_KWARGS:
        flag = "--" + name.replace("_", "-")
        assert f'"{flag}"' in src or f"'{flag}'" in src, \
            f"构造函数收 {name}，但命令行没有 {flag} —— 生产上没法配"


def test_worker_init_accepts_them():
    """第三段：签名本身。三段都在，链路才是通的。"""
    fn = next((n for n in ast.walk(_tree())
               if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    assert fn is not None, "找不到 Worker.__init__"
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert REQUIRED_KWARGS <= params, \
        f"Worker.__init__ 少了 {sorted(REQUIRED_KWARGS - params)}"


@pytest.mark.parametrize("name", sorted(REQUIRED_KWARGS))
def test_the_arg_actually_reaches_detection(name):
    """最后一段：存下来之后**有人读**。

    存进 `self._ckpt_dir` 却没人用，前三条照样全绿 —— 那正是「测试全绿
    不是证据」的典型形态。
    """
    src = RUNNER.read_text(encoding="utf-8")
    attr = f"self._{name}"
    assert src.count(attr) >= 2, \
        f"{attr} 只出现 {src.count(attr)} 次 —— 存下来了但没人读？"
