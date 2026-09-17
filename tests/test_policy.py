"""契约检查跑在子进程里。

`scripts/policy-contract.py` 是一份**能独立跑**的检查脚本（跟 `smoke.py`
一个路子）—— 客户端同学不装 pytest 也能 `python3 scripts/policy-contract.py`
对一遍属性名和状态机。这里只是把它拉进 CI，不重写一遍。
"""
import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "policy-contract.py"


def test_policy_contract_holds():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, f"契约检查没过：\n{r.stdout[-2000:]}\n{r.stderr[-500:]}"
