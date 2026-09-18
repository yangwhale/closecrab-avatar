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


# ── ⭐ 新契约：每角色一个开关，服务端分配 ──────────────────────────

def test_each_role_has_its_own_switch():
    from closecrab_avatar.policy import AvatarRole as R, ATTR_WANT_BY_ROLE, wanted_roles
    a = ATTR_WANT_BY_ROLE[R.ASSISTANT]
    p = ATTR_WANT_BY_ROLE[R.PRINCIPAL]
    assert wanted_roles({"me": {p: "true"}}) == {R.PRINCIPAL}
    assert wanted_roles({"me": {a: "true"}}) == {R.ASSISTANT}
    assert wanted_roles({"me": {a: "true", p: "true"}}) == {R.PRINCIPAL, R.ASSISTANT}


def test_nobody_wants_by_default():
    """老客户端不发这些键 —— 默认开的话每个旧客户端都会去抢一路 GPU。"""
    from closecrab_avatar.policy import wanted_roles
    assert wanted_roles({}) == set()
    assert wanted_roles({"old": {"unrelated": "1"}}) == set()


def test_anyone_wanting_counts():
    """任何一个人要就算要 —— 别让关开关的那个人把别人的画面也掐了。"""
    from closecrab_avatar.policy import AvatarRole as R, ATTR_WANT_BY_ROLE, wanted_roles
    p = ATTR_WANT_BY_ROLE[R.PRINCIPAL]
    assert wanted_roles({"a": {p: "false"}, "b": {p: "true"}}) == {R.PRINCIPAL}


def test_one_slot_goes_to_principal():
    """⭐ 只有一路时两个都开 → 给本体。Chris 定的优先级。"""
    from closecrab_avatar.policy import AvatarRole as R, allocate
    assert allocate({R.PRINCIPAL, R.ASSISTANT}, capacity=1) == [R.PRINCIPAL]
    assert allocate({R.ASSISTANT}, capacity=1) == [R.ASSISTANT]
    assert allocate(set(), capacity=1) == []


def test_two_slots_serve_both_without_protocol_change():
    """⭐ 以后多一路，**协议一个字不用改** —— 这正是「意图/分配分开」的收益。"""
    from closecrab_avatar.policy import AvatarRole as R, allocate
    assert allocate({R.PRINCIPAL, R.ASSISTANT}, capacity=2) == [R.PRINCIPAL, R.ASSISTANT]
    assert allocate({R.PRINCIPAL, R.ASSISTANT}, capacity=0) == []


def test_role_names_match_persona_roles():
    """角色名必须跟形象库共用一套，否则「换了助手的图，兔子的脸变了」。"""
    from closecrab_avatar.policy import AvatarRole as R
    assert {R.ASSISTANT.value, R.PRINCIPAL.value} == {"assistant", "principal"}
