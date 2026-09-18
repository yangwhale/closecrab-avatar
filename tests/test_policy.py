"""契约检查跑在子进程里。

`scripts/policy-contract.py` 是一份**能独立跑**的检查脚本（跟 `smoke.py`
一个路子）—— 客户端同学不装 pytest 也能 `python3 scripts/policy-contract.py`
对一遍属性名和状态机。这里只是把它拉进 CI，不重写一遍。
"""
import pathlib
import subprocess
import sys

from closecrab_avatar.policy import (
    ATTR_STATE,
    ATTR_STATE_BY_ROLE,
    ATTR_VISIBLE,
    ATTR_WANT,
    ATTR_WANT_BY_ROLE,
    AvatarRole,
    any_visible,
    wanted_roles,
)

SCRIPT =pathlib.Path(__file__).resolve().parent.parent / "scripts" / "policy-contract.py"


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


def test_avatar_identity_carries_role():
    """⭐ 两路同时在房间里不能撞名字 —— identity 是 LiveKit 的唯一键，
    撞了后进的会把先进的踢掉，而表现看起来像「切换成功」。"""
    from closecrab_avatar.policy import AvatarRole as R, avatar_identity
    a, p = avatar_identity(R.ASSISTANT), avatar_identity(R.PRINCIPAL)
    assert a != p and a.endswith("assistant") and p.endswith("principal")


def test_identity_round_trips():
    from closecrab_avatar.policy import AvatarRole as R, avatar_identity, role_of_identity
    for r in R:
        assert role_of_identity(avatar_identity(r)) is r


def test_unknown_identity_returns_none_not_a_guess():
    """认不出来返回 None。猜「当 principal」会把两路都算成本体 —— 静默错位。"""
    from closecrab_avatar.policy import role_of_identity
    for bad in ("cc-avatar", "cc-avatar-", "cc-avatar-乱写", "someone-else", ""):
        assert role_of_identity(bad) is None


# ── 老客户端兼容桥 ───────────────────────────────────────────────────


def test_legacy_client_still_gets_principal():
    """只发老键的旧 app 不能从升级那天起彻底失灵。"""
    assert wanted_roles({"old": {ATTR_WANT: "true"}}) == {AvatarRole.PRINCIPAL}


def test_legacy_off_wants_nothing():
    assert wanted_roles({"old": {ATTR_WANT: "false"}}) == set()


def test_explicit_false_is_not_overridden_by_legacy_key():
    """⭐ 新客户端**表过态的 false** 不能被老键盖回去。

    新 app 切到语音助手时写的是 principal=false + want=false；但升级窗口里
    两个键的更新不保证同一条信令到达。按「老键为真就加本体」写的话，
    中间那一瞬间本体会被莫名其妙拉起来 —— 一个自己会消失的幽灵。
    """
    attrs = {ATTR_WANT_BY_ROLE[AvatarRole.ASSISTANT]: "true",
             ATTR_WANT_BY_ROLE[AvatarRole.PRINCIPAL]: "false",
             ATTR_WANT: "true"}
    assert wanted_roles({"new": attrs}) == {AvatarRole.ASSISTANT}


def test_new_and_old_clients_coexist():
    """一屋子里新旧客户端各要各的，两个都算数。"""
    people = {"old": {ATTR_WANT: "true"},
              "new": {ATTR_WANT_BY_ROLE[AvatarRole.ASSISTANT]: "true"}}
    assert wanted_roles(people) == {AvatarRole.PRINCIPAL, AvatarRole.ASSISTANT}


# ── 可见性聚合 ───────────────────────────────────────────────────────


def test_any_visible_is_optimistic():
    assert any_visible({"a": {ATTR_VISIBLE: "false"}, "b": {ATTR_VISIBLE: "true"}})


def test_all_hidden_is_not_visible():
    assert not any_visible({"a": {ATTR_VISIBLE: "false"}})


def test_silent_client_counts_as_visible():
    """没报可见性的默认看得见 —— 默认不可见会让它永远停在 hidden 且不报错。"""
    assert any_visible({"a": {}})


def test_empty_room_is_not_visible():
    assert not any_visible({})


# ── 按角色回报的状态键 ───────────────────────────────────────────────


def test_each_role_reports_on_its_own_key():
    keys = set(ATTR_STATE_BY_ROLE.values())
    assert len(keys) == len(AvatarRole)
    assert ATTR_STATE not in keys, "按角色的键不能跟老的全房键重名，否则照样互相覆盖"
