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


# ── ⭐ 新契约：数字人归谁 ──────────────────────────────────────────

def test_target_parses_three_values():
    from closecrab_avatar.policy import AvatarTarget as T, parse_target
    assert parse_target("assistant") is T.ASSISTANT
    assert parse_target("principal") is T.PRINCIPAL
    assert parse_target("off") is T.OFF


def test_unknown_target_falls_back_to_off_not_previous():
    """⭐ 认不出来当 OFF。

    陌生值多半来自版本不匹配的客户端 —— 让它**不占 GPU** 比让它继续占着安全。
    而且绝不能抛：这条路径在房间事件回调里，抛出去那次属性变更整个丢掉，
    客户端不会重发，状态永久卡住。
    """
    from closecrab_avatar.policy import AvatarTarget as T, parse_target
    for bad in (None, "", "   ", "乱写", "true", "1"):
        assert parse_target(bad) is T.OFF


def test_room_target_is_stable_under_conflict():
    """⭐ 两个人选了不同角色时，结果**不能取决于字典顺序**。

    不稳定的话画面会在两个角色之间来回跳，而且复现不了。
    按 identity 排序取第一个明确表态的：任意但确定。
    """
    from closecrab_avatar.policy import ATTR_TARGET, AvatarTarget as T, target_for_room
    a = {"zoe": {ATTR_TARGET: "principal"}, "amy": {ATTR_TARGET: "assistant"}}
    b = {"amy": {ATTR_TARGET: "assistant"}, "zoe": {ATTR_TARGET: "principal"}}
    assert target_for_room(a) is target_for_room(b) is T.ASSISTANT


def test_room_target_off_when_nobody_picks():
    from closecrab_avatar.policy import ATTR_TARGET, AvatarTarget as T, target_for_room
    assert target_for_room({}) is T.OFF
    assert target_for_room({"a": {}, "b": {ATTR_TARGET: "off"}}) is T.OFF


def test_target_role_names_match_persona_roles():
    """⭐ 角色名必须跟形象库那套**完全一致**。

    两处各起一套名字的话迟早对不上，而对不上的表现是
    「换了助手的图，兔子的脸变了」—— 不报错，只是张冠李戴。
    """
    from closecrab_avatar.policy import AvatarTarget as T
    assert {T.ASSISTANT.value, T.PRINCIPAL.value} == {"assistant", "principal"}
