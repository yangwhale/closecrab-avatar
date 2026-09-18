"""要不要给这一路开数字人 —— 纯判定，不碰网络不碰 LiveKit。

**这是产品的对外契约，别在别处复刻。** 属性名（`cc.avatar.want` /
`cc.client.visible` / `cc.avatar.state`）和状态机归这个文件定义，
客户端和 agent 两边照着它实现；改这里就是改契约。

> 2026-09-18 从 CloseCrab 的 `closecrab/voice/avatar_policy.py` 搬过来。
> 那份当时挂在 bot 的 TTS 出口上 —— 房间是对的，但**做判定的必须是房间里
> 那个会说话的 agent**，因为只有它能把自己的音频改道给数字人。
> 判定和改道分在两个进程里，结果就是「状态写着 on，却没人改道」。


单独一个文件、只依赖标准库，是为了**能离线把它钉死**。这个判断有三个输入、
四种结果，而且错了都是「静默」的那种：该开没开，用户以为坏了；不该开却开了，
白占一路 GPU 而屏幕上根本看不见。

## 三个输入，两边各管各的

    客户端说：  want     用户那个开关拨没拨
                visible  app 现在看得见吗（前台 / 后台 / 锁屏）
    服务端说：  service  数字人服务可不可用（网关通不通、还有没有空槽）

**客户端不做最终决定，服务端也不猜客户端的意图** —— 各报各的事实，
在这里合成。

## 为什么 visible 这一条值钱

不是省一点，是**省掉一整张 GPU 卡**：网关一共 8 路槽位，给一个没人看的画面
占着一路是纯浪费。而且 app 在后台时本来就只出声（iOS 上锁屏没法显示实时视频，
PiP 锁屏即停、CallKit 显示不了远端画面、Live Activity 是快照）。

## ⚠️ 回滞不在这里

「后台多久才算真的看不见」是**客户端**的事（见 iOS 侧 CCAvatarPresence）。
这里收到的 `visible` 已经是去抖之后的结论。
放在这一层会让服务端凭空多一个计时器，而它根本不知道用户刚才是不是
只拉了一下通知中心。
"""

from __future__ import annotations

import os
from enum import Enum


class AvatarState(str, Enum):
    """最终状态。**同时也是回报给客户端的那个值** —— 客户端据此决定
    是显示「正在接入」还是「数字人暂不可用」，而不是干等一条永远不来的视频轨。
    """

    ON = "on"
    """开。"""

    OFF = "off"
    """用户自己关的。客户端不该显示任何「出错了」的提示。"""

    HIDDEN = "hidden"
    """用户要，但他现在看不见（后台/锁屏），所以先不生成。
    回前台会自己变回 on —— **这不是错误状态**，客户端别报错。"""

    UNAVAILABLE = "unavailable"
    """用户要、也看得见，但服务端给不了（网关挂了 / 槽位满了）。
    **这一条必须让用户看见** —— 否则他只会觉得「怎么没有脸」。"""


RESPECT_VISIBILITY = os.environ.get("CC_AVATAR_RESPECT_VISIBILITY", "0") == "1"
"""要不要把「app 在不在前台」算进判定。**2026-09-18 起默认关掉。**

Chris 的原话：「app 那个前台判断也先不要了，简化逻辑先，就看数字人手动开关」。

关掉的理由不只是省事 —— 它跟实际用法冲突：数字人由 bot 的播报驱动，而让
bot 播报得先去飞书里说话，一说话 app 就进后台了。于是「想看数字人」和
「能让它说话」这两件事**在操作上互斥**，判定结果永远停在 `hidden`。

代价说清楚：锁屏和后台时照样会挂一路 GPU 渲染，而屏幕上根本看不见 ——
那正是当初加这一条要省的东西。所以它是暂时关掉，不是删掉：
设 `CC_AVATAR_RESPECT_VISIBILITY=1` 就回来。

客户端**仍然照常上报** `cc.client.visible`，只是这一层不再据此决策 ——
数据留着，将来要恢复不用改客户端。
"""


def decide(*, want: bool, visible: bool, service_ok: bool,
           respect_visibility: bool | None = None) -> AvatarState:
    """把输入合成一个状态。

    判断顺序是有讲究的，**不能换**：

    1. `want` 最先 —— 用户关掉了就到此为止。后面两条再怎么样都不该
       让客户端看到 `unavailable` 之类的提示：他没要，服务好不好跟他无关。
    2. `visible` 其次（**默认已停用**，见 `RESPECT_VISIBILITY`）—— 启用时
       它**必须排在 service 前面**：服务正好挂了、而用户又在后台时，
       该报的是 `hidden` 不是 `unavailable`；报后者会让他回前台后看到
       一条「暂不可用」的假警报。
    3. `service_ok` 最后 —— 到这儿才是真的「想要、能看见、但给不了」。
    """
    if respect_visibility is None:
        respect_visibility = RESPECT_VISIBILITY
    if not want:
        return AvatarState.OFF
    if respect_visibility and not visible:
        return AvatarState.HIDDEN
    if not service_ok:
        return AvatarState.UNAVAILABLE
    return AvatarState.ON


# LiveKit 的 participant attributes 只能装字符串，所以布尔要自己解。
ATTR_WANT = "cc.avatar.want"
"""客户端写：用户那个开关。"""

ATTR_VISIBLE = "cc.client.visible"
"""客户端写：现在看得见吗（已经过客户端侧去抖）。"""

ATTR_STATE = "cc.avatar.state"
"""**服务端写**，回报最终状态。客户端订阅它来决定界面怎么显示。"""

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def parse_flag(raw: str | None, *, default: bool) -> bool:
    """把属性里的字符串解成布尔。**认不出来就用 default，不抛异常。**

    不抛是故意的：这条路径在房间事件回调里，抛出去只会让一次属性变更
    整个丢掉，而客户端不会重发 —— 状态就永久卡住了。认不出来退回默认值
    至少还是个确定的行为。
    """
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return default


def decide_from_attributes(attrs: dict[str, str], *, service_ok: bool) -> AvatarState:
    """从客户端属性直接算状态。两个默认值**方向相反**，都是故意的：

    - `want` 缺省 **False** —— 老客户端根本不发这个属性。默认开的话，
      每个连进来的旧客户端都会去抢一路 GPU 槽位，而它连显示的界面都没有。
    - `visible` 缺省 **True** —— 一个已经明确要 avatar、却没报可见性的客户端
      （比如只实现了一半的版本），默认成不可见会让它**永远停在 hidden**，
      而且没有任何报错，纯粹「功能不工作」。默认可见最坏是白渲染一会儿。

    一句话：**没表态的默认关，表了态但信息不全的默认放行。**
    """
    return decide(
        want=parse_flag(attrs.get(ATTR_WANT), default=False),
        visible=parse_flag(attrs.get(ATTR_VISIBLE), default=True),
        service_ok=service_ok,
    )


def decide_for_room(
    per_participant: dict[str, dict[str, str]], *, service_ok: bool
) -> AvatarState:
    """一屋子客户端合成**一个**状态。

    ## 为什么不能一人一个状态

    数字人是**一条视频轨**，全房共享 —— LiveKit 里没有「只发给某个人」这回事。
    所以「开不开」只能有一个答案。同理，回报用的 `cc.avatar.state` 写在
    我们自己这个 participant 上，也只有一份。

    ## 聚合规则：任何一个人要、且看得见，就开

    这是**故意偏向开**的：Chris 在 iPhone 上看着，屋里另一个人把开关关了 ——
    这时候必须继续发，否则关掉开关的那个人把别人的画面也掐了。
    一条轨的成本是固定的，多一个人看不多花钱。

    ## 回报的是「网关这一路的状态」，不是「你的状态」

    所以客户端**不能**把它当成唯一判据 —— 自己开关是关的却看到 `on`，
    那是别人开着，正常。客户端的规则是：自己开关开着 **且** 收到
    `unavailable` 才提示用户。两侧都要按这个约定写，见 iOS 的 `CCAvatarLink`。
    """
    if not per_participant:
        return AvatarState.OFF
    states = [
        decide_from_attributes(attrs, service_ok=service_ok)
        for attrs in per_participant.values()
    ]
    # 按「谁的诉求最强」排：有人真的在看 > 有人要但服务给不了 >
    # 有人要但看不见 > 没人要。**顺序跟 decide() 里的不是一回事**，
    # 那边是三个条件的短路，这边是多个已定状态之间的优先级。
    for s in (AvatarState.ON, AvatarState.UNAVAILABLE, AvatarState.HIDDEN):
        if s in states:
            return s
    return AvatarState.OFF


def should_generate(state: AvatarState) -> bool:
    """这个状态下要不要真的去占一路 worker。

    单独抽出来而不是让调用方写 `state == ON`：以后要加状态（比如
    `DEGRADED` 只发低帧率）时，占不占资源这件事有一个地方改。
    """
    return state is AvatarState.ON


def is_user_visible_problem(state: AvatarState) -> bool:
    """这个状态该不该在客户端界面上报出来。

    只有 `UNAVAILABLE` 该报 —— `OFF` 是用户自己关的，`HIDDEN` 是他看不见
    （也就无所谓看不看得到提示），两个都报出来就是噪音。
    """
    return state is AvatarState.UNAVAILABLE


# ── 每个角色各有一个开关，分配归服务端 ──────────────────────────────
#
# Chris 2026-09-18：「每一个角色应该有自己的 property。Bunny 有一个开和关，
# 语音助手有一个开和关。后台到时候可以按照资源来决定把 Live Avatar 给谁。」
#
# ## 为什么不是一个「归谁」的单选
#
# 我第一版写成了单选（`cc.avatar.target` = assistant / principal / off），
# **那是把两件事揉在一起了**：
#
#     客户端表达的是**意图** —— 这个角色我想让它有脸
#     服务端做的是**分配** —— 现在有几路 GPU，该给谁
#
# 揉在一起的后果：客户端被迫替服务端做资源决策，而它根本不知道有几路空闲。
# 而且以后多一路 Live Avatar，协议就得跟着改 —— 分开的话协议一个字不用动，
# 只是 `allocate()` 的 capacity 变大。
#
# 现在 iOS 上是「切换」形态（同时只开一个），但**那是客户端的产品选择，
# 不是协议的限制**。协议允许两个都开。
#
# ⚠️ 旧的 `ATTR_WANT` 暂时保留：三层（iOS / bot / 网关）不可能同一秒切换，
#    中间必然有新旧并存的窗口。三层都上了新的再删。


class AvatarRole(str, Enum):
    """谁要这张脸。**名字跟形象库的 `persona_role` 共用一套。**"""

    PRINCIPAL = "principal"
    """本体（bot 自己的播报那一路）。"""

    ASSISTANT = "assistant"
    """语音助手。"""


ATTR_WANT_BY_ROLE = {
    AvatarRole.PRINCIPAL: "cc.avatar.principal",
    AvatarRole.ASSISTANT: "cc.avatar.assistant",
}
"""客户端写：这个角色要不要数字人。**一个角色一个键。**"""

ATTR_STATE_BY_ROLE = {
    AvatarRole.PRINCIPAL: "cc.avatar.state.principal",
    AvatarRole.ASSISTANT: "cc.avatar.state.assistant",
}
"""**服务端写**，按角色回报。每一路写自己那个键，谁也盖不掉谁。

老的 `ATTR_STATE` 是全房共享的一份，两路同时在的时候会互相覆盖 ——
而覆盖的表现是「本体明明开着，客户端却显示 off」，看起来像开关失灵。
所以新增按角色的键；老键仍然写，但**只有被分配到的那一路写**
（理由见 `avatar_link` 里那段）。
"""

# ── ⭐ 一个角色一个数字人，不是「换主人」──────────────────────────
#
# Chris 2026-09-18：「Bunny 用 Bunny 的，语音助手用语音助手的。切换的时候是
# 其中一个关了、另外一个加进来，而不是把房间里那个 Avatar 换主人。因为将来
# 资源多的时候，两个 Avatar 都进来。」
#
# 所以每个被分配到的角色**各起一路会话、各进一个 participant**：
#
#     cc-avatar-principal   publish_on_behalf = <bot>-speaker
#     cc-avatar-assistant   publish_on_behalf = <助手的 identity>
#
# 切换 = 关掉一路会话 + 开另一路，**不是改现有那一路的归属**。
#
# ⚠️ 为什么这个区别要紧：如果做成「换主人」，那么「两个都在」这件事在
#    数据模型里根本表达不出来 —— 等资源变多时要重写一遍，而且客户端认
#    参与者的那套逻辑（按 `lk.publish_on_behalf` 关联）也要跟着改。
#    各带各的，从一路到两路只是**多建一个会话**，别的地方一个字不用动。
#
# ⚠️ 现在的 `cc-avatar` 这个固定 identity 要改成带角色后缀 —— 否则两路同时
#    在房间里会撞名字。改的时候 iOS 那边按 `lk.avatar_provider` 全房扫的
#    写法**不用动**（它本来就不依赖具体名字），只有「哪条轨对应哪个角色」
#    需要按 `publish_on_behalf` 去认。

# 只有一路 Live Avatar 时先给谁。**本体优先** —— Chris 定的。
ALLOC_PRIORITY = (AvatarRole.PRINCIPAL, AvatarRole.ASSISTANT)


def wanted_roles(per_participant: dict[str, dict[str, str]]) -> set[AvatarRole]:
    """一屋子客户端合出「哪些角色被要了」。

    **任何一个人要，就算要** —— 跟 `decide_for_room()` 同一条聚合规则：
    屋里另一个人把开关关了，不该把正在看的人的画面也掐掉。
    一条轨的成本是固定的，多一个人看不多花钱。

    缺省 False：老客户端根本不发这些键，默认开的话每个连进来的旧客户端
    都会去抢一路 GPU，而它连显示的界面都没有。

    ## ⚠️ 老客户端走兼容桥：只发 `cc.avatar.want` 的算「要本体」

    三层（iOS / bot / 语音助手进程）不可能同一秒都升上去。装着旧 app 的人
    只发老键，一个角色键都不发 —— 不桥接的话他的开关**从升级那一刻起
    彻底失效**，而且没有任何报错。

    桥接的判据是「**一个角色键都没有**」，不是「老键为真」：新客户端明确
    把关掉的角色写成 `false`，那是一个表过态的 false，不能被老键盖回去。
    分不清这两者的后果是：新客户端切到语音助手，老键跟着变 false 没问题；
    但如果反过来按「老键为真就加本体」，用户在新 app 上关掉本体、
    老键因为镜像逻辑还是 true 的那一瞬间，本体会被莫名其妙地拉起来。
    """
    out: set[AvatarRole] = set()
    for attrs in per_participant.values():
        said_anything = False
        for role, key in ATTR_WANT_BY_ROLE.items():
            if key in attrs:
                said_anything = True
            if parse_flag(attrs.get(key), default=False):
                out.add(role)
        if not said_anything and parse_flag(attrs.get(ATTR_WANT), default=False):
            out.add(AvatarRole.PRINCIPAL)
    return out


def any_visible(per_participant: dict[str, dict[str, str]]) -> bool:
    """屋里有没有人真的看得见。

    跟 `wanted_roles()` 同一条聚合规则：**任何一个人看得见就算看得见**。
    一条轨的成本是固定的，不能因为屋里有个人把手机扣下了就把别人的画面掐掉。

    缺省 True —— 跟 `decide_from_attributes()` 里那条一致：一个明确要 avatar
    却没报可见性的客户端（只实现了一半的版本），默认成不可见会让它**永远停在
    hidden**，而且没有任何报错，纯粹「功能不工作」。
    """
    return any(parse_flag(a.get(ATTR_VISIBLE), default=True)
               for a in per_participant.values())


def allocate(wanted: set[AvatarRole], *, capacity: int = 1) -> list[AvatarRole]:
    """按现有资源决定实际给谁。**这是服务端的决定，不是客户端的。**

    容量不够时按 `ALLOC_PRIORITY` 取前 N 个 —— 现在只有一路，
    两个都开就给本体。

    ⚠️ 返回**列表不是单个** —— 以后多一路的时候这里一个字不用改，
    协议也不用动。把「现在只有一路」写死进返回类型，等于把一个临时的
    资源现状焊进契约。
    """
    return [r for r in ALLOC_PRIORITY if r in wanted][:max(0, capacity)]


def avatar_identity(role: AvatarRole, *, prefix: str = "cc-avatar") -> str:
    """这一路数字人在房间里叫什么。**必须带角色后缀。**

    两路同时在房间里的时候，固定用 `cc-avatar` 会撞名字 —— LiveKit 里
    identity 是唯一键，撞了的后果是后进的把先进的踢掉，**而且看起来像
    「切换成功了」**：房间里确实只剩一个数字人，只是不是你以为的那个。

    放在契约层而不是各自拼字符串：bot 建会话时用它、网关派活时用它、
    iOS 认参与者时也会照着它推 —— 三处各拼一遍迟早有一处写错，
    而写错的表现是「数字人在房间里但客户端找不到它」。
    """
    return f"{prefix}-{role.value}"


def role_of_identity(identity: str, *, prefix: str = "cc-avatar") -> AvatarRole | None:
    """反过来：从 identity 认出角色。认不出来返回 None，**不猜**。

    猜的话（比如「认不出就当 principal」）会在改名或版本不匹配时把两路
    数字人都算成本体，而那是静默的错位。
    """
    head = f"{prefix}-"
    if not identity.startswith(head):
        return None
    try:
        return AvatarRole(identity[len(head):])
    except ValueError:
        return None
