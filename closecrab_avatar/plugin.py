"""给 **agent 侧**用的 livekit-agents 插件 —— 一行接上数字人。

    from closecrab_avatar.plugin import CloseCrabAvatar

    avatar = CloseCrabAvatar(gateway_url=..., key_id=..., secret=...)
    await avatar.start(agent_session, room=ctx.room)

## 它到底改了什么

平时 agent 把 TTS 音频**直接发布**成一条音轨。接上数字人之后，音频改为
经 LiveKit DataStream **定向发给数字人那个参与者**，由它对口型、再把
音视频一起发布出来。所以：

- 房间里的听众听到的声音来自**数字人**，不是 agent
- agent 自己不再发布音轨

这不是我们发明的，是 `livekit-agents` 既定的 avatar 契约
（`AvatarSession` / `DataStreamAudioOutput` / `DataStreamAudioReceiver`）。
我们只是把「谁来当这个数字人」换成自己的控制面。

## ⚠️ 一定要在 `agent_session.start()` **之后**调

`start()` 里要改 `agent_session.output.audio`，而那个对象是
`session.start()` 建 RoomIO 时才有的。提前调拿到的是 `None`，
**不报错**，只是音频照旧直接发布 —— 数字人在房间里张着嘴没声音。
"""

from __future__ import annotations

import logging
from typing import Any

from .auth import sign_client_token

log = logging.getLogger("closecrab.avatar.plugin")

DEFAULT_AVATAR_IDENTITY = "cc-avatar"


class CloseCrabAvatarError(RuntimeError):
    pass


class CloseCrabAvatar:
    """CloseCrab Avatar 的 `livekit-agents` 插件。

    ⚠️ **不在模块顶层继承 `AvatarSession`** —— 那要 import `livekit.agents`，
    而控制面那台机器上没装（它只要 `livekit-api` 铸票）。基类在
    `start()` 里才拿，这个类靠鸭子类型满足同样的契约。

    真要 `isinstance` 检查的场合用 `as_avatar_session()`。
    """

    def __init__(self, *, gateway_url: str, key_id: str, secret: str,
                 livekit_url: str,
                 livekit_api_key: str = "", livekit_api_secret: str = "",
                 avatar_identity: str = DEFAULT_AVATAR_IDENTITY,
                 avatar_name: str = "CloseCrab Avatar",
                 sample_rate: int = 16000,
                 size: str = "720*400",
                 join_timeout: float = 60.0,
                 http_timeout: float = 10.0):
        self._gw = gateway_url.rstrip("/")
        self._key_id = key_id
        self._secret = secret
        self._livekit_url = livekit_url
        self._lk_key = livekit_api_key
        self._lk_secret = livekit_api_secret
        self._identity = avatar_identity
        self._name = avatar_name
        self._sample_rate = sample_rate
        self._size = size
        self._join_timeout = join_timeout
        self._http_timeout = http_timeout

        self._session_id: str | None = None
        self._terminate_token: str | None = None
        self._prev_audio_out: Any = None
        self._agent_session: Any = None
        self._room: Any = None

    # ── 契约 ──────────────────────────────────────────────────────

    @property
    def avatar_identity(self) -> str:
        return self._identity

    @property
    def provider(self) -> str:
        return "closecrab"

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def active(self) -> bool:
        return self._session_id is not None

    # ── 起 ────────────────────────────────────────────────────────

    async def start(self, agent_session: Any, room: Any) -> None:
        """派一个数字人进房，并把 agent 的音频改道给它。

        **顺序是有讲究的**：先派人、再改道。反过来的话，改道之后到数字人
        真进房之间那段时间，agent 说的话会被发进一个没人收的 DataStream ——
        听众那几秒是彻底静音的，而且不报错。
        """
        import aiohttp
        from livekit.agents.voice.avatar import DataStreamAudioOutput

        if self._session_id is not None:
            return
        self._agent_session = agent_session
        self._room = room

        body = {
            "provider": "liveavatar",
            "livekit_url": self._livekit_url,
            "room_name": room.name,
            "room_sid": await _room_sid(room),
            "avatar_identity": self._identity,
            "avatar_name": self._name,
            "agent_identity": room.local_participant.identity,
            "size": self._size,
            "sample_rate": self._sample_rate,
        }
        headers = {"Authorization":
                   f"Bearer {sign_client_token(self._key_id, self._secret)}"}
        timeout = aiohttp.ClientTimeout(total=self._http_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(f"{self._gw}/avatar/sessions",
                                 json=body, headers=headers) as r:
                text = await r.text()
                if r.status != 200:
                    # 429 = 槽位满，是**容量问题不是故障**，调用方该退回无数字人
                    # 模式而不是让整通对话失败。所以带上状态码往外抛。
                    raise CloseCrabAvatarError(f"建数字人会话失败 {r.status}：{text[:200]}")
                data = await r.json()
        self._session_id = data["provider_session_id"]
        self._terminate_token = data.get("terminate_token", "")
        log.info("数字人会话 %s 已派出（房间 %s）", self._session_id, room.name)

        # 改道。**记下原来的**，关掉的时候要还回去，否则拨掉开关之后
        # agent 变成哑巴 —— 音频还在往一个已经走了的参与者发。
        self._prev_audio_out = getattr(agent_session.output, "audio", None)
        agent_session.output.audio = DataStreamAudioOutput(
            room, destination_identity=self._identity,
            sample_rate=self._sample_rate)

    async def wait_for_join(self, *, timeout: float | None = None) -> None:
        """等数字人真的进房。**进不来要当失败处理**，别默默继续。"""
        import asyncio

        from livekit.agents import utils

        await asyncio.wait_for(
            utils.wait_for_participant(room=self._room, identity=self._identity),
            timeout if timeout is not None else self._join_timeout)

    # ── 收 ────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """还槽位、把音频改回直接发布。**两件事都要做，哪件漏了都很难查。**

        - 不还槽位：下一次建会话 429「所有槽位都忙」，看起来像容量不够
        - 不改回来：agent 从此静音，音频发给一个已经走了的参与者
        """
        import aiohttp

        room = self._room
        if self._agent_session is not None:
            try:
                self._agent_session.output.audio = self._prev_audio_out
            except Exception:                       # noqa: BLE001
                log.warning("音频出口没还回去 —— agent 可能会哑", exc_info=True)

        # ⭐ **把数字人踢出房间。** 只跟控制面说一声是不够的 —— worker 那头在等
        #    agent 离开，控制面结不结账它根本不知道，于是它就一直站在房间里。
        #    实测过：agent 日志一路 on → off、会话也终止了，房间里那个
        #    `cc-avatar` 纹丝不动，客户端还看着一张不动的脸。
        #    （上游 `AvatarSession.aclose()` 本来就有这一步，我们是鸭子类型
        #    实现，没继承到，得自己补。）
        await self._evict(room)

        sid, tok = self._session_id, self._terminate_token
        self._session_id = self._terminate_token = None
        self._agent_session = self._prev_audio_out = self._room = None
        if not sid:
            return
        headers = {"Authorization":
                   f"Bearer {sign_client_token(self._key_id, self._secret)}"}
        try:
            timeout = aiohttp.ClientTimeout(total=self._http_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(
                    f"{self._gw}/avatar/sessions/terminate",
                    json={"provider": "liveavatar", "provider_session_id": sid,
                          "terminate_token": tok},
                    headers=headers,
                ) as r:
                    if r.status != 200:
                        log.warning("数字人会话 %s 没还干净（%s）—— 下次可能 429",
                                    sid, r.status)
        except Exception:                           # noqa: BLE001
            log.warning("还数字人会话时出错，靠控制面的 reaper 兜底", exc_info=True)
        else:
            log.info("数字人会话 %s 已结束", sid)

    # ── 会话重建时重新挂上 ─────────────────────────────────────────

    def rebind(self, agent_session: Any) -> None:
        """换了一个新的 `AgentSession`，把音频改道重新挂上去。

        用得着是因为有些 realtime 模型会周期性掉线重连，上层每次都建一个
        **新的** `AgentSession`。数字人本身不用换（它还在房间里），
        但新会话的音频出口是默认的，不重新挂就直接发布了 ——
        表现是「说了几句之后数字人的嘴不动了，但还有声音」。
        """
        from livekit.agents.voice.avatar import DataStreamAudioOutput

        if not self.active:
            return
        self._agent_session = agent_session
        self._prev_audio_out = getattr(agent_session.output, "audio", None)
        agent_session.output.audio = DataStreamAudioOutput(
            self._room, destination_identity=self._identity,
            sample_rate=self._sample_rate)


    async def _evict(self, room: Any) -> None:
        """把数字人从房间里请出去。踢掉之后 worker 那边会收到
        `disconnected`，自己收摊。

        优先用 job context 里那个 api client（agent 运行时都有），
        没有再用显式凭据。两个都没有就明说 —— **不要静默跳过**，
        跳过的后果是房间里留一个不动的人，比报错难查得多。
        """
        from livekit import api

        if room is None:
            return
        req = api.RoomParticipantIdentity(room=room.name, identity=self._identity)
        try:
            from livekit.agents import get_job_context

            ctx = get_job_context(required=False)
        except Exception:                           # noqa: BLE001
            ctx = None

        if ctx is not None:
            await ctx.api.room.remove_participant(req)
            return
        if not (self._lk_key and self._lk_secret):
            log.warning("踢不掉数字人：既不在 job context 里，也没给 LiveKit 凭据。"
                        "房间里会留一个不动的人")
            return
        lk = api.LiveKitAPI(self._livekit_url.replace("ws://", "http://")
                            .replace("wss://", "https://"),
                            self._lk_key, self._lk_secret)
        try:
            await lk.room.remove_participant(req)
        finally:
            await lk.aclose()


async def _room_sid(room: Any) -> str:
    """`room.sid` 在新版 SDK 里是 awaitable，老版是字符串。两种都吃。"""
    sid = room.sid
    if hasattr(sid, "__await__"):
        sid = await sid
    return str(sid)
