"""LiveAvatar Gateway —— 对 LiveKit 表现为第 9 家数字人供应商。

对外只有两个端点，形状与 LiveKit 调 HeyGen / Tavus 等供应商时完全一致
（契约取自 livekit/agents/inference/avatar.py）：

    POST /avatar/sessions
    POST /avatar/sessions/terminate

另有一组 /internal/* 给 worker 注册与长轮询用，**不对外暴露**。
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .auth import ApiKey, AuthError, KeyRing, make_terminate_token, verify_bearer, verify_terminate_token
from .config import Settings
from .lk_token import mint_worker_token
from .persona import PersonaError, PersonaStore
from .scheduler import NoCapacity, Scheduler
from .store import Session, Store

log = logging.getLogger("closecrab.avatar")

REAP_INTERVAL_S = 10.0
LONGPOLL_TIMEOUT_S = 25.0
LONGPOLL_TICK_S = 0.25
RETRY_AFTER = {"Retry-After": "5"}


# ── 请求/响应模型（字段名必须逐字对上 LiveKit）────────────────────
class CreateSessionRequest(BaseModel):
    provider: str
    livekit_url: str
    room_name: str
    room_sid: str = ""
    avatar_identity: str
    avatar_name: str = ""
    agent_identity: str
    avatar_id: str | None = None
    image_url: str | None = None
    # ⚠️ 这两个**必须声明出来**。Pydantic 默认忽略未知字段 —— 没声明的话
    #    调用方传了也是静默丢掉，网关照用自己的默认值。
    #    2026-09-18 踩到：agent 侧按 48 kHz 发音频、这里按 16 kHz 建音频源，
    #    worker 在重新发布时崩在
    #    `InvalidState - sample_rate and num_channels don't match`，
    #    而 POST 返回的是 200 —— 参数进了黑洞。
    #
    #    采样率**由发送方定**：它才是产音频的那一个。这里只在没给时兜底。
    sample_rate: int | None = None
    size: str | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class TerminateRequest(BaseModel):
    provider: str
    provider_session_id: str
    terminate_token: str


class WorkerRegister(BaseModel):
    worker_id: str
    capacity: int = 1
    meta: dict[str, Any] = Field(default_factory=dict)


def create_app(settings: Settings, ring: KeyRing) -> FastAPI:
    store = Store(settings.db_path)
    sched = Scheduler(store, heartbeat_timeout_s=settings.worker_heartbeat_timeout_s)
    personas = PersonaStore(settings.persona_dir)

    async def _reaper() -> None:
        while True:
            try:
                for s in store.reap(idle_timeout_s=settings.idle_timeout_s,
                                    max_session_s=settings.max_session_s):
                    log.warning("回收超时会话 %s（room=%s）", s.provider_session_id, s.room_name)
                for wid in sched.sweep_dead_workers():
                    log.warning("worker %s 心跳丢失，已摘除", wid)
            except Exception:
                log.exception("reaper 出错")   # 绝不让它自己死掉
            await asyncio.sleep(REAP_INTERVAL_S)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(_reaper())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="LiveAvatar Gateway", lifespan=lifespan)

    def auth(authorization: str = Header(default="")) -> ApiKey:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "缺少 Bearer token")
        try:
            return verify_bearer(authorization[7:], ring)
        except AuthError as e:
            raise HTTPException(401, str(e)) from e

    # ── 对外：创建会话 ─────────────────────────────────────────
    @app.post("/avatar/sessions")
    async def create_session(
        body: CreateSessionRequest,
        api_key: ApiKey = Depends(auth),
        idempotency_key: str = Header(default="", alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        if body.provider != settings.provider_name:
            raise HTTPException(400, f"provider 必须是 {settings.provider_name}")

        # 重试复用第一次的结果，否则一次重试就多占一张卡
        if idempotency_key:
            cached = store.get_idempotent(idempotency_key, api_key.key_id)
            if cached is not None:
                return cached

        # ⚠️ Retry-After 必须挂在 HTTPException 上。设在注入的 Response 对象上没用 ——
        #    抛异常时 FastAPI 会另建一个响应，那个 header 会被丢掉（冒烟测试抓到过）。
        if store.active_for_key(api_key.key_id) >= api_key.max_concurrency:
            raise HTTPException(429, f"该 key 并发已达上限 {api_key.max_concurrency}",
                                headers=RETRY_AFTER)

        try:
            worker = sched.pick_worker()
        except NoCapacity as e:
            raise HTTPException(429, str(e), headers=RETRY_AFTER) from e

        session_id = f"las_{uuid.uuid4().hex}"
        provider_session_id = f"lap_{uuid.uuid4().hex}"
        avatar_name = body.avatar_name or body.avatar_identity

        room_token = mint_worker_token(
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            room_name=body.room_name,
            avatar_identity=body.avatar_identity,
            avatar_name=avatar_name,
            agent_identity=body.agent_identity,
            provider=settings.provider_name,
            ttl_s=settings.max_session_s + 300,
        )

        # ⭐ 这一场用哪张脸。**在建会话时就定下来**，别让 worker 自己去猜 ——
        #    「哪个角色」这件事只有控制面同时看得见 room 和 agent_identity。
        #
        #    判据：数字人挂在谁名下就用谁的脸。`<bot>-speaker` 是本体播报那一路
        #    （本人），其余按语音助手算。这条跟 iOS 那边 roster 的角色判定
        #    是同一套语义，改一处要两处一起改。
        persona_role = ("principal" if body.agent_identity.endswith("-speaker")
                        else "assistant")
        persona_version = None
        for key in persona_keys_for_read(body.room_name, persona_role):
            try:
                got = personas.get(key)
            except PersonaError:
                got = None
            if got is not None:
                persona_version = got.version
                break

        job = {
            "provider_session_id": provider_session_id,
            "livekit_url": body.livekit_url,
            "room_name": body.room_name,
            "room_token": room_token,          # ⛔ 只下发给 worker，绝不回给调用方
            "avatar_identity": body.avatar_identity,
            "agent_identity": body.agent_identity,
            "avatar_id": body.avatar_id,
            "image_url": body.image_url,
            # worker 拿这三个决定要不要换脸。version 是 None 就是「这个角色
            # 还没设过图」—— worker 保持启动时那张，**不要报错**：
            # 没设过是正常状态，不是故障。
            "persona_role": persona_role,
            "persona_version": persona_version,
            "persona_url": (f"/internal/persona/{body.room_name}/image"
                            f"?role={persona_role}") if persona_version else None,
            "size": body.size or settings.size,
            "trim_k": settings.trim_k,
            "sample_rate": body.sample_rate or settings.sample_rate,
            "extra_kwargs": body.extra_kwargs,
        }
        now = time.time()
        store.create_session(Session(
            provider_session_id=provider_session_id, session_id=session_id,
            worker_id=worker.worker_id, key_id=api_key.key_id, room_name=body.room_name,
            state="pending", created_at=now, last_activity=now, job=job,
        ))

        result = {
            "session_id": session_id,
            "provider_session_id": provider_session_id,
            "terminate_token": make_terminate_token(api_key.secret, provider_session_id),
            "sample_rate": body.sample_rate or settings.sample_rate,
        }
        if idempotency_key:
            store.put_idempotent(idempotency_key, api_key.key_id, result)
        log.info("会话 %s → worker %s（room=%s）", provider_session_id, worker.worker_id, body.room_name)
        return result

    # ── 对外：终止会话 ─────────────────────────────────────────
    @app.post("/avatar/sessions/terminate")
    async def terminate(body: TerminateRequest, api_key: ApiKey = Depends(auth)) -> dict[str, str]:
        # 认 terminate_token，不认调用者身份 —— 契约如此
        if not verify_terminate_token(api_key.secret, body.provider_session_id, body.terminate_token):
            raise HTTPException(403, "terminate_token 不匹配")
        s = store.get_session(body.provider_session_id)
        if s is None:
            raise HTTPException(404, "会话不存在")
        store.set_state(body.provider_session_id, "closed")
        log.info("会话 %s 已终止", body.provider_session_id)
        return {"status": "terminated"}

    # ── 形象库：这个房间现在用哪张脸 ────────────────────────────
    #
    # 参考图原来是 worker 的命令行参数，写死在进程里 —— 换一张要重启五卡，
    # 而且外面看不到当前用的是哪张。这三个端点把它变成可读可写的。
    #
    # ⚠️ 上传走**原始字节**不走 multipart：客户端（iOS / 我的脚本 / curl）
    #    都只是发一张图，multipart 只是多一层编解码和一个容易搞错的边界。

    # 一个房间里**不止一张脸**。Chris 2026-09-18：语音助手和本人各挂各的形象，
    # 将来两边可能同时说话，得分得开。
    #
    # 角色写进存储键（`<房间>--<角色>`）而不是新开一层目录：`PersonaStore` 那套
    # 原子写、历史归档、扩展名清理全是按「一个键一张图」写的，换成两层要重写
    # 一遍那些边界。`--` 做分隔符是因为它过得了 `_safe()` 的白名单，而房间名里
    # 基本不会出现。
    ROLES = {"principal", "assistant"}

    def persona_key(room: str, role: str) -> str:
        if role not in ROLES:
            raise HTTPException(400, f"角色只能是 {sorted(ROLES)}，给的是 {role!r}")
        return f"{room}--{role}"

    def persona_keys_for_read(room: str, role: str) -> list[str]:
        """读的时候按顺序试。

        ⚠️ 第二个是**老键**（没有角色的那种）。加角色之前存的图都在那儿，
        不兜住的话 Chris 之前传的形象会凭空消失一次 —— 而那种「东西没了」
        比报错更难让人相信是升级导致的。只有 principal 兜，
        因为老数据在语义上就是「本人」。
        """
        keys = [persona_key(room, role)]
        if role == "principal":
            keys.append(room)
        return keys

    @app.put("/avatar/persona/{room}")
    async def put_persona(room: str, request: Request,
                          note: str = "", role: str = "principal",
                          api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        data = await request.body()
        if not data:
            raise HTTPException(400, "请求体是空的 —— 图片要放在 body 里发原始字节")
        try:
            p = personas.put(persona_key(room, role), data, note=note)
        except PersonaError as e:
            # 400 不是 500：这是调用方给错了东西，不是我们坏了。
            raise HTTPException(400, str(e)) from e
        return {"status": "ok", "role": role, **p.to_json()}

    @app.get("/avatar/persona/{room}")
    async def get_persona(room: str, role: str = "principal",
                          api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        for key in persona_keys_for_read(room, role):
            try:
                p = personas.get(key)
            except PersonaError as e:
                raise HTTPException(400, str(e)) from e
            if p is not None:
                return {"role": role, **p.to_json()}
        # **404 而不是空对象** —— 「没设置过」和「设置成空」是两件事，
        # 客户端要据此决定显示默认脸还是显示上传过的那张。
        raise HTTPException(404, f"房间 {room} 的 {role} 还没设过形象图")

    @app.get("/avatar/persona/{room}/image")
    async def get_persona_image(room: str, role: str = "principal",
                                api_key: ApiKey = Depends(auth)) -> Response:
        for key in persona_keys_for_read(room, role):
            try:
                got = personas.read_image(key)
            except PersonaError as e:
                raise HTTPException(400, str(e)) from e
            if got is None:
                continue
            data, ctype = got
            meta = personas.get(key)
            # 带上版本当 ETag：客户端和 worker 都靠它判断「要不要重新拉」。
            # ⚠️ 角色要进 ETag —— 两个角色的版本号各算各的，会撞。撞了的后果是
            #    换了头像客户端还显示旧的，而且**不报错**。
            headers = {"ETag": f'"{role}-{meta.version}"'} if meta else {}
            return Response(content=data, media_type=ctype, headers=headers)
        raise HTTPException(404, f"房间 {room} 的 {role} 还没设过形象图")

    @app.delete("/avatar/persona/{room}")
    async def del_persona(room: str, role: str = "principal",
                          api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        try:
            existed = personas.delete(persona_key(room, role))
        except PersonaError as e:
            raise HTTPException(400, str(e)) from e
        return {"status": "deleted" if existed else "nothing-to-delete", "role": role}

    # ── 运维 ──────────────────────────────────────────────────
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        cap = sched.capacity()
        return {"status": "ok", "slots_total": cap.total, "slots_used": cap.used,
                "slots_free": cap.free, "keys": len(ring),
                "personas": personas.rooms()}

    # ── 内部：worker 注册与长轮询（不对外暴露）──────────────────
    @app.get("/internal/persona/{room}/image")
    async def internal_persona_image(room: str, role: str = "principal") -> Response:
        """给 worker 取参考图。**不鉴权**，跟其余 `/internal/*` 一致。

        为什么不让 worker 走对外那个端点：worker 手上没有客户端密钥，
        给它一份意味着多一个要轮换、要分发的副本 —— 而它本来就在 VPC 内网，
        跟控制面同一层信任域。（今天刚因为「一个密钥三处副本」漏掉一处栽过。）
        """
        for key in persona_keys_for_read(room, role):
            try:
                got = personas.read_image(key)
            except PersonaError:
                got = None
            if got is None:
                continue
            data, ctype = got
            meta = personas.get(key)
            headers = {"ETag": f'"{role}-{meta.version}"'} if meta else {}
            return Response(content=data, media_type=ctype, headers=headers)
        raise HTTPException(404, f"房间 {room} 的 {role} 还没设过形象图")

    @app.post("/internal/workers/register")
    async def register(body: WorkerRegister) -> dict[str, Any]:
        store.upsert_worker(body.worker_id, body.capacity, body.meta)
        log.info("worker %s 上线，容量 %d", body.worker_id, body.capacity)
        return {"status": "registered", "idle_timeout_s": settings.idle_timeout_s}

    @app.post("/internal/workers/{worker_id}/heartbeat")
    async def heartbeat(worker_id: str, payload: dict[str, Any] | None = None) -> dict[str, str]:
        if not store.touch_worker(worker_id):
            # 被 reaper 摘掉过（心跳断过）→ 让它重新注册，别让它以为自己还在册
            raise HTTPException(410, "worker 未注册，请重新 register")
        for psid in (payload or {}).get("active_sessions", []):
            store.touch_session(psid)     # 有音频流动就算活着
        return {"status": "ok"}

    @app.get("/internal/workers/{worker_id}/jobs")
    async def poll_jobs(worker_id: str, request: Request) -> dict[str, Any]:
        """长轮询。有活立刻返回，没活挂到超时返回空 —— 省掉轮询风暴。"""
        deadline = time.monotonic() + LONGPOLL_TIMEOUT_S
        while time.monotonic() < deadline:
            if await request.is_disconnected():
                return {"job": None}
            s = store.claim_pending(worker_id)
            if s is not None:
                return {"job": s.job}
            await asyncio.sleep(LONGPOLL_TICK_S)
        return {"job": None}

    @app.post("/internal/sessions/{provider_session_id}/closed")
    async def worker_closed(provider_session_id: str) -> dict[str, str]:
        store.set_state(provider_session_id, "closed")
        return {"status": "ok"}

    return app
