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

        job = {
            "provider_session_id": provider_session_id,
            "livekit_url": body.livekit_url,
            "room_name": body.room_name,
            "room_token": room_token,          # ⛔ 只下发给 worker，绝不回给调用方
            "avatar_identity": body.avatar_identity,
            "agent_identity": body.agent_identity,
            "avatar_id": body.avatar_id,
            "image_url": body.image_url,
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

    @app.put("/avatar/persona/{room}")
    async def put_persona(room: str, request: Request,
                          note: str = "",
                          api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        data = await request.body()
        if not data:
            raise HTTPException(400, "请求体是空的 —— 图片要放在 body 里发原始字节")
        try:
            p = personas.put(room, data, note=note)
        except PersonaError as e:
            # 400 不是 500：这是调用方给错了东西，不是我们坏了。
            raise HTTPException(400, str(e)) from e
        return {"status": "ok", **p.to_json()}

    @app.get("/avatar/persona/{room}")
    async def get_persona(room: str, api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        try:
            p = personas.get(room)
        except PersonaError as e:
            raise HTTPException(400, str(e)) from e
        if p is None:
            # **404 而不是空对象** —— 「没设置过」和「设置成空」是两件事，
            # 客户端要据此决定显示默认脸还是显示上传过的那张。
            raise HTTPException(404, f"房间 {room} 还没设过形象图")
        return p.to_json()

    @app.get("/avatar/persona/{room}/image")
    async def get_persona_image(room: str, api_key: ApiKey = Depends(auth)) -> Response:
        try:
            got = personas.read_image(room)
        except PersonaError as e:
            raise HTTPException(400, str(e)) from e
        if got is None:
            raise HTTPException(404, f"房间 {room} 还没设过形象图")
        data, ctype = got
        meta = personas.get(room)
        # 带上版本当 ETag：客户端和 worker 都靠它判断「要不要重新拉」。
        headers = {"ETag": f'"{meta.version}"'} if meta else {}
        return Response(content=data, media_type=ctype, headers=headers)

    @app.delete("/avatar/persona/{room}")
    async def del_persona(room: str, api_key: ApiKey = Depends(auth)) -> dict[str, Any]:
        try:
            existed = personas.delete(room)
        except PersonaError as e:
            raise HTTPException(400, str(e)) from e
        return {"status": "deleted" if existed else "nothing-to-delete"}

    # ── 运维 ──────────────────────────────────────────────────
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        cap = sched.capacity()
        return {"status": "ok", "slots_total": cap.total, "slots_used": cap.used,
                "slots_free": cap.free, "keys": len(ring),
                "personas": personas.rooms()}

    # ── 内部：worker 注册与长轮询（不对外暴露）──────────────────
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
