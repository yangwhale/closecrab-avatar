# 对外 API

字段名**不可改** —— 它们是 LiveKit 官方 avatar 供应商契约，
出处 `livekit/agents/inference/avatar.py`。改了客户端 SDK 就不认。

## 建会话

```http
POST /avatar/sessions
Authorization: Bearer <客户端用 api_key/api_secret 签的短期 JWT>
Idempotency-Key: <每次 start() 一个>

{"provider":"closecrab","livekit_url":...,"room_name":...,"room_sid":...,
 "avatar_identity":...,"avatar_name":...,"agent_identity":...,
 "avatar_id":...|"image_url":...,"extra_kwargs":{}}

→ 200 {"session_id","provider_session_id","terminate_token","sample_rate"}
→ 429 {"detail":"没有空闲槽位"}   带 Retry-After
```

## 停会话

```http
POST /avatar/sessions/terminate
{"provider","provider_session_id","terminate_token"}
```

## 健康

```http
GET /healthz
→ {"status":"ok","slots_total":8,"slots_used":3,"slots_free":5,"keys":2}
```

## 内部（worker 用，**不要对外暴露**）

```
POST /internal/workers/register           {"worker_id","capacity","meta"}
POST /internal/workers/{id}/heartbeat     {"active_sessions":[...]}
GET  /internal/workers/{id}/jobs          长轮询取活
POST /internal/sessions/{psid}/closed
```

这几个**没有鉴权** —— 靠网络隔离。worker 是 pull 模型，只需要
worker → 控制面的**出向** HTTP，GPU 机器不用开任何入向端口。
所以把控制面的端口限制在你自己的网段就够了。

## 三个容易做错的地方

1. **网关自己铸 LiveKit token，客户端不传 token 进来。**
   源码注释（`avatar.py:396`）：roomJoin 限定到 `room_name`，以 `avatar_identity`
   加入，`lk.publish_on_behalf` 设成 `agent_identity`。
   **少了第三条，客户端 SDK 不认这是 agent 的化身**，`Room.agentParticipants`
   查找会落空 —— 前端会以为 agent 没上线。

2. **`terminate_token` 认 token 不认调用者。**
   它是 `HMAC(secret, provider_session_id)`，常数时间比较。
   谁拿到它谁能停那一路 —— 所以它只在建会话的响应里出现一次。

3. **`Retry-After` 要挂在 `HTTPException` 上**，设在注入的 `Response` 对象上
   会被丢掉 —— 抛异常时 FastAPI 另建响应。这个 bug 是端到端冒烟抓到的。
