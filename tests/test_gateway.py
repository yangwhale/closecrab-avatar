"""网关单元测试 —— 重点在鉴权和调度这两块「错了就出事」的逻辑。

刻意包含 negative test：只测 happy path 的测试等于没测。
"""
from __future__ import annotations

import time

import pytest

from closecrab_avatar.auth import (
    ApiKey, AuthError, KeyRing, make_terminate_token, sign_client_token,
    verify_bearer, verify_terminate_token,
)
from closecrab_avatar.scheduler import NoCapacity, Scheduler
from closecrab_avatar.store import Session, Store

KEY = ApiKey(key_id="k1", secret="s" * 32, max_concurrency=2, label="test")
OTHER = ApiKey(key_id="k2", secret="t" * 32)
RING = KeyRing({"k1": KEY, "k2": OTHER})


# ── 鉴权 ────────────────────────────────────────────────────
def test_valid_token_passes():
    assert verify_bearer(sign_client_token("k1", KEY.secret), RING).key_id == "k1"


def test_wrong_secret_rejected():
    """用别人的 secret 签、冒充 k1 —— 必须拒。"""
    bad = sign_client_token("k1", OTHER.secret)
    with pytest.raises(AuthError):
        verify_bearer(bad, RING)


def test_unknown_key_rejected():
    with pytest.raises(AuthError):
        verify_bearer(sign_client_token("k9", "whatever"), RING)


def test_expired_token_rejected():
    with pytest.raises(AuthError):
        verify_bearer(sign_client_token("k1", KEY.secret, ttl_s=-10), RING)


def test_token_without_exp_rejected():
    """没有 exp 的 token 永不过期 —— 必须拒，否则泄漏一次等于永久失守。"""
    import jwt
    forever = jwt.encode({"iss": "k1"}, KEY.secret, algorithm="HS256")
    with pytest.raises(AuthError):
        verify_bearer(forever, RING)


def test_garbage_token_rejected():
    with pytest.raises(AuthError):
        verify_bearer("not-a-jwt", RING)


# ── terminate_token ─────────────────────────────────────────
def test_terminate_token_roundtrip():
    t = make_terminate_token(KEY.secret, "lap_x")
    assert verify_terminate_token(KEY.secret, "lap_x", t)


def test_terminate_token_wrong_session_rejected():
    """拿 A 会话的 token 去终止 B 会话 —— 必须拒。"""
    t = make_terminate_token(KEY.secret, "lap_a")
    assert not verify_terminate_token(KEY.secret, "lap_b", t)


def test_terminate_token_other_tenant_rejected():
    t = make_terminate_token(OTHER.secret, "lap_x")
    assert not verify_terminate_token(KEY.secret, "lap_x", t)


# ── 调度 ────────────────────────────────────────────────────
@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _sess(psid: str, worker: str, key_id: str = "k1") -> Session:
    now = time.time()
    return Session(psid, "las_" + psid, worker, key_id, "room", "pending", now, now, {})


def test_no_workers_means_no_capacity(store):
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=30).pick_worker()


def test_picks_least_loaded(store):
    store.upsert_worker("w1", 2, {})
    store.upsert_worker("w2", 2, {})
    store.create_session(_sess("p1", "w1"))
    assert Scheduler(store, heartbeat_timeout_s=30).pick_worker().worker_id == "w2"


def test_full_pool_raises(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=30).pick_worker()


def test_stale_worker_not_scheduled(store):
    """心跳早就断了的 worker 不能再派活给它。"""
    store.upsert_worker("w1", 1, {})
    with pytest.raises(NoCapacity):
        Scheduler(store, heartbeat_timeout_s=-1).pick_worker()


def test_closed_session_frees_slot(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    store.set_state("p1", "closed")
    assert Scheduler(store, heartbeat_timeout_s=30).pick_worker().worker_id == "w1"


def test_claim_pending_is_exactly_once(store):
    """两次 claim 不能拿到同一个活 —— 否则一路会被派给两张卡。"""
    store.upsert_worker("w1", 2, {})
    store.create_session(_sess("p1", "w1"))
    assert store.claim_pending("w1").provider_session_id == "p1"
    assert store.claim_pending("w1") is None


def test_idle_reap_closes_session(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    reaped = store.reap(idle_timeout_s=-1, max_session_s=10_000)
    assert [s.provider_session_id for s in reaped] == ["p1"]
    assert store.get_session("p1").state == "closed"


def test_max_session_reap_closes_even_if_active(store):
    """一直有音频也不能永远占着卡。"""
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    store.touch_session("p1")
    assert len(store.reap(idle_timeout_s=10_000, max_session_s=-1)) == 1


def test_dead_worker_sweep_frees_its_sessions(store):
    store.upsert_worker("w1", 1, {})
    store.create_session(_sess("p1", "w1"))
    assert Scheduler(store, heartbeat_timeout_s=-1).sweep_dead_workers() == ["w1"]
    assert store.get_session("p1").state == "closed"


def test_per_key_concurrency_is_counted(store):
    store.upsert_worker("w1", 8, {})
    store.create_session(_sess("p1", "w1", "k1"))
    store.create_session(_sess("p2", "w1", "k2"))
    assert store.active_for_key("k1") == 1
    assert store.active_for_key("k2") == 1


def test_idempotency_replays_first_result(store):
    store.put_idempotent("idem-1", "k1", {"session_id": "A"})
    store.put_idempotent("idem-1", "k1", {"session_id": "B"})   # 重试
    assert store.get_idempotent("idem-1", "k1")["session_id"] == "A"


def test_idempotency_is_per_key(store):
    """别的租户用同一个 Idempotency-Key 不能读到你的结果。"""
    store.put_idempotent("idem-1", "k1", {"session_id": "A"})
    assert store.get_idempotent("idem-1", "k2") is None


# ── 2026-09-17 部署时抓到的：空闲 worker 死掉是完全无声的 ──────────────

def test_sweeps_an_idle_worker_that_died(tmp_path):
    """⭐ 一个**从没接过活**的 worker 心跳断了，也必须被摘掉并报出来。

    原实现遍历的是 `active_counts()` —— 只含有 pending/active 会话的 worker。
    空闲 worker 一条会话都没有，于是永远不在那个字典里：
    行留在表里、`sweep_dead_workers()` 返回空、**那条 warning 一个字都不打**。

    而 `docs/deployment.md` 教人「槽位不对就去日志里查心跳丢失」——
    最该报警的情况恰恰完全无声。骗人的地方在于 `capacity()` 走的是
    `live_workers()`，槽位数**会**正确地掉下去，看起来一切正常。
    """
    import time as _t
    from closecrab_avatar.scheduler import Scheduler
    from closecrab_avatar.store import Store

    store = Store(str(tmp_path / "s.db"))
    sched = Scheduler(store, heartbeat_timeout_s=0.05)
    store.upsert_worker("idle-gpu", 8, {})          # 注册完就再没动静

    assert sched.capacity().total == 8
    _t.sleep(0.08)                                   # 心跳过期

    assert sched.sweep_dead_workers() == ["idle-gpu"], "空闲 worker 没被摘"
    assert sched.capacity().total == 0
    # 摘干净了：行要真的没了，否则 Spot 每重建一次多一行
    assert sched.sweep_dead_workers() == [], "摘完还在，drop_worker 没生效"


def test_still_sweeps_a_busy_worker(tmp_path):
    """原来那条路径不能修坏：有会话的 worker 照样要摘，会话置 closed 不删。"""
    import time as _t
    from closecrab_avatar.scheduler import Scheduler
    from closecrab_avatar.store import Store

    store = Store(str(tmp_path / "s.db"))
    sched = Scheduler(store, heartbeat_timeout_s=0.05)
    store.upsert_worker("busy-gpu", 4, {})
    store.create_session(_sess("p-busy", "busy-gpu"))

    _t.sleep(0.08)
    assert sched.sweep_dead_workers() == ["busy-gpu"]
    assert sched.capacity().used == 0, "会话没被关掉，槽位仍被占着"


# ── 形象库端点 ────────────────────────────────────────────────────

def _client(tmp_path):
    """起一个真的 FastAPI 应用，走 HTTP 层。

    上面那些测试是单元级的（直接调函数）；形象库这几条必须走 HTTP ——
    要验的恰恰是**路由层**的东西：原始字节怎么进来、错误落成哪个状态码、
    ETag 有没有带上。绕过路由就等于没测。
    """
    from fastapi.testclient import TestClient

    from closecrab_avatar.app import create_app
    from closecrab_avatar.config import Settings

    settings = Settings(
        livekit_api_key="devkey", livekit_api_secret="d" * 32,
        db_path=":memory:", persona_dir=str(tmp_path / "personas"))
    app = create_app(settings, RING)
    return TestClient(app), {"Authorization": f"Bearer {sign_client_token('k1', KEY.secret)}"}


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def test_persona_upload_get_and_image(tmp_path, monkeypatch):
    """三个端点串一遍：传 → 问元数据 → 取图。"""
    c, auth = _client(tmp_path)
    assert c.get("/avatar/persona/bunny", headers=auth).status_code == 404

    r = c.put("/avatar/persona/bunny", content=JPEG_BYTES, headers=auth)
    assert r.status_code == 200, r.text
    ver = r.json()["version"]

    m = c.get("/avatar/persona/bunny", headers=auth)
    assert m.status_code == 200 and m.json()["version"] == ver

    img = c.get("/avatar/persona/bunny/image", headers=auth)
    assert img.status_code == 200
    assert img.content == JPEG_BYTES
    assert img.headers["content-type"].startswith("image/jpeg")
    # ETag 是 worker 判断「要不要重开循环」的依据，不能丢。
    assert ver in img.headers.get("etag", "")


def test_persona_rejects_non_image(tmp_path):
    """⭐ 400 不是 500 —— 是调用方给错了东西，不是我们坏了。"""
    c, auth = _client(tmp_path)
    r = c.put("/avatar/persona/bunny", content=b"<html>nope</html>", headers=auth)
    assert r.status_code == 400 and "认不出" in r.text


def test_persona_rejects_empty_body(tmp_path):
    c, auth = _client(tmp_path)
    assert c.put("/avatar/persona/bunny", content=b"", headers=auth).status_code == 400


def test_persona_rejects_path_traversal(tmp_path):
    """⭐ 房间名进了文件路径，`..` 必须被挡在 400。"""
    c, auth = _client(tmp_path)
    r = c.put("/avatar/persona/..%2F..%2Fetc%2Fpasswd", content=JPEG_BYTES, headers=auth)
    assert r.status_code in (400, 404), r.status_code


def test_persona_needs_auth(tmp_path):
    c, _ = _client(tmp_path)
    assert c.put("/avatar/persona/bunny", content=JPEG_BYTES).status_code == 401
    assert c.get("/avatar/persona/bunny/image").status_code == 401
