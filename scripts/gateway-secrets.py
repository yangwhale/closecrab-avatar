#!/usr/bin/env python3
"""网关凭据的两件事：**铸 key** 和 **渲染 env 文件**。

合成一个脚本是因为它们本来就是同一件事的两半 —— 同一个密钥库、同一份文档、
同一个部署流程，拆成两个文件只是当时顺手。

    # 给调用方铸一把 key（幂等；只在铸出来那一次显示 secret）
    gateway-secrets.py mint <key_id> [并发上限] [--rotate]

    # 从密钥库渲染 systemd 的 EnvironmentFile
    gateway-secrets.py render

**这个仓库是公开的**，所以一个项目 ID、数据库名、文档路径都不写死 ——
全部从环境变量来，哪个组织用它就配自己那套：

    LA_FS_PROJECT    密钥库项目
    LA_FS_DATABASE   数据库名（默认 "(default)"）
    LA_FS_LIVEKIT    存 LiveKit 凭据的文档路径，要有 api_key / api_secret
    LA_FS_KEYS       存网关 API key 的文档路径，要有 api_keys: [...]
    LA_ENV_OUT       render 的输出路径

## 两条贯穿全文的规矩

1. **secret 只在铸出来那一次可见。** 之后任何地方都不回显，报错信息里也不带值
   —— 报错经常被原样贴进聊天窗口。丢了就 `--rotate` 重发，别去库里捞。
2. **输出文件 0600 要在创建那一刻定死**（`os.open` 带 mode），
   不能先 `open()` 后 `chmod` —— 那中间有一瞬它是 0644。
"""
from __future__ import annotations

import json
import os
import secrets
import sys


def _need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"缺环境变量 {name}")
    return v


def _db():
    from google.cloud import firestore

    # 显式传 project/database：同名的脏库会静默吞掉读写，
    # 而现象是「读出来是空的」而不是报错。
    return firestore.Client(project=_need("LA_FS_PROJECT"),
                            database=os.environ.get("LA_FS_DATABASE", "(default)"))


# ── mint ────────────────────────────────────────────────────────────

def cmd_mint(args: list[str]) -> None:
    """铸一把 key。**幂等** —— 不幂等的话重跑一次就换了 secret，
    而已经配好的调用方不会知道，它们下一次请求开始 401，看起来像网关坏了。
    要轮换得显式 `--rotate`。
    """
    rotate = "--rotate" in args
    pos = [a for a in args if a != "--rotate"]
    if not pos:
        sys.exit("用法: mint <key_id> [并发上限] [--rotate]")
    key_id = pos[0]
    concurrency = int(pos[1]) if len(pos) > 1 else 8

    ref = _db().document(_need("LA_FS_KEYS"))
    ring = list((ref.get().to_dict() or {}).get("api_keys") or [])
    existing = next((k for k in ring if k.get("key_id") == key_id), None)

    if existing and not rotate:
        # 只改并发上限是安全的 —— 不影响已经发出去的 secret。
        if existing.get("max_concurrency") != concurrency:
            existing["max_concurrency"] = concurrency
            ref.set({"api_keys": ring}, merge=True)
            print(f"{key_id} 已存在，并发上限改为 {concurrency}")
        else:
            print(f"{key_id} 已存在，没动。要换 secret 加 --rotate")
        return

    sec = secrets.token_urlsafe(32)
    if existing:
        existing.update(secret=sec, max_concurrency=concurrency)
    else:
        ring.append({"key_id": key_id, "secret": sec, "max_concurrency": concurrency})
    ref.set({"api_keys": ring}, merge=True)

    print(f"{'轮换' if existing else '新建'} {key_id}（并发 {concurrency}）")
    print("secret 只显示这一次，记下来配到调用方：")
    print(sec)


# ── render ──────────────────────────────────────────────────────────

_SELF_INPUTS = {"LA_FS_PROJECT", "LA_FS_DATABASE", "LA_FS_LIVEKIT",
                "LA_FS_KEYS", "LA_ENV_OUT", "LA_API_KEYS"}


def cmd_render(_args: list[str]) -> None:
    db = _db()
    lk_path, keys_path, out = (_need("LA_FS_LIVEKIT"), _need("LA_FS_KEYS"),
                               _need("LA_ENV_OUT"))

    lk = db.document(lk_path).get().to_dict() or {}
    for f in ("api_key", "api_secret"):
        if not lk.get(f):
            sys.exit(f"{lk_path} 里没有 {f}")

    ring = (db.document(keys_path).get().to_dict() or {}).get("api_keys")
    if not ring:
        sys.exit(f"{keys_path} 里没有 api_keys（先跑 `{sys.argv[0]} mint <id>`）")

    lines = [
        "# 由 scripts/gateway-secrets.py render 生成，别手改 —— 下次渲染会覆盖。",
        f"LIVEKIT_API_KEY={lk['api_key']}",
        f"LIVEKIT_API_SECRET={lk['api_secret']}",
        # JSON 里没有换行，直接塞；systemd 的 EnvironmentFile 不要加引号，
        # 加了会被当成值的一部分。
        f"LA_API_KEYS={json.dumps(ring, separators=(',', ':'), ensure_ascii=False)}",
    ]
    # 调用方额外指定的 LA_* 调优项（LA_IDLE_TIMEOUT_S 之类）一起带上，
    # 但不要把本脚本自己的输入项写进去。
    lines += [f"{k}={v}" for k, v in os.environ.items()
              if k.startswith("LA_") and k not in _SELF_INPUTS]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"已写 {out}（0600），{len(ring)} 个 API key")   # 不打印任何值


# ── 入口 ────────────────────────────────────────────────────────────

_CMDS = {"mint": cmd_mint, "render": cmd_render}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        sys.exit(__doc__)
    _CMDS[sys.argv[1]](sys.argv[2:])
