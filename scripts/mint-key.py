#!/usr/bin/env python3
"""给调用方铸一把网关 API key，存进 Firestore。**幂等：已存在就不动。**

    LA_FS_PROJECT=... LA_FS_KEYS=config/xxx \\
    python3 scripts/mint-key.py <key_id> [max_concurrency]

## 为什么幂等很重要

不幂等的话，重跑一次就换了 secret，而**已经配好的调用方不会知道** ——
它们下一次请求开始 401，看起来像网关坏了。要轮换得显式加 `--rotate`。

## secret 只在铸出来那一次可见

之后任何地方都不回显（`render-env.py` 也只报数量）。丢了就 `--rotate`
重发一把，别去库里捞——那等于多一个它出现在日志里的机会。
"""
from __future__ import annotations

import os
import secrets
import sys


def _need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"缺环境变量 {name}")
    return v


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--rotate"]
    rotate = "--rotate" in sys.argv
    if not args:
        sys.exit(__doc__)
    key_id = args[0]
    concurrency = int(args[1]) if len(args) > 1 else 8

    from google.cloud import firestore

    db = firestore.Client(project=_need("LA_FS_PROJECT"),
                          database=os.environ.get("LA_FS_DATABASE", "(default)"))
    ref = db.document(_need("LA_FS_KEYS"))
    doc = ref.get().to_dict() or {}
    ring = list(doc.get("api_keys") or [])

    existing = next((k for k in ring if k.get("key_id") == key_id), None)
    if existing and not rotate:
        # 只改并发上限是安全的（不影响已发出去的 secret）
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


if __name__ == "__main__":
    main()
