#!/usr/bin/env python3
"""从 Firestore 拉凭据，渲染出给 systemd 用的 EnvironmentFile。

**这个仓库是公开的**，所以这里一个项目 ID、数据库名、文档路径都不写死 ——
全部从环境变量来。哪个组织用它，就配自己那套：

    LA_FS_PROJECT    Firestore 项目
    LA_FS_DATABASE   数据库名（默认 "(default)"）
    LA_FS_LIVEKIT    存 LiveKit 凭据的文档路径，要有 api_key / api_secret
    LA_FS_KEYS       存网关 API key 的文档路径，要有 api_keys: [ {key_id, secret, max_concurrency} ]
    LA_ENV_OUT       输出文件路径

用法：

    LA_FS_PROJECT=... LA_FS_LIVEKIT=config/livekit LA_FS_KEYS=config/xxx \\
    LA_ENV_OUT=/etc/liveavatar-gateway/env  python3 scripts/render-env.py

## 两条不能省的

1. **输出文件必须 0600。** 里面是明文 secret。先 `os.open(..., 0o600)` 再写，
   不要先 `open()` 后 `chmod` —— 那中间有一个窗口文件是 0644 的。
2. **secret 绝不打印。** 脚本只报「写了几个 key」，不报内容。
   出错信息里也不许带值 —— 报错信息经常被原样贴进聊天窗口。
"""
from __future__ import annotations

import json
import os
import sys


def _need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"缺环境变量 {name}")
    return v


def main() -> None:
    project = _need("LA_FS_PROJECT")
    database = os.environ.get("LA_FS_DATABASE", "(default)")
    lk_path = _need("LA_FS_LIVEKIT")
    keys_path = _need("LA_FS_KEYS")
    out = _need("LA_ENV_OUT")

    from google.cloud import firestore

    # 显式传 project/database：同名的脏库会静默吞掉读写，而现象是
    # 「读出来是空的」而不是报错。
    db = firestore.Client(project=project, database=database)

    lk = db.document(lk_path).get().to_dict() or {}
    for f in ("api_key", "api_secret"):
        if not lk.get(f):
            sys.exit(f"{lk_path} 里没有 {f}")

    ring = (db.document(keys_path).get().to_dict() or {}).get("api_keys")
    if not ring:
        sys.exit(f"{keys_path} 里没有 api_keys（先跑 scripts/mint-key.py）")

    lines = [
        "# 由 scripts/render-env.py 生成，别手改 —— 下次渲染会覆盖。",
        f"LIVEKIT_API_KEY={lk['api_key']}",
        f"LIVEKIT_API_SECRET={lk['api_secret']}",
        # JSON 里没有换行，直接塞；systemd 的 EnvironmentFile 不需要引号，
        # 加引号反而会被当成值的一部分。
        f"LA_API_KEYS={json.dumps(ring, separators=(',', ':'), ensure_ascii=False)}",
    ]
    for k, v in os.environ.items():
        # 把调用方额外指定的 LA_* 调优项一起带上（LA_IDLE_TIMEOUT_S 之类），
        # 但不要把本脚本自己的输入项写进去。
        if k.startswith("LA_") and k not in {
            "LA_FS_PROJECT", "LA_FS_DATABASE", "LA_FS_LIVEKIT",
            "LA_FS_KEYS", "LA_ENV_OUT", "LA_API_KEYS",
        }:
            lines.append(f"{k}={v}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # ⚠️ 0600 要在创建那一刻就定死。先 open 后 chmod 中间那一瞬是 0644。
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"已写 {out}（0600），{len(ring)} 个 API key")   # 不打印任何值


if __name__ == "__main__":
    main()
