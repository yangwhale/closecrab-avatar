"""可以直接 `uvicorn --factory` 起的入口。

`docs/deployment.md` 里那段「控制面写个几行的入口」，多数人写出来的都长一样：
读一堆环境变量 → 拼 KeyRing → `create_app`。与其让每个人抄一遍，不如在这里
给一个通用的，**同时把两个容易写错的地方堵死**。

## 密钥怎么进来：一个 JSON，不是分隔符拼接

    LA_API_KEYS='[{"key_id":"ios","secret":"...","max_concurrency":4}]'

一开始想写成 `id:secret:并发,id:secret:并发`。**不行** —— secret 是随机串，
里面完全可能有 `:` 或 `,`，分隔符切出来的东西看起来还挺像回事
（少一段就当默认值），于是并发上限变成默认值、或者 key_id 少了尾巴。
**那种错不会抛异常，只会让鉴权悄悄按错的配置跑。** JSON 没有这个问题。

## 缺了就炸，不给默认值

一个 key 都没有的网关**能起来、能响应 /healthz、但任何请求都是 401** ——
看起来像「部署好了但客户端配错了」，实际是服务端根本没配 key。
所以这里宁可起不来。理由同 `Settings.from_env` 对 LiveKit 凭据的处理。
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI

from .app import create_app
from .auth import ApiKey, KeyRing
from .config import Settings

ENV_KEYS = "LA_API_KEYS"


def load_keyring(raw: str | None = None) -> KeyRing:
    """从 JSON 解析出 KeyRing。**任何形状不对都抛**，不做静默兜底。"""
    if raw is None:
        raw = os.environ.get(ENV_KEYS, "")
    raw = raw.strip()
    if not raw:
        raise RuntimeError(
            f"{ENV_KEYS} 没设 —— 没有 key 的网关能起来、能过 /healthz，"
            "但每个请求都 401，看起来像客户端配错了。所以这里直接不起。"
        )
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{ENV_KEYS} 不是合法 JSON: {e}") from e
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"{ENV_KEYS} 必须是非空数组")

    keys: dict[str, ApiKey] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise RuntimeError(f"{ENV_KEYS}[{i}] 不是对象")
        kid, sec = it.get("key_id"), it.get("secret")
        if not kid or not sec:
            raise RuntimeError(f"{ENV_KEYS}[{i}] 缺 key_id 或 secret")
        if kid in keys:
            # ⚠️ 重复的 key_id 会**静默覆盖**前一个，包括它的并发上限。
            #    配额是安全边界，不能靠「后面那条正好也对」。
            raise RuntimeError(f"{ENV_KEYS} 里 key_id 重复: {kid}")
        n = it.get("max_concurrency", ApiKey.max_concurrency)
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            # bool 是 int 的子类，`True` 会被当成 1 —— 单独挡掉。
            raise RuntimeError(f"{ENV_KEYS}[{i}].max_concurrency 必须是正整数")
        keys[kid] = ApiKey(key_id=str(kid), secret=str(sec), max_concurrency=n)
    return KeyRing(keys)


def build() -> FastAPI:
    """`uvicorn --factory liveavatar_gateway.entry:build`"""
    return create_app(Settings.from_env(), load_keyring())
