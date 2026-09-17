# 鉴权

四层，各防各的。少一层就多一个洞。

| 层 | 机制 | 防什么 |
|---|---|---|
| 调用方身份 | `api_key`/`api_secret` 签短期 JWT（HS256，**强制 `exp`**） | 冒名调用 |
| 会话归属 | `terminate_token = HMAC(secret, provider_session_id)`，常数时间比较 | 别人终止你的会话 |
| 重试安全 | `Idempotency-Key`，**按 key 隔离** | 重试多占一张卡 / 跨租户读到别人的结果 |
| 房间授权 | 铸出的 token 的 roomJoin **限定到那一个房间** | 一把 token 串所有房间 |
| 配额 | 每 key 最大并发 | 一个人占满全部卡 |

## 发 key

```bash
# 幂等：已存在就不动。要换 secret 得显式 --rotate
LA_FS_PROJECT=... LA_FS_KEYS=config/xxx \
  ./scripts/gateway-secrets.py mint <key_id> [并发上限]
```

**不幂等会出事**：重跑一次就换了 secret，而已经配好的调用方不会知道，
它们下一次请求开始 401 —— 看起来像网关坏了。

secret **只在铸出来那一次可见**。之后任何地方都不回显，报错信息里也不带值
（报错经常被原样贴进聊天窗口）。丢了就 `--rotate` 重发。

## 凭据放哪儿

`/etc/closecrab-avatar/env`，**0600，root 所有**。systemd 以 root 读完再降权，
所以跑服务的那个用户自己读不到这个文件，只有进程环境里有。

`LA_API_KEYS` 是 **JSON 不是分隔符拼接**：

```
LA_API_KEYS=[{"key_id":"ios","secret":"...","max_concurrency":4}]
```

一开始想写成 `id:secret:并发,id:secret:并发`。**不行** —— secret 是随机串，
里面完全可能有 `:` 或 `,`，切出来的东西看起来还挺像回事（少一段就落到默认值），
**而且不会报错**。配额是安全边界，不能这么来。

## 一个 key 都没有会怎样

**起不来。** 一个没有 key 的网关能起来、能过 `/healthz`、但每个请求都 401 ——
看起来像「部署好了但客户端配错了」，实际是服务端根本没配。所以宁可拒绝启动。
