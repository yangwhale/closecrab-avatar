# 运维

## 一眼看全

```bash
./scripts/install.sh --check
curl -s localhost:8080/healthz | jq
journalctl -u closecrab-avatar -f
journalctl -u 'closecrab-avatar-worker@*' -f
```

## 槽位不对的时候

`slots_total` 比你起的 worker 进程数少 → 有 worker 心跳丢了。日志里找：

```
WARNING closecrab.avatar: worker <id> 心跳丢失，已摘除
WARNING closecrab.avatar: 回收超时会话 <psid>（room=<room>）
```

**这两条能打出来本身就是个功能。** 早期版本有两个 bug 让它们永远不出现：

1. uvicorn 只配自己那几个 logger，应用的 logger 走 root、而 root 默认
   WARNING 且没 handler → **一条都不打**。
2. `sweep_dead_workers()` 只遍历「身上有活跃会话的 worker」——
   **一个空闲 worker 死掉永远不会被摘、也不会报**，而 `slots_total` 照样
   正确地掉下去。**表面现象全对，缺的只有那条线索。**

两个都修了。如果你 fork 出去改这块，**记得这两条是配套的**。

## 槽位泄漏

长期占满不掉 → 多半是客户端 start 了不 terminate。两道防线：

- `LA_IDLE_TIMEOUT_S`（默认 120）—— 没有音频流动就回收
- `LA_MAX_SESSION_S`（默认 3600）—— 一直有音频也有上限

会话记录**只置 `closed` 不删**，所以查得出「这一路是怎么没的」：

```bash
sqlite3 /var/lib/closecrab-avatar/state.db \
  "select provider_session_id, room_name, state, datetime(created_at,'unixepoch')
   from sessions order by created_at desc limit 20"
```

## Spot 被回收了

worker 那台机器直接没了。控制面这边：心跳超时 → 摘掉 → 它名下的会话置 closed。
新机器起来跑一遍 `install.sh --role worker` 重新注册即可，**控制面不用动**。

⚠️ 新实例**IP 会变**。别信 `~/.ssh/config` 里的旧地址，用
`gcloud compute instances list` 现查。

## 换 key

```bash
./scripts/gateway-secrets.py mint <key_id> --rotate
sudo systemctl restart closecrab-avatar
```

**旧 secret 立刻失效** —— 先通知调用方，别在他们用着的时候轮换。
只改并发上限不用 `--rotate`（不影响已发出去的 secret）。

## 备份

要备份的只有两个东西：

| | 路径 | 丢了会怎样 |
|---|---|---|
| 凭据 | `/etc/closecrab-avatar/env` | 得重新铸 key、重配所有调用方 |
| 会话账 | `/var/lib/closecrab-avatar/state.db` | 只丢历史，服务照跑 |

模型权重不用备份 —— 47 GB，从 HF 重拉比恢复快。

## 日志级别的规矩

**日志级别是给下游机器看的判据，不是给人看的语气。**

- **可自愈、有固定节奏的事件不许进 ERROR。** 打 INFO 并在正文里写「例行」。
  例行事件长期占着 ERROR，等价于把告警通道烧掉 —— 而**自动化的下游会当真**。
- **「没有 X」要分清「从来就不该有 X」和「本来有、现在没了」。**
  前者常态打 INFO，后者才报警。混成一条就是误报工厂。
