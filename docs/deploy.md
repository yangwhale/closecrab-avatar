# 部署

## 放哪儿

| 组件 | 机器 | 理由 |
|---|---|---|
| 控制面 | **常驻 CPU 小机器**（2 vCPU 够） | ① GPU 机器（尤其 Spot）会消失，控制面不能跟着消失 ② 它**不在媒体路径上**，只在建/停会话时调一次，放哪儿都不影响帧率 |
| Worker | **GPU 机器，一卡一进程** | 模型常驻显存，每路会话不付启动成本 |
| 状态 | 控制面本地 SQLite 单文件 | 单实例够用、零外部依赖。要多实例再换 Postgres（只需替换 `store.py`） |

**同机也完全可以**（`--role all`）—— 分开只是为了让控制面活得比 GPU 久。

## 网络：只需要一个方向

```
Worker ──出向 HTTP──► 控制面:8080
```

**GPU 机器不用开任何入向端口**，可以待在 NAT / 不同 VPC / 不同云后面。
控制面从不主动连 worker。Spot 被回收后换台机器重新注册即可。

⚠️ `/internal/*` 没有鉴权，靠网络隔离。**把控制面端口限制在你自己的网段。**
公网默认拒绝就够了 —— 但要**实测**，别看规则：

```bash
# 从一个真正的外部服务打，不要用本机打自己的外网 IP（那是回环，不算数）
curl -s "https://api.allorigins.win/raw?url=http%3A%2F%2F<外网IP>%3A8080%2Fhealthz"
```

## 分开装

```bash
# CPU 机器
./scripts/install.sh --role control --livekit-key <k> --livekit-secret <s>

# GPU 机器（指向控制面内网地址）
./scripts/install.sh --role worker --gateway http://10.x.x.x:8080
sudo systemctl enable --now closecrab-avatar-worker@0
```

装完两边都跑一次 `./scripts/install.sh --check`。

## 必须配的两个超时

| 参数 | 默认 | 为什么不能不配 |
|---|---|---|
| `LA_IDLE_TIMEOUT_S` | 120 | 客户端跑掉而不调 terminate 时回收槽位。**漏一路就少一张卡。** LiveKit 源码里自己警告过两次会话会「一直计费到 idle 超时」 |
| `LA_MAX_SESSION_S` | 3600 | 一直有音频也不能永远占着卡 |

`LA_WORKER_HB_TIMEOUT_S`（默认 30）超时后，控制面会关掉该 worker 名下的会话并摘掉它。
**会话记录只置 `closed` 不删** —— 留痕才查得出「这一路是怎么没的」。

## 升级

```bash
git pull
sudo systemctl restart closecrab-avatar                  # 控制面
sudo systemctl restart 'closecrab-avatar-worker@*'       # worker
```

控制面重启不丢状态（SQLite 落盘）。worker 重启会被心跳摘掉再重新注册，
在跑的会话会断 —— 要无损升级就先把 worker 的 capacity 调 0 等它排空。

## Spot 上的注意事项

worker 环境**必须靠脚本装，不能手敲** —— 实例说没就没，回来是全新机器，
家目录连同 venv 和权重一起清空。`worker-bootstrap.sh` 幂等，`@reboot` 里直接调。

SSH 别名里的 IP 也会失效（MIG 重建会换 IP）。用 `gcloud compute instances list`
现查，别信配置文件里的。
