# CloseCrab Avatar

**音频进，视频流出。复杂度全包在里面。**

```
你的 agent 只写这一句 —— 跟接 HeyGen / Tavus 完全一样
    AvatarSession("closecrab/<形象>", base_url=..., api_key=..., api_secret=...)
       ↓  音频
    ┌────────────────────────────────────────┐
    │  控制面：鉴权 · 调度 · 铸票 · 槽位回收     │  CPU 小机器
    │  Worker：模型常驻 · 逐块出帧              │  GPU 机器 × N
    └────────────────────────────────────────┘
       ↓  视频 + 原音频（LiveKit 房间里的一路 participant）
```

调用方**不需要知道**：模型在哪张卡上、卡够不够、Spot 被回收了怎么办、
token 怎么铸、会话怎么回收、音频怎么切块喂给模型。
它只知道一件事：**给我一个形象和一路音频，房间里会出现一个会说话的人。**

## 五分钟跑起来

```bash
git clone https://github.com/yangwhale/closecrab-avatar && cd closecrab-avatar

# 单机（控制面 + GPU worker 同一台）
./scripts/install.sh --role all \
    --livekit-key <你的 LiveKit key> --livekit-secret <secret>

# 分开部署：CPU 机器上
./scripts/install.sh --role control --livekit-key <k> --livekit-secret <s>
# GPU 机器上（指向控制面）
./scripts/install.sh --role worker --gateway http://<控制面>:8080
sudo systemctl enable --now closecrab-avatar-worker@0     # 一卡一个进程

./scripts/install.sh --check        # 随时体检，什么都不动
```

装完 `--role control` 会打印**只显示一次**的 API secret，拿它配到调用方。

## 文档

| 想干什么 | 看哪份 |
|---|---|
| 第一次装，想快点跑通 | [quickstart.md](docs/quickstart.md) |
| 控制面 / worker 怎么分布式部署 | [deploy.md](docs/deploy.md) |
| 我的 agent 怎么接进来 | [api.md](docs/api.md) |
| 鉴权、配额、多租户 | [auth.md](docs/auth.md) |
| GPU 机器装不起来 / 版本对不上 | [gpu-setup.md](docs/gpu-setup.md) |
| 能跑多快、几路并发 | [benchmarks.md](docs/benchmarks.md) |
| 上线之后怎么看、怎么修 | [operations.md](docs/operations.md) |
| 为什么是这个架构 | [architecture.md](docs/architecture.md) |

## 现状

| 阶段 | 内容 | 状态 |
|---|---|---|
| 控制面 | HTTP 契约 · 鉴权 · 槽位调度 · 铸票 · 回收 | ✅ 生产在跑 |
| 部署 | 一键脚本 · systemd · 幂等 · 体检 | ✅ |
| GPU 环境 | 版本坑全固化进脚本，装完自检 | ✅ |
| 流式出帧 | 逐块 `yield`，实测 1.66× 实时 | ✅ 已验证 |
| Worker 端到端 | 音频回调 ⇄ LiveKit 视频轨 | 🟡 收尾中 |
| 转场 / 待机策略 | 出场入场动效、空闲时不烧卡 | ⬜ |

## 测试

```bash
python -m pytest tests/ -q     # 122 条，含 negative test
python smoke.py                # 端到端契约冒烟，不碰 GPU、不连 LiveKit
```

**做过变异测试**：往鉴权/调度/铸票/音频几何/打断逻辑里注入了 30+ 种错误
（跳过签名校验、不强制 `exp`、`terminate_token` 恒真、忽略心跳超时、满了还派活、
并发计数恒 0、`publish_on_behalf` 拼错、房间授权不限定、帧率拿错配置树、
打断后不清在途帧……），**全部被测试抓到**。

## 许可

Apache-2.0（见 [LICENSE](LICENSE)）。模型权重与上游 [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) 的许可另见其仓库。
