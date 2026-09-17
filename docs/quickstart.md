# Quickstart

目标：**从一台干净机器到房间里出现一个会说话的人**。

## 你需要什么

| | 最低 | 说明 |
|---|---|---|
| 控制面 | 2 vCPU / 2 GB | 不在媒体路径上，放哪儿都不影响帧率 |
| Worker | 1× 80 GB GPU | 单卡够用；5 卡 TPP 见 [benchmarks](benchmarks.md) |
| 磁盘 | 60 GB | 模型权重 47 GB |
| LiveKit | key + secret | 网关要自己铸 worker 的房间 token |

## 先决定一件事：并发还是画质

| 你要 | 怎么起 worker | 每路 | 实时倍数 |
|---|---|---|---|
| **多路并发** | 一卡一进程 `@0…@7` + `LA_TRIM_K=4` + 384×256 | 1 GPU | 1.357× |
| **单路最好** | 5 卡 TPP + 720×400 | 5 GPU | 1.610× |

选错了不会报错，只会「怎么只能开一路」或者「怎么画质这么糊」。
数字和取舍全在 [benchmarks.md](benchmarks.md)。下面按**多路并发**写。

## 一、装

```bash
git clone https://github.com/yangwhale/closecrab-avatar && cd closecrab-avatar
./scripts/install.sh --role all --livekit-key <k> --livekit-secret <s>
```

`--role all` = 控制面 + worker 同机。分开装见 [deploy.md](deploy.md)。

**装完会打印一次 API secret，记下来** —— 之后任何地方都不回显。
丢了就 `./scripts/gateway-secrets.py mint <key_id> --rotate` 重发。

权重 47 GB 在后台拉，进度：

```bash
tail -f /tmp/la-worker-bootstrap.log
./scripts/install.sh --check          # 按体积看权重齐了没
```

## 二、起 worker

权重齐了之后，一卡一个进程：

```bash
sudo systemctl enable --now closecrab-avatar-worker@0
# 8 张卡就 @0 … @7
curl -s localhost:8080/healthz        # slots_total 应该等于你起的进程数
```

## 三、接进来

```python
from livekit.agents.voice.avatar import AvatarSession

avatar = AvatarSession(
    "closecrab/bunny-scholar",              # 或 image_url=<自定义形象>
    base_url="http://<控制面>:8080",
    api_key="default",                      # install.sh 给的
    api_secret="<只显示过一次的那个>",
)
await avatar.start(session, room=ctx.room)
```

就这样。agent 侧写法跟接 HeyGen / Tavus 一模一样。

## 验收

```bash
curl -s localhost:8080/healthz | jq
# {"status":"ok","slots_total":8,"slots_used":0,"slots_free":8,"keys":1}
```

房间里会多一路 participant，identity 是你传的 `avatar_identity`，
它带 `lk.publish_on_behalf` 指向你的 agent —— 前端的
`Room.agentParticipants` 因此会把它认成你 agent 的化身，而不是第三个人。

## 出事了看哪儿

```bash
journalctl -u closecrab-avatar -f              # 控制面
journalctl -u closecrab-avatar-worker@0 -f     # worker
./scripts/install.sh --check                   # 一眼看全
```

装不起来的话，**九成是 GPU 环境那三个坑** → [gpu-setup.md](gpu-setup.md#三个坑)。
