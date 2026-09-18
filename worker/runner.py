"""GPU worker —— 注册、长轮询取活、进房间干活。

**pull 模型，不是 push。** worker 主动连控制面，所以 GPU 机器
不需要对外开端口，可以在 NAT / 不同 VPC 后面，Spot 被回收后
换台机器重新注册就行。控制面只记账，从不主动连 worker。

## 五卡流式下这个进程长什么样

`torchrun --nproc_per_node=5` 起 5 个**本文件**的进程，它们分两种角色：

    rank 0–3   DiT。`load()` 之后就在主线程里把 `generate()` 跑到底，
               **完全不碰控制面、不碰 LiveKit**。它们只是算力。
    rank 4     VAE。模型跑在后台线程，主线程跑下面这个 asyncio Worker。

所以「一个 worker」在控制面眼里是**一个** worker（rank 4 注册的那个），
底下吃 5 张卡。不要让 5 个 rank 都去注册 —— 那会凭空多出 4 个幽灵 worker，
控制面按它们的容量派活，派过去没人接。

单卡 / 无模型时 `WORLD_SIZE=1`，走静帧兜底，形态跟以前一样。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, DataStreamAudioReceiver

from .imageutil import upright
from .publish_opts import TunedAvatarRunner
from .generators import StaticImageGenerator

# ⚠️ **顶层不 import torch，也不 import 任何拖 torch 的东西。**
#    这个文件在控制面机器上也会被 import（跑测试、查签名），而那台没有 torch。
#    `live_generator` 只要 livekit + numpy，`pipeline_source` 顶层只要 numpy。
from .live_generator import LiveAvatarGenerator
from .pipeline_source import LiveAvatarPipelineSource

log = logging.getLogger("liveavatar.worker")

HEARTBEAT_S = 10.0
RECONNECT_BACKOFF_S = 3.0

# 换脸：把新图路径写进这个文件，**上游补丁**会在下一轮开头读它并就地换掉
# 参考图那几个张量（见 scripts/patch-upstream-faceswap.py 和 docs/face-swap.md）。
# worker 这边只负责「把图放好、把路径写上」，换的动作在模型循环里。
_FACE_FILE = os.environ.get("CCA_FACE_FILE", "/tmp/cca-current-face.txt")


def _load_image(path: str | None, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if path and os.path.exists(path):
        from PIL import Image
        im = Image.open(path).convert("RGBA").resize((w, h))
        return np.asarray(im, dtype=np.uint8)
    # 没给图就发一张纯色板 —— 至少证明视频轨通了
    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[..., 0], a[..., 1], a[..., 2], a[..., 3] = 30, 40, 60, 255
    return a

class Worker:
    """控制面那一侧。**对模型只知道一件事：有没有 `source`。**

    `source` 是已经 `load()` 过、正在跑的 `FrameSource`；`None` 表示这台
    没有可用模型，退回静帧。判定在 `main()` 里做完，这里不再猜。
    """

    def __init__(self, gateway_url: str, worker_id: str, *, capacity: int = 1,
                 image_path: str | None = None, source=None):
        self._gw = gateway_url.rstrip("/")
        self._id = worker_id
        self._capacity = capacity
        self._image_path = image_path
        self._source = source
        self._active: set[str] = set()
        self._face_version: str | None = None

    async def run(self) -> None:
        async with aiohttp.ClientSession() as http:
            while True:
                try:
                    await self._register(http)
                    hb = asyncio.create_task(self._heartbeat_loop(http))
                    try:
                        await self._poll_loop(http)
                    finally:
                        hb.cancel()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.warning("与控制面断开（%s），%.0fs 后重连", e, RECONNECT_BACKOFF_S)
                    await asyncio.sleep(RECONNECT_BACKOFF_S)

    async def _register(self, http: aiohttp.ClientSession) -> None:
        meta = {"host": socket.gethostname(), "pid": os.getpid(),
                "model": bool(self._source)}
        async with http.post(f"{self._gw}/internal/workers/register",
                             json={"worker_id": self._id, "capacity": self._capacity,
                                   "meta": meta}) as r:
            r.raise_for_status()
        log.info("已注册到控制面：%s（容量 %d，%s）", self._gw, self._capacity,
                 "真模型" if self._source else "静帧兜底")

    async def _heartbeat_loop(self, http: aiohttp.ClientSession) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            try:
                async with http.post(
                    f"{self._gw}/internal/workers/{self._id}/heartbeat",
                    json={"active_sessions": sorted(self._active)},
                ) as r:
                    if r.status == 410:
                        # 心跳断过被摘了 —— 必须重新注册，不能装作还在册
                        log.warning("控制面说我没注册，重新注册")
                        await self._register(http)
            except Exception as e:
                log.warning("心跳失败: %s", e)

    async def _poll_loop(self, http: aiohttp.ClientSession) -> None:
        while True:
            async with http.get(f"{self._gw}/internal/workers/{self._id}/jobs",
                                timeout=aiohttp.ClientTimeout(total=60)) as r:
                r.raise_for_status()
                job = (await r.json()).get("job")
            if job:
                asyncio.create_task(self._run_session(http, job))

    async def _apply_persona(self, http: aiohttp.ClientSession, job: dict) -> None:
        """把这一场该用的脸取下来、写进换脸文件。**只做这两件事。**

        真正的切换由上游补丁在生成循环里完成（下一轮开头就地重算参考图张量），
        这边不碰模型、不发信号、不重启任何东西。

        没设过形象图是正常状态，不是故障：一声不吭地继续用当前那张。
        """
        ver, url = job.get("persona_version"), job.get("persona_url")
        if not ver or not url or ver == self._face_version:
            return
        try:
            async with http.get(f"{self._gw}{url}",
                                timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    log.warning("取形象图失败 %s —— 继续用当前这张", r.status)
                    return
                data = await r.read()
                ext = {"image/png": ".png", "image/webp": ".webp"}.get(
                    r.headers.get("Content-Type", ""), ".jpg")
        except Exception as e:                        # noqa: BLE001
            # 取不到图**不能让会话起不来** —— 旧脸总比没脸好。
            log.warning("取形象图出错（%s）—— 继续用当前这张", e)
            return
        img = f"/tmp/cca-face-{job.get('persona_role', 'principal')}-{ver}{ext}"
        try:
            data = upright(data) or data
            with open(img, "wb") as f:
                f.write(data)
            # 原子写：模型循环随时可能在读，写一半被读到会拿到半个路径。
            tmp = _FACE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(img)
            os.replace(tmp, _FACE_FILE)
        except OSError as e:
            log.warning("形象图落盘失败（%s）—— 继续用当前这张", e)
            return
        self._face_version = ver
        log.info("换脸：角色=%s 版本=%s → %s（下一轮生效）",
                 job.get("persona_role"), ver, img)

    async def _run_session(self, http: aiohttp.ClientSession, job: dict) -> None:
        psid = job["provider_session_id"]
        self._active.add(psid)
        room = rtc.Room()
        runner = None
        try:
            await room.connect(job["livekit_url"], job["room_token"])
            log.info("会话 %s 已进房 %s", psid, job["room_name"])

            if self._source is not None:
                # ⚠️ 尺寸**取模型的真实出帧尺寸**，不取 job 里请求的那个。
                #    模型按 64 的网格取整（请求 720×400 实出 704×384），
                #    `AvatarOptions` 配错了帧尺寸对不上。
                w, h = self._source.size
                gen = LiveAvatarGenerator(self._source)
                fps = self._source.geometry.fps
                await self._apply_persona(http, job)
                # 上一场可能留下半句话没念完的 PCM —— 新会话从干净状态开始。
                self._source.reset()
                log.info("会话 %s 用真模型（%d×%d @ %d fps）", psid, w, h, fps)
            else:
                w, h = (int(x) for x in job.get("size", "384*256").split("*"))
                fps = 25
                img = _load_image(job.get("image_path") or self._image_path, (w, h))
                gen = StaticImageGenerator(img, fps=fps)
                log.warning("会话 %s **退回静帧** —— 这台没有可用模型，"
                            "起动时的 warning 里写了是哪一条不满足", psid)

            # ⚠️ 用 TunedAvatarRunner 不用 AvatarRunner —— 库里那个发布视频轨时
            #    **一个编码参数都不传**，拿到的默认值对着手机是坏的。理由见
            #    `publish_opts.py`。
            runner = TunedAvatarRunner(
                room,
                audio_recv=DataStreamAudioReceiver(room, sender_identity=job["agent_identity"]),
                video_gen=gen,
                options=AvatarOptions(
                    video_width=w, video_height=h, video_fps=fps,
                    audio_sample_rate=job.get("sample_rate", 16000), audio_channels=1,
                ),
            )
            # ⚠️ **`runner.start()` 会无声地卡死，必须给它加一个超时和分步日志。**
            #
            #   它第一步是 `DataStreamAudioReceiver.start()` →
            #   `wait_for_participant(identity=agent_identity)`，而那个函数只认
            #   **ACTIVE** 状态。一个进了房但一直不 active 的 agent 会让它
            #   永远等下去 —— 音视频轨一个都发不出来，控制面这边看到的是
            #   「会话建了、worker 也进房了、就是没画面」，跟模型崩了长得一样。
            #   卡住的那一路还占着槽位，直到 120 s 空闲回收才还。
            #
            #   所以宁可**快失败**：超时就抛出去，`finally` 那段会还槽位并
            #   写清楚卡在哪一步。诊断信息比多等两分钟值钱。
            start_grace = float(job.get("runner_start_grace_s", 30))
            log.info("会话 %s：挂音频接收 + 发布音视频轨（上限 %.0f s）…",
                     psid, start_grace)
            try:
                await asyncio.wait_for(runner.start(), timeout=start_grace)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"runner.start() 超过 {start_grace:.0f}s 没完成。"
                    f"最常见的原因是 agent {job['agent_identity']!r} 进了房但一直不 "
                    f"active —— 它得真发一条轨（生产上 bot 本来就发音轨），"
                    f"光进房不发东西是不够的。") from None
            log.info("会话 %s：音视频轨已发布，开始收音频", psid)

            # ⚠️ **不能只听 `disconnected`。** 那个事件只在**自己**被断开时触发，
            #    agent 走了我们照样连着 —— 于是一个人待在空房间里，心跳还
            #    照常把会话的 `updated_at` 往前推，控制面那 120 秒空闲回收
            #    **永远等不到**。槽位就被一个没人的房间占死了。
            #    （实测过：会话建了 94 秒，updated_at 永远停在 17 秒前。）
            over = asyncio.Event()
            agent_id = job["agent_identity"]
            room.on("disconnected", lambda *_: over.set())

            @room.on("participant_disconnected")
            def _(p: rtc.RemoteParticipant) -> None:
                if p.identity == agent_id:
                    log.info("会话 %s：agent %s 走了，跟着收摊", psid, agent_id)
                    over.set()

            # agent 可能在我们挂上监听之前就进了又走了 —— 那样事件永远不来。
            # **事件之外再补一次当下的状态**，别只信事件。
            grace = job.get("agent_join_grace_s", 30)
            deadline = asyncio.get_running_loop().time() + grace
            while (not over.is_set() and agent_id not in room.remote_participants
                   and asyncio.get_running_loop().time() < deadline):
                await asyncio.sleep(0.5)
            if not over.is_set() and agent_id not in room.remote_participants:
                log.warning("会话 %s：%ss 内等不到 agent %s 进房，收摊",
                            psid, grace, agent_id)
                over.set()

            await over.wait()
        except Exception:
            log.exception("会话 %s 异常结束", psid)
        finally:
            self._active.discard(psid)
            # ⭐⭐ **必须关掉 runner。** 2026-09-18 之前这里漏了，后果不是
            #    「多占一点内存」，是**画面严重卡顿，而且一场比一场重**：
            #
            #    `AvatarRunner` 自己起了三条常驻任务（读音频 / 转发视频 /
            #    AVSynchronizer 的 `_capture_video`）。不关的话它们**永远活着**，
            #    而每一场的 generator 包的都是**同一个** pipeline source ——
            #    于是 N 个漏下来的消费者跟当前这一场抢 `next_frame()`，
            #    真正在播的那一路只拿到 1/N 的帧。
            #
            #    日志里的指纹很清楚：同一毫秒出现 **2～3 条一模一样**的
            #    `Frame capture was behind schedule for 4188.51 ms` ——
            #    一个同步器不可能重复报同一个值，那是几个漏下来的实例
            #    各报各的。重数 ＝ 当时活着的 runner 数。
            #
            #    房间 disconnect **救不了**：`_capture_video` 拉的是自己的队列，
            #    跟房间没关系。
            if runner is not None:
                try:
                    await runner.aclose()
                except Exception:                    # noqa: BLE001
                    log.warning("会话 %s 关 runner 失败", psid, exc_info=True)
            if self._source is not None:
                # 人走了，别让没念完的音频喂给下一场
                self._source.reset()
            try:
                await room.disconnect()
            except Exception:
                pass
            try:
                async with http.post(f"{self._gw}/internal/sessions/{psid}/closed"):
                    pass
            except Exception:
                pass          # 报不上去也没关系，控制面的 reaper 会兜住
            log.info("会话 %s 收尾完成", psid)


# ── 启动 ──────────────────────────────────────────────────────────


def _init_distributed() -> tuple[int, int]:
    """torchrun 只设环境变量，进程组得自己建。

    ⚠️ 这一步漏了的症状极难查：注册、心跳、建会话、出帧一路全绿，
    只是 `dist.is_initialized()` 为 False，于是判定「没模型」安静退回静帧。
    **兜底越体面，漏越难发现。**
    """
    import datetime

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # ⭐ **超时必须放到很长，默认 10 分钟会把我们打死。**
        #
        # 这条设计是「没人说话时整组停在 dist.recv 上省电」（见
        # pipeline_source 文件头）。但 NCCL 的 watchdog 不知道那是故意的 ——
        # 它只看到一个 RECV 挂了 600 秒，判定集合通信挂死，**SIGABRT 掉整组**：
        #
        #   Watchdog caught collective operation timeout:
        #   WorkNCCL(OpType=RECV) ran for 600087 ms before timing out
        #
        # 实测就是这么死的：最后一次说话之后整十分钟，五个进程一起没。
        # 现象是「worker 好好的突然就不在册了」，而日志要翻到最底下
        # 那一大段 C++ 栈才看得到原因。
        #
        # 代价说清楚：真挂死的时候也要等这么久才报。这条路径上「挂死」
        # 本来就靠控制面的心跳超时发现（30 秒），不靠 NCCL。
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=datetime.timedelta(hours=12))
    return dist.get_rank(), dist.get_world_size()


def _why_no_model(a) -> str | None:
    """不能用真模型的话，说清楚是**哪一条**不满足。

    三个条件缺一不可，而缺任何一个的症状都是「画面是张不动的图」——
    所以逐条查、逐条说，别笼统报一句「模型不可用」。
    """
    if int(os.environ.get("WORLD_SIZE", "1")) < 5:
        return ("不在 torchrun 下或卡数不足（WORLD_SIZE=%s，五卡流式要 5）。"
                "起法见 docs/deploy.md「五卡流式怎么起」"
                % os.environ.get("WORLD_SIZE", "未设"))
    for label, path in (("权重目录", a.ckpt_dir), ("训练配置", a.training_config),
                        ("参考图", a.image), ("预热音频", a.warmup_audio)):
        if not path:
            return f"没指定{label}"
        if not os.path.exists(path):
            return f"{label}不存在：{path}"
    try:
        import torch  # noqa: F401
    except Exception as e:
        return f"torch import 不了：{e}"
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="LiveAvatar GPU worker")
    ap.add_argument("--gateway", default=os.environ.get("LA_GATEWAY_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--worker-id", default=os.environ.get(
        "LA_WORKER_ID", f"{socket.gethostname()}-{os.environ.get('CUDA_VISIBLE_DEVICES','0')}"))
    ap.add_argument("--capacity", type=int, default=int(os.environ.get("LA_WORKER_CAPACITY", "1")))
    ap.add_argument("--image", default=os.environ.get(
        "LA_AVATAR_IMAGE", os.path.expanduser("~/LiveAvatar/examples/dwarven_blacksmith.jpg")))
    ap.add_argument("--prompt", default=os.environ.get("LA_AVATAR_PROMPT", ""))
    ap.add_argument("--ckpt-dir", default=os.environ.get(
        "LA_CKPT_DIR", os.path.expanduser("~/LiveAvatar/ckpt/Wan2.2-S2V-14B")))
    ap.add_argument("--training-config", default=os.environ.get(
        "LA_TRAINING_CONFIG",
        os.path.expanduser("~/LiveAvatar/liveavatar/configs/s2v_causal_sft.yaml")))
    # 第 0、1 轮预热用的音频，内容无所谓 —— 只用来建编码模板。见 pipeline_source。
    ap.add_argument("--warmup-audio", default=os.environ.get(
        "LA_WARMUP_AUDIO",
        os.path.expanduser("~/LiveAvatar/examples/dwarven_blacksmith.wav")))
    # LoRA 本地权重。**给绝对路径**，否则上游按 CWD 重下一份 1.35 GB。
    ap.add_argument("--lora-path", default=os.environ.get(
        "LA_LORA_PATH",
        os.path.expanduser("~/LiveAvatar/ckpt/LiveAvatar/liveavatar.safetensors")))
    ap.add_argument("--size", default=os.environ.get("LA_SIZE", "720*400"))
    ap.add_argument("--infer-frames", type=int, default=int(os.environ.get("LA_INFER_FRAMES", "48")))
    ap.add_argument("--num-gpus-dit", type=int, default=4)
    a = ap.parse_args()

    reason = _why_no_model(a)
    source = None
    rank = 0
    if reason:
        log.warning("**退回静帧**：%s", reason)
    else:
        rank, world = _init_distributed()
        source = LiveAvatarPipelineSource(
            ref_image_path=a.image, prompt=a.prompt,
            ckpt_dir=a.ckpt_dir, training_config=a.training_config,
            warmup_audio=a.warmup_audio, lora_path=a.lora_path, size=a.size,
            infer_frames=a.infer_frames, num_gpus_dit=a.num_gpus_dit,
        )
        # 开机就装，不拖到第一句话 —— 否则第一个用户等三分钟。
        source.load()

        if rank < a.num_gpus_dit:
            # DiT rank：主线程直接把 generate() 跑到底，不注册、不进房间。
            log.info("rank %d 是 DiT，只出算力", rank)
            try:
                source.run_blocking()
            except KeyboardInterrupt:
                return 130
            return 0

        source.start()                 # VAE rank：模型进后台线程

    try:
        asyncio.run(Worker(a.gateway, a.worker_id, capacity=a.capacity,
                           image_path=a.image, source=source).run())
    except KeyboardInterrupt:
        return 130
    finally:
        if source is not None:
            source.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
