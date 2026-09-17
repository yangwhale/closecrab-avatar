"""GPU worker —— 注册、长轮询取活、进房间干活。

**pull 模型，不是 push。** worker 主动连控制面，所以 GPU 机器
不需要对外开端口，可以在 NAT / 不同 VPC 后面，Spot 被回收后
换台机器重新注册就行。控制面只记账，从不主动连 worker。

一张卡一个 worker 进程，模型常驻 —— 这样每路会话不用付启动成本。
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

from .generators import StaticImageGenerator

# ⚠️ **不在顶层 import pipeline_source 之外的任何重依赖。**
#    这个文件在控制面机器上也可能被 import（跑测试、查签名），
#    而那台没有 torch。`pipeline_source` 顶层也只有 numpy。
from .live_generator import LiveAvatarGenerator
from .pipeline_source import LiveAvatarPipelineSource

log = logging.getLogger("liveavatar.worker")

HEARTBEAT_S = 10.0
RECONNECT_BACKOFF_S = 3.0


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
    def __init__(self, gateway_url: str, worker_id: str, *, capacity: int = 1,
                 image_path: str | None = None,
                 ckpt_dir: str = "", training_config: str = ""):
        self._gw = gateway_url.rstrip("/")
        self._id = worker_id
        self._capacity = capacity
        self._image_path = image_path
        self._ckpt_dir = ckpt_dir
        self._training_config = training_config
        self._active: set[str] = set()
        self._model_ready = self._detect_model()

    def _detect_model(self) -> bool:
        """有没有真模型可用。**判据是「能不能跑」，不是「装没装」。**

        三个条件缺一不可，而且缺任何一个的症状都是「画面是张不动的图」——
        所以这里逐条查、逐条说，别笼统报一句「模型不可用」。
        """
        import os.path

        if not self._ckpt_dir or not os.path.isdir(self._ckpt_dir):
            log.warning("没有权重目录（%s）→ 退回静帧", self._ckpt_dir or "未指定")
            return False
        try:
            import torch.distributed as dist
        except Exception:
            log.warning("没装 torch → 退回静帧")
            return False
        if not dist.is_available() or not dist.is_initialized():
            # 五卡流式必须在 torchrun 下跑。**这是最常见的一种「没生效」**：
            # 直接 `python -m worker.runner` 起来一切正常，就是画面不动。
            log.warning("不在 torchrun 下（dist 未初始化）→ 退回静帧。"
                        "五卡流式见 docs/deploy.md「五卡流式怎么起」")
            return False
        log.info("真模型可用：rank %d/%d，权重 %s",
                 dist.get_rank(), dist.get_world_size(), self._ckpt_dir)
        return True

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
        meta = {"host": socket.gethostname(), "pid": os.getpid()}
        async with http.post(f"{self._gw}/internal/workers/register",
                             json={"worker_id": self._id, "capacity": self._capacity,
                                   "meta": meta}) as r:
            r.raise_for_status()
        log.info("已注册到控制面：%s（容量 %d）", self._gw, self._capacity)

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

    async def _run_session(self, http: aiohttp.ClientSession, job: dict) -> None:
        psid = job["provider_session_id"]
        self._active.add(psid)
        room = rtc.Room()
        try:
            await room.connect(job["livekit_url"], job["room_token"])
            log.info("会话 %s 已进房 %s", psid, job["room_name"])

            w, h = (int(x) for x in job.get("size", "384*256").split("*"))

            # 有模型就用真的，没有就退回静帧 —— **退回要说出来**，
            # 否则「怎么是张不动的图」得查半天。
            if self._model_ready:
                src = LiveAvatarPipelineSource(
                    ref_image_path=job.get("image_path") or self._image_path,
                    prompt=job.get("prompt", ""),
                    ckpt_dir=self._ckpt_dir,
                    training_config=self._training_config,
                    size=f"{w}*{h}",
                    infer_frames=int(job.get("infer_frames", 48)),
                )
                gen = LiveAvatarGenerator(src)
                log.info("会话 %s 用真模型", psid)
            else:
                img = _load_image(job.get("image_path") or self._image_path, (w, h))
                gen = StaticImageGenerator(img, fps=25)
                log.warning("会话 %s **退回静帧**（没检测到模型，"
                            "看 --ckpt-dir 和 torchrun 起没起）", psid)

            runner = AvatarRunner(
                room,
                audio_recv=DataStreamAudioReceiver(room, sender_identity=job["agent_identity"]),
                video_gen=gen,
                options=AvatarOptions(
                    video_width=w, video_height=h, video_fps=25,
                    audio_sample_rate=job.get("sample_rate", 16000), audio_channels=1,
                ),
            )
            await runner.start()
            # 房间断开即结束；控制面那边还有 idle 超时兜底
            disconnected = asyncio.Event()
            room.on("disconnected", lambda *_: disconnected.set())
            await disconnected.wait()
        except Exception:
            log.exception("会话 %s 异常结束", psid)
        finally:
            self._active.discard(psid)
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


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="LiveAvatar GPU worker")
    ap.add_argument("--gateway", default=os.environ.get("LA_GATEWAY_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--worker-id", default=os.environ.get(
        "LA_WORKER_ID", f"{socket.gethostname()}-{os.environ.get('CUDA_VISIBLE_DEVICES','0')}"))
    ap.add_argument("--capacity", type=int, default=int(os.environ.get("LA_WORKER_CAPACITY", "1")))
    ap.add_argument("--image", default=os.environ.get("LA_AVATAR_IMAGE"))
    ap.add_argument("--ckpt-dir", default=os.environ.get(
        "LA_CKPT_DIR", os.path.expanduser("~/LiveAvatar/ckpt/Wan2.2-S2V-14B")))
    ap.add_argument("--training-config", default=os.environ.get(
        "LA_TRAINING_CONFIG",
        os.path.expanduser("~/LiveAvatar/liveavatar/configs/s2v_causal_sft.yaml")))
    a = ap.parse_args()
    try:
        asyncio.run(Worker(a.gateway, a.worker_id, capacity=a.capacity, image_path=a.image).run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
