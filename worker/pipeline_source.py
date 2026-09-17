"""`FrameSource` 的真实现 —— 驱动上游流式 pipeline。

## ⚠️ 这个文件决定了 worker 的进程形态，先读这一段

五卡流式路径**不是「一进程一张卡」**。它是 `torchrun --nproc_per_node=5`
拉起的一组 5 个进程：

    rank 0–3   DiT，纯算，不碰 LiveKit
    rank 4     VAE，**只有它**拉音频、出帧 —— LiveKit 那一头全在这个 rank 上

所以：

- systemd 的 worker unit 是 **一组 5 张卡一个 unit**，不是一卡一个
- `worker/runner.py` 在 rank 4 上跑完整逻辑，在 rank 0–3 上只驱动 pipeline
- 一台 8 卡机器跑五卡流式 = **1 组**（剩 3 张闲着）

要「一卡一路、8 路并发」得走单卡 pipeline，而**单卡那条没有流式出帧**
（`causal_s2v_pipeline` 里一个 `yield` 都没有），只能整段生成完再返回。
两条路的取舍见 `docs/benchmarks.md` 第一节 —— **这不是疏忽，是量出来的**。

## 上游缺的那一个钩子

流式 pipeline 第 0、1 轮用预置音频预热，**从第 2 轮起每个 block 调一次
`self.get_audio_callback()` 要实时 PCM**。上游没有任何入口装这个钩子，
这是「流式音频」缺的唯一一环 —— 装上就通。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Callable

import numpy as np

from .audio_stream import BlockGeometry

log = logging.getLogger("closecrab.avatar.pipeline")


class LiveAvatarPipelineSource:
    """把上游 blockwise pipeline 包成 `FrameSource`。

    模型是**阻塞的同步生成器**（torch，不是 asyncio），所以它跑在一个
    单独的线程里，产出的帧用 `call_soon_threadsafe` 送回事件循环。
    """

    def __init__(self, *, ref_image_path: str, prompt: str,
                 ckpt_dir: str, training_config: str,
                 size: str = "720*400", infer_frames: int = 48,
                 geom: BlockGeometry | None = None):
        self._geom = geom or BlockGeometry()
        self._cfg = dict(ref_image_path=ref_image_path, prompt=prompt,
                         ckpt_dir=ckpt_dir, training_config=training_config,
                         size=size, infer_frames=infer_frames)
        self._pipe = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # ⚠️ 每次打断换一个「代号」。在途那一块回来时代号对不上就丢掉。
        #    这里**不能**靠 asyncio 取消 —— 模型跑在普通线程里，
        #    forward 进去了就取消不掉，只能等它回来再扔。
        #    （编排层那边靠 `Task.cancel()` 就够，那是 asyncio；这里不是。）
        self._epoch = 0
        self._size: tuple[int, int] = (0, 0)

    # ── FrameSource 协议 ──────────────────────────────────────────

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def start(self, audio_cb: Callable[[], np.ndarray], out: asyncio.Queue) -> None:
        if self._thread is not None:
            return
        loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._run, args=(audio_cb, out, loop),
            name="liveavatar-pipeline", daemon=True)
        self._thread.start()

    def reset(self) -> None:
        """被打断。换代号 —— 在途那一块回来会被丢掉。

        **不重启 pipeline** —— 重启要重新预热自回归状态，代价是几秒。
        丢掉在途那一块就够了；下一块自然用新的音频。
        """
        self._epoch += 1

    def stop(self) -> None:
        self._stop.set()

    # ── 模型线程 ──────────────────────────────────────────────────

    def _run(self, audio_cb, out: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        try:
            pipe = self._build()
        except Exception:
            log.exception("模型起不来，这一路退化成只出声")
            return

        # ⭐ 上游缺的就是这一行。callback 必须**恒定返回 block_samples 个
        #    采样**（编排层的 `_pull_block` 保证了这点）。
        pipe.get_audio_callback = audio_cb

        epoch = self._epoch
        try:
            for item in self._generate(pipe):
                if self._stop.is_set():
                    break
                if item is None:
                    continue            # DiT rank 只 yield None，不出帧
                if epoch != self._epoch:
                    # 打断过了。**整批扔掉** —— 不扔的话用户打断之后
                    # 还会看到上一句话的尾巴在动嘴。
                    epoch = self._epoch
                    continue
                for img in self._to_rgba(item):
                    loop.call_soon_threadsafe(out.put_nowait, img)
        except Exception:
            # 模型炸了不能把整条会话带走 —— 掉回「只出声」比整路断掉好。
            log.exception("生成中断，这一路退化成只出声")

    def _build(self):
        """载入模型。**惰性 import** —— 控制面那台机器上没有 torch，
        这个模块被 `worker/runner.py` 顶层 import 时不能因此炸掉。
        """
        import torch.distributed as dist
        from liveavatar.models.wan.causal_s2v_pipeline_tpp_blockwise import WanS2V
        from liveavatar.models.wan.wan_base.configs import WAN_CONFIGS

        if not dist.is_initialized():
            raise RuntimeError(
                "五卡流式 pipeline 必须在 torchrun 下跑（--nproc_per_node=5）。"
                "见本文件顶部关于进程形态那一段。")

        cfg = WAN_CONFIGS["s2v-14B"]
        pipe = WanS2V(config=cfg, checkpoint_dir=self._cfg["ckpt_dir"],
                      device_id=int(os.environ.get("LOCAL_RANK", 0)),
                      rank=dist.get_rank(), convert_model_dtype=True)
        w, h = (int(x) for x in self._cfg["size"].split("*"))
        self._size = (w, h)
        self._pipe = pipe
        return pipe

    def _generate(self, pipe):
        """调 `generate()`。它是生成器 —— 这正是 blockwise 跟另外两条的区别。"""
        return pipe.generate(
            input_prompt=self._cfg["prompt"],
            ref_image_path=self._cfg["ref_image_path"],
            audio_path=None,                    # 音频走 callback，不走文件
            num_repeat=10**9,                   # 流式本来就是无限跑，靠 stop() 收
            generate_size=self._cfg["size"],
            infer_frames=self._cfg["infer_frames"],
            sample_solver="euler", sampling_steps=4, guide_scale=0,
            offload_model=False, num_gpus_dit=4, enable_vae_parallel=True,
        )

    @staticmethod
    def _to_rgba(item) -> list[np.ndarray]:
        """模型吐的是 `[B,C,T,H,W]` 的 `[-1,1]` tensor，转成一串 RGBA `HxWx4`。"""
        import torch

        if not isinstance(item, torch.Tensor):
            return []
        x = item.detach().float().clamp_(-1, 1)
        x = (x + 1.0) * 127.5                    # [-1,1] → [0,255]
        if x.dim() == 5:
            x = x[0]                             # 去掉 batch
        # [C,T,H,W] → T 张 [H,W,C]
        frames = x.permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()
        out = []
        for f in frames:
            if f.shape[-1] == 3:                 # RGB → RGBA，alpha 全不透明
                f = np.concatenate(
                    [f, np.full(f.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
            out.append(np.ascontiguousarray(f))
        return out
