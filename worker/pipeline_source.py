"""`FrameSource` 的真实现 —— 驱动上游流式 pipeline。

## ⚠️ 这个文件决定了 worker 的进程形态，先读这一段

五卡流式路径**不是「一进程一张卡」**。它是 `torchrun --nproc_per_node=5`
拉起的一组 5 个进程：

    rank 0–3   DiT，纯算。`generate()` 全程 `yield None`，不碰 LiveKit
    rank 4     VAE。**只有它**调音频回调、只有它 `yield` 真图

这不是我们的分工，是上游 pipeline 里写死的（`in_dit_device = rank < num_gpus_dit`，
音频回调在 `not in_dit_device` 分支里）。所以：

- systemd 的 worker unit 是 **一组 5 张卡一个 unit**，不是一卡一个
- 一台 8 卡机器跑五卡流式 = **1 组**（剩 3 张闲着）

## ⛔ 五卡这条就是单租户，这是定下来的，不要再去改它

Chris 2026-09-18 拍板：「咱们就单租户就行。多租户到时候我们再优化好
一卡一用户的时候再说。这 5 个卡的就不多租户。」

`batch>1` 在这条管线上结构性不成立：`_initialize_kv_cache(batch_size=1)`
把 batch 写死了，而且 KV cache / 运动帧 / 注意力锚点 / 音频模板**全挂在模型
对象上**，两路请求会互相踩。提并发靠**加组**，不是组内 batch。

## 换脸不在这一层做

见 `docs/face-swap.md`。一句话：改上游 `generate()`，在轮次开头就地重算
参考图那几个张量。**这个文件里不要再出现任何「绕过上游」的机制** ——
试过三版，全部撞墙，记录在那份文档里。

要「一卡一路、8 路并发」得走单卡 pipeline，而**单卡那条没有流式出帧**
（`causal_s2v_pipeline` 里一个 `yield` 都没有），只能整段生成完再返回。
两条路的取舍见 `docs/benchmarks.md` 第一节 —— **这不是疏忽，是量出来的**。

## 一个进程 = 一个形象，`generate()` 开机就跑、永不结束

看起来激进，其实是**唯一不需要跨 rank 协调的形态**：

五个 rank 靠 `dist.send/recv` 逐块咬合前进，中途谁想退出，别人就永远卡在
`recv` 上。要支持「会话结束 → 停 → 下次再起」，就得再发明一套五卡同步的
停机协议。而只要 `generate()` 一直转，这个问题根本不存在。

**空转的代价被音频回调挡住了**：没人说话时 `PcmInbox.pull_block()` 阻塞，
rank 4 停在回调里，rank 0–3 自然堵在 `dist.recv`，整组一起停。来音频了
回调返回，整组自己接着跑。**省电这件事不需要额外写一行协调代码。**

代价是形象和 prompt 在进程启动时定死，换形象 = 换进程。对「一个 bot 一张脸」
的产品形态来说这正好；真要多形象，起多组。

## 上游缺的那一个钩子

流式 pipeline 第 0、1 轮用预置音频预热，**从第 2 轮起每个 block 调一次
`self.get_audio_callback()` 要实时 PCM**。上游没有任何入口装这个钩子，
这是「流式音频」缺的唯一一环 —— 装上就通。
"""

from __future__ import annotations

import logging
import os
import queue
import threading

import numpy as np

from .audio_stream import BlockGeometry, PcmInbox

log = logging.getLogger("closecrab.avatar.pipeline")

# 出帧队列上限。会话没接上的时候帧会堆在这儿 —— 堆满就丢**最旧**的，
# 因为数字人只有「现在」有意义，补播两秒前的嘴型比丢帧更糟。
_FRAME_QUEUE_MAX = 64


class LiveAvatarPipelineSource:
    """把上游 blockwise pipeline 包成 `FrameSource`。

    模型是**阻塞的同步生成器**（torch，不是 asyncio），跑在一个单独的线程里。
    跟事件循环的交接只有两个点，两个都不需要事件循环的引用：

        inbox        带锁的 PCM 缓冲（事件循环写，模型线程读）
        next_frame() 非阻塞轮询（事件循环读，模型线程写）
    """

    def __init__(self, *, ref_image_path: str, prompt: str,
                 ckpt_dir: str, training_config: str,
                 warmup_audio: str, lora_path: str = "",
                 size: str = "720*400", infer_frames: int = 48,
                 task: str = "s2v-14B", seed: int = 420,
                 use_fp8: bool = True, num_gpus_dit: int = 4):
        self._cfg = dict(ref_image_path=ref_image_path, prompt=prompt,
                         ckpt_dir=ckpt_dir, training_config=training_config,
                         warmup_audio=warmup_audio, lora_path=lora_path, size=size,
                         infer_frames=infer_frames, task=task, seed=seed,
                         use_fp8=use_fp8, num_gpus_dit=num_gpus_dit)
        self._pipe = None
        self._geom: BlockGeometry | None = None
        self._inbox: PcmInbox | None = None
        self._frames: queue.Queue = queue.Queue(maxsize=_FRAME_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._size: tuple[int, int] = (0, 0)
        self._is_vae_rank = False
        self._frame_seq = 0
        self._dump_dir = os.environ.get("CCA_FRAME_DUMP") or None
        if self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)

    # ── FrameSource 协议 ──────────────────────────────────────────

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def geometry(self) -> BlockGeometry:
        assert self._geom is not None, "还没 load()"
        return self._geom

    @property
    def inbox(self) -> PcmInbox:
        assert self._inbox is not None, "还没 load()"
        return self._inbox

    def next_frame(self) -> np.ndarray | None:
        try:
            return self._frames.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """被打断：把没念的 PCM 和已生成的在途帧一起扔掉。

        **不重启 pipeline** —— 重启要重新预热自回归状态，代价是几秒，
        而且五个 rank 得一起重启（见文件头）。丢在途的就够了。
        """
        self.inbox.clear()
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    # ── 装载 / 起跑 ───────────────────────────────────────────────

    def load(self) -> None:
        """**在 worker 启动时调，不要拖到第一句话。**

        权重 47 GB、`torch.compile` 还要几十秒，实测冷启动 ~160 s。拖到
        第一句话意味着第一个用户等三分钟 —— benchmark 里「常驻后 0.7 s」
        那个数字的前提就是这一步已经付过账了。
        """
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "五卡流式 pipeline 必须在 torchrun 下跑（--nproc_per_node=5）。"
                "见本文件顶部关于进程形态那一段。")

        # ⚠️ 配置树认准 `wan_2_2`。`wan_base` 下有个同名的 `WAN_CONFIGS`，
        #    import 一样成功，但 `sample_fps` 是 16 不是 25 —— 出来的视频
        #    时长对不上音频，而且不报错。官方入口用的就是 wan_2_2 这棵。
        from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from liveavatar.models.wan.causal_s2v_pipeline_tpp_blockwise import WanS2V
        from liveavatar.utils.args_config import parse_args_for_training_config

        rank, world = dist.get_rank(), dist.get_world_size()
        n_dit = self._cfg["num_gpus_dit"]
        self._is_vae_rank = rank >= n_dit

        cfg = WAN_CONFIGS[self._cfg["task"]]
        self._geom = BlockGeometry.from_model_config(cfg)
        self._inbox = PcmInbox(self._geom)
        log.info("块几何来自模型配置：fps=%d，一块 %d 帧 / %.2f s / %d 采样",
                 self._geom.fps, self._geom.frames_per_block,
                 self._geom.block_seconds, self._geom.block_samples)

        ts = parse_args_for_training_config(self._cfg["training_config"])

        pipe = WanS2V(
            config=cfg,
            checkpoint_dir=self._cfg["ckpt_dir"],
            device_id=int(os.environ.get("LOCAL_RANK", 0)),
            rank=rank,
            t5_fsdp=False, dit_fsdp=False,
            use_sp=False, sp_size=1, t5_cpu=False,
            convert_model_dtype=True,
            single_gpu=False,
            offload_kv_cache=False,
        )

        # ⭐ **没有这一步就不是 Live-Avatar，只是底座 Wan2.2-S2V。**
        #    4 步蒸馏（sampling_steps=4 / guide_scale=0）全靠这个 DMD LoRA；
        #    不加载的话模型照样跑、照样出画面，只是 4 步下画质稀烂 ——
        #    典型的「看起来在工作」的失效。
        # ⚠️ 传 HF repo id 的话上游会 `hf_hub_download(cache_dir="ckpt/LiveAvatar")`
        #    —— **相对 CWD**。worker 的 CWD 不是 LiveAvatar 目录，于是每台机器
        #    每次都重下一份 1.35 GB 到别的地方。给绝对路径就走本地分支。
        lora = (self._cfg["lora_path"] or ts.get("pretrained_lora_path")
                or "Quark-Vision/Live-Avatar")
        pipe.noise_model = pipe.add_lora_to_model(
            pipe.noise_model,
            lora_rank=ts["lora_rank"], lora_alpha=ts["lora_alpha"],
            lora_target_modules=ts["lora_target_modules"],
            init_lora_weights=ts["init_lora_weights"],
            pretrained_lora_path=lora,
            load_lora_weight_only=False,
        )
        log.info("已加载 LoRA：%s（rank %s / alpha %s）",
                 lora, ts["lora_rank"], ts["lora_alpha"])

        if self._cfg["use_fp8"] and hasattr(torch, "_scaled_mm"):
            from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
            replace_linear_with_scaled_fp8(pipe.noise_model, ignore_keys=[
                'text_embedding', 'time_embedding', 'time_projection',
                'head.head', 'casual_audio_encoder.encoder.final_linear',
            ])
            log.info("已切 FP8 线性层")

        # ⚠️ 真实出帧尺寸**不是请求值**。`generate_size` 在这个 pipeline 里是
        #    死参数（tpp 版那行还被注释掉了），分辨率由 `max_area` ＋ 参考图
        #    宽高比按 64 取整决定 —— 请求 720×400 实际出 704×384。
        #    `AvatarOptions` 必须拿这个真实值，否则帧尺寸对不上。
        self._max_area = MAX_AREA_CONFIGS[self._cfg["size"]]
        from PIL import Image
        with Image.open(self._cfg["ref_image_path"]) as im:
            src_h, src_w = im.size[1], im.size[0]
        h, w = pipe.get_size_less_than_area(src_h, src_w, target_area=self._max_area)
        self._size = (int(w), int(h))
        if (w, h) != tuple(int(x) for x in self._cfg["size"].split("*"))[::-1]:
            log.info("出帧尺寸 %d×%d（请求 %s，按 64 的网格取整）",
                     w, h, self._cfg["size"])

        self._pipe = pipe
        log.info("rank %d/%d 模型就绪（%s）", rank, world,
                 "VAE：拉音频 + 出帧" if self._is_vae_rank else "DiT：纯算")

    def start(self) -> None:
        """起模型线程 —— 给 **VAE rank** 用，它的主线程要跑 asyncio。"""
        assert self._pipe is not None, "先 load() 再 start()"
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="liveavatar-pipeline",
                                        daemon=True)
        self._thread.start()

    def run_blocking(self) -> None:
        """在**当前线程**把 `generate()` 跑到底 —— 给 **DiT rank** 用。

        它们不碰控制面也不碰 LiveKit，没有别的事要做，凭空起个线程再让
        主线程 join 只是多一层。
        """
        assert self._pipe is not None, "先 load() 再 run_blocking()"
        self._run()

    def stop(self) -> None:
        if self._inbox is not None:
            self._inbox.close()          # 把阻塞在回调里的模型线程放出来

    # ── 模型线程 ──────────────────────────────────────────────────

    def _run(self) -> None:
        pipe = self._pipe
        # ⭐ 上游缺的就是这一行。只有 VAE rank 会调它（见文件头）；
        #    DiT rank 装了也不会被调，装上无害、少一个分支。
        pipe.get_audio_callback = self.inbox.pull_block

        try:
            for item in self._generate(pipe):
                if self.inbox.closed:
                    break
                if item is None:
                    continue             # DiT rank 只 yield None，不出帧
                for img in self._to_rgba(item):
                    self._probe(img)
                    self._offer(img)
        except Exception:
            # 模型炸了不能把整条会话带走 —— 掉回「只出声」比整路断掉好。
            log.exception("生成中断，这一路退化成只出声")

    def _probe(self, img: np.ndarray) -> None:
        """量一下**发出去的像素本身**，别只量「有没有帧」。

        ⚠️ 这条是补上一次误判的。上一轮排障我一路在数「有没有视频轨」「有没有
        帧」，两项都正常，于是判定链路通了 —— 而屏幕上是纯黑。**「帧在流」和
        「帧上有东西」是两件事**，中间隔着整个模型；只量前者，一个全黑的
        输出会一路绿灯走到用户眼前。

        代价是每帧三个 numpy 归约（几十微秒，相对 40 ms 的帧间隔可以忽略），
        而且默认 25 帧才打一行，不刷屏。要看单帧就设 `CCA_FRAME_DUMP=<目录>`。
        """
        self._frame_seq += 1
        rgb = img[..., :3]
        if self._dump_dir and self._frame_seq <= 8:
            try:
                from PIL import Image
                Image.fromarray(rgb).save(
                    f"{self._dump_dir}/f{self._frame_seq:04d}.png")
            except Exception:
                log.debug("帧转储失败", exc_info=True)
        if self._frame_seq % 25:
            return
        mean = float(rgb.mean())
        black = float((rgb.max(axis=2) < 16).mean())
        # 全黑不是「偏暗」，是**另一类故障** —— 单独说出来，别让人去调亮度。
        tag = "  ⚠️ 基本全黑" if black > 0.9 else ""
        log.info("帧 #%d  均值=%.1f  标准差=%.1f  黑像素比=%.2f%s",
                 self._frame_seq, mean, float(rgb.std()), black, tag)

    def _offer(self, img: np.ndarray) -> None:
        """塞一帧。满了丢**最旧**的 —— 数字人只有「现在」有意义。"""
        try:
            self._frames.put_nowait(img)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(img)
            except queue.Full:
                pass

    def _generate(self, pipe):
        """调 `generate()`。它是生成器 —— 这正是 blockwise 跟另外两条的区别。

        参数照抄官方 `infinite_inference_multi_gpu.sh`，两处例外都在注释里。
        """
        return pipe.generate(
            input_prompt=self._cfg["prompt"],
            ref_image_path=self._cfg["ref_image_path"],
            # ⚠️ **不能传 None。** 第 0、1 轮要用文件音频建编码模板
            #    （`self.audio_template = audio_emb[...]`），传 None 直接炸在
            #    `encode_audio()` 里。实时 PCM 从第 2 轮起才走回调。
            audio_path=self._cfg["warmup_audio"],
            max_area=self._max_area,      # 真正决定分辨率的是它，不是 generate_size
            infer_frames=self._cfg["infer_frames"],
            sample_solver="euler", sampling_steps=4, guide_scale=0,
            seed=self._cfg["seed"],
            offload_model=False,
            num_gpus_dit=self._cfg["num_gpus_dit"],
            enable_vae_parallel=True,
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
