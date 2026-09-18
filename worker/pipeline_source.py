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

# 换脸（块边界上整组重开生成循环）。**默认关** —— 见 `_should_reload()` 顶部。
_FACE_SWAP_ENABLED = os.environ.get("CCA_FACE_SWAP", "0") == "1"


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
        # 换脸：VAE rank 记意愿，`_should_reload()` 广播给其余四个。
        # 加锁是因为 `request_face()` 在**事件循环线程**上调，而读它的是模型线程。
        self._face_lock = threading.Lock()
        self._pending_face: str | None = None
        self._vae_rank = 0
        # 换脸标志位专用的通信组，在 load() 里建。见那里的注释。
        self._ctrl_group = None
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
        # 广播 src 用它。**不能写死 0**：0 是 DiT rank，它不知道会话的事。
        self._vae_rank = n_dit

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

        # ⭐⭐ **换脸的标志位走一个专用通信组，绝不能借默认组。**
        #
        # 上游 `generate()` 在默认组里本来就有一对**不对称**的 broadcast：
        #   · DiT rank 在 `r==0 / r==1` 的轮次开头收（`rank != vae_rank` 分支）
        #   · VAE rank 在 `r==0` 最后一个 block 发
        # 也就是说同一个组里，不同 rank 在**完全不同的位置**发起集合通信，
        # 全靠「每个组内按发起顺序配对」这条规则才对得上。
        #
        # 我再往这个组里插一个「每块一次」的 broadcast，就会跟上游那两个
        # 抢着配对 —— 轻则读到别人的张量，重则整组死等。**而且不报错。**
        #
        # 建一个独立的组就把这整类风险切断了：两个组的配对各算各的。
        # `new_group()` 必须**所有 rank 都调、且顺序一致** —— 放在 load() 里
        # 正好满足（五个 rank 都会走到这儿，且只走一次）。
        self._ctrl_group = dist.new_group(ranks=list(range(world)))

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

    def request_face(self, path: str) -> None:
        """换一张参考图。**不立刻生效** —— 下一个块边界上整组一起换。

        只有拿到会话的那个 rank（VAE rank）会调这个方法；它把意愿记下来，
        由 `_should_reload()` 广播给其余四个 rank。理由见那个方法。
        """
        with self._face_lock:
            if path and path != self._cfg["ref_image_path"]:
                self._pending_face = path
                log.info("收到换脸请求：%s（下一个块边界生效）", path)

    def _should_reload(self) -> bool:
        """整组要不要在这里一起重开生成循环。**这是一次集合通信。**

        ## 为什么必须广播，不能各自判断

        参考图是在 `generate()` 开始那一刻传进去的，换脸 ＝ 重开那个循环。
        而这个循环是五个 rank 一起跑的：DiT rank 在里面 `dist.send`，
        VAE rank 在里面 `dist.recv`。**只要有一个 rank 提前跳出去，
        剩下的就会永远堵在通信上** —— 不报错，整条流水线静默死掉。

        「谁想换脸」这件事只有 VAE rank 知道（会话在它手上），所以由它
        广播一个标志位。这里的关键是**每个 rank 每个块都调一次**，
        调用点和次数完全一致，集合通信才不会错位。

        ⚠️ 上游那个生成器在每个 rank 上都是**一个块 yield 一次**
        （DiT rank `yield None`，VAE rank `yield image`），所以「每 yield 一次
        调一次」在五个 rank 上是对齐的。这条是这套协调能成立的前提，
        改上游版本时要重新确认。
        """
        # ⚠️ **默认关掉。** 2026-09-18 上线这套之后 worker 零出帧 —— 会话建得起来、
        #    视频轨也发了，但一帧都没生成，高度怀疑是这个每块一次的集合通信
        #    在某个 rank 上对不齐、整组卡死（正是这套协调最怕的那种失效：
        #    不报错，只是安静地停住）。
        #
        #    没查清之前先让它不生效 —— **能用的旧脸 >> 卡死的新脸**。
        #    查清后设 `CCA_FACE_SWAP=1` 打开，或者去掉这个开关。
        if not _FACE_SWAP_ENABLED:
            return False

        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            # 单卡跑（测试 / 退化模式）：没有别人要等，自己说了算。
            with self._face_lock:
                return self._pending_face is not None

        if self._ctrl_group is None:
            # 组还没建就别发集合通信 —— 借默认组正是这里要避免的事。
            return False

        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        if self._is_vae_rank:
            with self._face_lock:
                flag[0] = 1 if self._pending_face is not None else 0
        # src 必须是 VAE rank 的全局 rank —— 它是唯一知道会话的那个。
        dist.broadcast(flag, src=self._vae_rank, group=self._ctrl_group)
        if int(flag[0]) == 0:
            return False

        # 换脸的路径本身也要广播：DiT rank 不知道图在哪，而 `generate()`
        # 在每个 rank 上都要用到它（各自都会去读那张图）。
        buf = torch.zeros(512, dtype=torch.uint8, device="cuda")
        if self._is_vae_rank:
            with self._face_lock:
                raw = (self._pending_face or "").encode()[:512]
            buf[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device="cuda")
        dist.broadcast(buf, src=self._vae_rank, group=self._ctrl_group)
        path = bytes(buf.cpu().numpy()).rstrip(b"\x00").decode(errors="replace")
        if path:
            self._cfg["ref_image_path"] = path
        with self._face_lock:
            self._pending_face = None
        return True

    def _run(self) -> None:
        pipe = self._pipe
        # ⭐ 上游缺的就是这一行。只有 VAE rank 会调它（见文件头）；
        #    DiT rank 装了也不会被调，装上无害、少一个分支。
        pipe.get_audio_callback = self.inbox.pull_block

        # 外层这个 while 就是「换脸」：内层的 generate() 一旦跳出来，
        # 就带着新的 ref_image_path 重开一轮。重开只重跑 Step 1（编码参考图 +
        # VAE encode motion + 一次 barrier），**不重新加载模型权重** ——
        # 所以是秒级，不是启动时那四分钟。
        while not self.inbox.closed:
            try:
                self._run_once(pipe)
            except Exception:
                # 模型炸了不能把整条会话带走 —— 掉回「只出声」比整路断掉好。
                log.exception("生成中断，这一路退化成只出声")
                return
            if self.inbox.closed:
                return
            log.info("换脸重开生成循环：%s", self._cfg["ref_image_path"])

    def _run_once(self, pipe) -> None:
        """跑一轮生成循环。**正常返回 = 要换脸重开**，抛异常 = 真出事了。"""
        for item in self._generate(pipe):
            if self.inbox.closed:
                break
            # ⚠️ 放在这里而不是循环末尾：每个 rank 每个块都要走到一次，
            #    `continue` 之前也不能跳过（DiT rank 走的正是 continue 那条）。
            #    漏掉任何一个 rank 的任何一次，集合通信就错位、整组死等。
            if self._should_reload():
                return
            if item is None:
                continue                 # DiT rank 只 yield None，不出帧
            for img in self._to_rgba(item):
                self._probe(img)
                self._offer(img)

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
