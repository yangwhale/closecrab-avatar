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

import json
import logging
import os
import pathlib
import queue
import threading

import numpy as np

from .audio_feat import StreamingAudioFeat, UtteranceAudioFeat
from .audio_stream import BlockGeometry, PcmInbox

log = logging.getLogger("closecrab.avatar.pipeline")


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        log.warning("%s 不是整数，用默认 %d", name, default)
        return default

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
        self._feat: StreamingAudioFeat | None = None
        self._frame_seq = 0
        self._real_dropped = 0
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

    @property
    def pending(self) -> int:
        """还有多少帧攒着没被取走。给上层做抖动缓冲用。

        `queue.Queue.qsize()` 在多线程下是**近似值**（文档明说）。这里够用：
        判据是「攒够了没有」，差一两帧无所谓；而要精确就得加锁，
        那会把模型线程和事件循环绑在一起 —— 为一个阈值判断不值当。
        """
        return self._frames.qsize()

    def next_frame(self) -> np.ndarray | None:
        try:
            return self._frames.get_nowait()
        except queue.Empty:
            return None

    def drop_pending_frames(self, why: str) -> int:
        """把**已经生成、还没发出去**的帧全部扔掉，配对基准归零。

        返回扔了几帧。

        ## 为什么这件事必须由上层在「一句话开始」那一刻触发

        ⚠️ **不要挂在「音频输入队列变空」上。** 试过，错得很彻底：

        实时推流下 inbox **本来就一直在空** —— 音频按 40 ms 一小块推进来，
        模型消费得比推得快，于是一句话中间 inbox 每秒要见底好几次。挂在
        那个事件上等于**一边说话一边不停清自己的帧**，实测同一段音频两次
        会话，一次收到 0 帧、一次 140 帧，差别只是推送节奏碰巧谁快一点。

        「队列空了」是个**会反复发生的状态**；「一句话开始了」才是**边界**。
        清缓冲这种破坏性动作只能挂在边界上。
        """
        n = 0
        while True:
            try:
                self._frames.get_nowait(); n += 1
            except queue.Empty:
                break
        if n:
            log.info("抽干出帧队列（%s）：扔掉残留 %d 帧，配对基准归零", why, n)
        return n

    def reset(self) -> None:
        """被打断：把没念的 PCM 和已生成的在途帧一起扔掉。

        **不重启 pipeline** —— 重启要重新预热自回归状态，代价是几秒，
        而且五个 rank 得一起重启（见文件头）。丢在途的就够了。
        """
        self.inbox.clear()
        if self._feat is not None:
            # 回看窗口里那一两秒是上一句的 —— 留着会当成下一句的上下文，
            # 而 wav2vec 带自注意力，上下文会实实在在改掉特征值。
            self._feat.reset()
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
        # ⭐ 包一层再挂上去：**这是唯一能知道「第几块被吃掉了」的地方**。
        #    两个使用点（`get_audio_callback` 和 `StreamingAudioFeat` 的 pull）
        #    都走它，漏掉任何一个账本就从此错位。
        # ⚠️ 临时计数：查「一块音频是不是被拉了两次」。
        #    实测配对落盘里音频正好是视频的 2 倍 —— 要么两条路都在拉，
        #    要么有一条在白拉（拉走的那块直接丢了）。**后者是重大 bug**。

        pipe.get_audio_callback = self.inbox.pull_block

        # ⭐ 连**怎么编码**也得换掉，不只是「从哪拿音频」。
        #    上游 `_streaming_encode_next_audio_block_or_random` 把每一块
        #    0.48 s 音频**单独**过一遍 wav2vec，于是时间轴被拉伸 10%、
        #    每 12 帧还有 1 帧被喂静音。量化证据和推导全在 `audio_feat.py`。
        #    这里换成滑动窗口版（只回看、不预看，不增加延迟）。
        #    实例属性遮住类方法，上游那句 `self._streaming_...()` 就走到这儿。
        def _old_way(chunk, block_frames: int):
            """回退：上游那条逐块编码。**拿已经取出来的那一块，不重新拉。**

            直接调上游那个方法的话它会自己再 `get_audio_callback()` 拉一块，
            每次回退吃掉两块音频 —— 持续回退时画面跑成两倍速。
            """
            emb, _ = pipe.encode_audio_from_array(chunk, infer_frames=block_frames)
            return emb[..., :block_frames].contiguous()

        # ⭐ **可以一键换回上游那条。** `CCA_AUDIO_FEAT=upstream`
        #    口型不准时这是最值钱的一刀：我们只换了「怎么把音频编成特征」
        #    这一件事，换回去还不准，说明问题不在这儿，可以整块排除。
        #    没有这个开关的话，只能靠读代码猜 —— 而这一段的对错**不可能
        #    靠读代码判断**，它是个实测问题（见 `block_indices` 的注释：
        #    物理上更准的做法实测反而更差）。
        # 三种音频编码，`CCA_AUDIO_FEAT` 选：
        #   utterance（默认）整句到齐再编码 —— 跟离线一致，见 UtteranceAudioFeat
        #   sliding        边来边翻的滑动窗口 —— 我们的旧版，天花板 0.81
        #   upstream       上游原版逐块编码 —— 对照用，0.49
        # 留着后两个不是为了将来可能用，是因为**这三条的优劣只能实测**，
        # 没有开关就只能靠读代码猜，而这件事已经猜错过两轮。
        # ⚠️ **默认仍是 sliding，不是 utterance。** utterance 那条口型对得多，
        #    但**还没完工**：没人说话时上游照样要块（不给就五卡死锁），那些
        #    静音块生成的画面目前会照发 —— 实测 12 s 音频出 57.8 s 视频，
        #    音频跟着空转的画面跑，等于又不同步。
        #
        #    缺的那一环是「这一帧是哪一块生成的」，好把静音块的帧扔掉。
        #    那正是今天上午删掉的 av_ledger 干的事 —— 删的时候以为抽干方案
        #    让它没用了，其实只是那会儿还没遇到「必须空转」这个约束。
        #    **要装回来，但今晚不装。**
        mode = (os.environ.get("CCA_AUDIO_FEAT") or "sliding").strip().lower()
        if mode == "upstream":
            log.warning("⚠️ CCA_AUDIO_FEAT=upstream：走上游原版逐块编码。对照用，不是常态。")
            self._feat = None
        elif mode == "sliding":
            log.warning("⚠️ CCA_AUDIO_FEAT=sliding：走旧的滑动窗口。对照用，不是常态。")
            self._feat = StreamingAudioFeat(
                pipe.audio_encoder, pull=self.inbox.pull_block,
                block_samples=self.geometry.block_samples, fps=self.geometry.fps,
                device=pipe.device, dtype=pipe.param_dtype, fallback=_old_way)
            pipe._streaming_encode_next_audio_block_or_random = self._feat.next_block
        else:
            self._feat = UtteranceAudioFeat(
                pipe.audio_encoder, pull_utterance=self.inbox.pull_utterance,
                fps=self.geometry.fps, sample_rate=self.geometry.sample_rate,
                device=pipe.device, dtype=pipe.param_dtype, fallback=_old_way)
            pipe._streaming_encode_next_audio_block_or_random = self._feat.next_block

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
        """塞一帧。满了丢**最旧**的 —— 数字人只有「现在」有意义。

        ⚠️ **这里丢掉的帧才是真的丢了**，而且每丢一帧，音画的对应关系就
        永久错开一帧（音频那一路不丢）。原来这件事没有任何计数、没有日志 ——
        一个持续发生、谁也不知道发生了多少次的静默退化。现在数它。
        """
        try:
            self._frames.put_nowait(img)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
            self._real_dropped += 1
            if self._real_dropped in (1, 10) or self._real_dropped % 100 == 0:
                log.warning("出帧队列满，已丢掉最旧的 %d 帧（每丢一帧，"
                            "音画对应关系就永久错开 40 ms）", self._real_dropped)
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
