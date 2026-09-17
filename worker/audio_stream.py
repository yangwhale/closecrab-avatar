"""音频流式化的调度算术 —— P1 的主要工作量，这里是能离线钉死的那一半。

## 这个文件解决什么

上游 `get_audio_embed_bucket_fps()` 的签名是「**给我整段音频**，我告诉你要转几轮」。
直播里拿不到整段：音频一块一块来，得「来一块编一块，还有音频就多转一轮」。

问题是能不能这么改**不是想当然** —— 得确认生成循环不会回头读到还没到的音频。
下面每个常数和每条公式都标了上游出处，是读源码抄的，不是推的。

## 上游源码里的事实（fork `b200-realtime` @ 87fa0ea）

`liveavatar/models/wan/wan_2_2/modules/s2v/audio_encoder.py`：

    video_rate = 30                                              # :64
    scale        = video_rate / fps                              # :176
    min_batch_num = int(audio_frame_num / (batch_frames*scale)) + 1   # :178
    bucket_num    = min_batch_num * batch_frames                 # :180
    batch_idx     = get_sample_indices(..., fixed_start=0)       # :183
    return batch_audio_eb, min_batch_num                         # :216  ← num_repeat

`get_sample_indices` 在 `fixed_start=0` 时退化成一条等差斜坡：

    time_points   = linspace(0, num_sample/fps, num_sample, endpoint=False)
    frame_indices = round(time_points * video_rate)  = round(i * scale)

**⭐ 关键：第 i 个槽位取哪一帧只跟 i 有关，跟总长无关。**
总长只影响两件事 —— 一共有多少槽位，以及 `bi >= audio_frame_num` 的槽位填零。
这一条就是「流式改造架构上不是死路」的真正依据。

消费端 `causal_s2v_pipeline_tpp_blockwise.py` 也确认了只往前读：

    audio_input = audio_emb[..., r*infer_frames : (r+1)*infer_frames]   # :843-848
    ... audio_input[..., blk*(nfpb*4) : (blk+1)*(nfpb*4)]               # :864-873

**没有任何一处读到 `(r+1)*infer_frames` 之后。**

## 实测参数（全部来自源码，不是猜的）

    video_rate    = 30     audio_encoder.py:64
    fps           = 16     wan_base/configs/shared_config.py:18 (sample_fps)
    infer_frames  = 80     causal_s2v_pipeline_tpp_blockwise.py:705 默认值
    audio_sample_m = 0     causal_s2v_pipeline_tpp_blockwise.py:192

于是 scale = 1.875，一轮吃 80×1.875 = 150 个音频嵌入帧 = **5 秒**，
出 80 个视频帧 @16fps = **5 秒**。两边对得上，这就是外部锚点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketGeometry:
    """把上游那套桶几何抄成可计算的形状。默认值即生产值，出处见模块注释。"""

    video_rate: int = 30
    fps: int = 16
    infer_frames: int = 80
    audio_sample_m: int = 0

    def __post_init__(self) -> None:
        # 这几条都是上游隐含的前提。不满足时**必须炸**而不是算出个错的数 ——
        # 算错的后果是嘴型和声音对不上，而那不会报任何错。
        if self.fps <= 0 or self.video_rate <= 0 or self.infer_frames <= 0:
            raise ValueError("video_rate / fps / infer_frames 都必须为正")
        if self.audio_sample_m < 0:
            raise ValueError("audio_sample_m 不能为负")

    @property
    def scale(self) -> float:
        """一个视频帧对应几个音频嵌入帧。上游 `scale = video_rate / fps`。"""
        return self.video_rate / self.fps

    @property
    def stride(self) -> int:
        """取上下文时的步长。上游写的是 `int(video_rate / fps)` —— **截断不是四舍五入**，
        30/16 → 1。照抄，别自作主张改成 round。
        """
        return int(self.video_rate / self.fps)

    @property
    def frames_per_repeat(self) -> float:
        """一轮吃掉多少音频嵌入帧（可能不是整数）。"""
        return self.infer_frames * self.scale

    def slot_frame(self, slot: int) -> int:
        """第 `slot` 个桶槽位取哪一个音频嵌入帧。

        ⚠️ 这里**不做上限裁剪**。上游的裁剪用的是「整段音频的长度」，
        流式下拿那个长度是不存在的；硬用当前已到长度去裁会得到不同的值，
        而且是静默不同 —— 口型会整体前移，没人会报错。
        裁剪的判断交给 `frames_needed_for_repeat` / `repeats_ready`。
        """
        if slot < 0:
            raise ValueError("slot 不能为负")
        # linspace(0, n/fps, n, endpoint=False) 的第 i 项就是 i/fps，
        # 乘 video_rate 再 round —— 等价于 round(i * scale)。
        # 用 numpy 的 round 语义（banker's rounding）与上游 np.round 对齐。
        return _np_round_half_even(slot * self.scale)

    def frames_needed_for_repeat(self, repeat: int) -> int:
        """第 `repeat` 轮要**至少有多少个音频嵌入帧已经到了**，才能不靠猜地生成。

        返回的是「帧数」不是「最大下标」—— 下标 + 1。
        """
        if repeat < 0:
            raise ValueError("repeat 不能为负")
        last_slot = (repeat + 1) * self.infer_frames - 1
        # m > 0 时每个槽位还要往后看 m*stride 帧（上游 chosen_idx 的右端）。
        # m == 0（生产值）时这一项是 0。
        return self.slot_frame(last_slot) + self.audio_sample_m * self.stride + 1

    def repeats_ready(self, frames_arrived: int, *, ended: bool) -> int:
        """已经到了 `frames_arrived` 帧，现在能安全生成几轮。

        - 流**没结束**：只算那些「所有槽位的音频都已到」的轮次。少一帧都不能开 ——
          开了就要拿裁剪值凑，口型会错。
        - 流**已结束**：剩下的尾巴也要出，不足的部分按上游语义填零。
        """
        if frames_arrived < 0:
            raise ValueError("frames_arrived 不能为负")
        if ended:
            return self.repeats_for_finished_stream(frames_arrived)
        r = 0
        while self.frames_needed_for_repeat(r) <= frames_arrived:
            r += 1
        return r

    def repeats_for_finished_stream(self, audio_frame_num: int) -> int:
        """整段音频一共要转几轮。**这里故意跟上游不一样，看下面。**

        上游是 `int(N / frames_per_repeat) + 1`（`min_batch_num`）。那个 `+1`
        保证覆盖，但**恒定多出一轮**：N 正好等于 150 时它给 2，而第二轮里
        第一个槽位取的帧下标就是 150 —— 已经越界，整轮几乎全是零填充。

        离线多渲染 5 秒只是浪费。**直播里那是数字人对着空气动 5 秒嘴**，
        用户会以为卡住了。所以流式这边用 `ceil`：

            上游   int(N/150) + 1      N=150 → 2 轮（第 2 轮全是静音）
            这里   max(1, ceil(N/150)) N=150 → 1 轮

        **这是有意偏离，不是 bug。** 写成单独一个函数、配对拍测试，
        就是为了让它不会被当成「哪里算错了」而被人「修」回去。
        两者的差只可能是 0 或 1 轮，由 `upstream_num_repeat` 对拍。
        """
        if audio_frame_num < 0:
            raise ValueError("audio_frame_num 不能为负")
        if audio_frame_num == 0:
            return 0          # 一点音频都没有就别转 —— 上游在这里会给 1 轮纯静音
        return max(1, math.ceil(audio_frame_num / self.frames_per_repeat))

    def upstream_num_repeat(self, audio_frame_num: int) -> int:
        """**照抄**上游 `min_batch_num`，只用来对拍，不参与调度。

        留着它是为了让偏离这件事有个锚：测试里逐点比对两者，
        确认差值永远只是 0 或 1，而不是某个地方算歪了。
        """
        if audio_frame_num < 0:
            raise ValueError("audio_frame_num 不能为负")
        return int(audio_frame_num / self.frames_per_repeat) + 1


def _np_round_half_even(x: float) -> int:
    """跟 `np.round` 一样的「五入到偶数」，不是 Python 内置 round 的语义……

    其实两者一致（Python3 的 round 也是 banker's rounding），单独写一个函数
    是为了**把这件事说出来**：上游用的是 `np.round`，如果哪天改成
    `np.floor` 或者 `int()`，这里要跟着改，而不是各自为政。

    差一帧的后果：整段口型平移 1/30 秒。听不出、但看得出，而且不会报错。
    """
    return int(round(x))


def seconds_to_embed_frames(seconds: float, geom: BucketGeometry) -> int:
    """秒 → 音频嵌入帧数。

    嵌入帧率是 30 Hz（wav2vec 出来是 50 Hz，`audio_encoder.py:86` 用
    `linear_interpolation` 重采样到 `video_rate`）。**别拿 PCM 采样率去换算。**
    """
    if seconds < 0:
        raise ValueError("seconds 不能为负")
    return int(seconds * geom.video_rate)
