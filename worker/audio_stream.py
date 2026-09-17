"""块几何 —— 回调该返回多长一段 PCM。

## 这个文件曾经有 200 行，现在只剩这么点

原来这里有一整套「什么时候攒够一轮音频可以开工」的调度算术：
`slot_frame` / `frames_needed_for_repeat` / `repeats_ready` / `gather_slots`…
跟上游 `get_audio_embed_bucket_fps` 逐点对拍过，测试也齐。

**然后发现它全是多余的。**

上游的流式 pipeline 自己按块拉音频、自己编码
（`_streaming_encode_next_audio_block_or_random` → `self.get_audio_callback()`），
调用方**只需要回答一个问题：一块是多少个采样**。那套调度是我在不知道
有这个回调时自己搭的第二套实现。

留个记号是因为这类错误会重犯：**在自己实现一套调度之前，先确认被调用方
是不是已经自己调度了。** 判据很简单 —— 看它向你「要」什么。它要整段
（`audio_path`），你才需要切；它要下一块（callback），切的是它。

（旧实现在 git 历史里，`git log -- worker/audio_stream.py`。）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockGeometry:
    """一块的大小。默认值即生产值，出处见下。

        fps               25     wan_2_2/configs/shared_config.py（**不是 wan_base 那份的 16**）
        frames_per_block  12     num_frames_per_block(3) × 4
        sample_rate    16000     audio_encoder.extract_audio_feat_from_array 的默认值
    """

    fps: int = 25
    frames_per_block: int = 12
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        # 这几个值错了不会报错，只会让口型跟声音差一截 —— 宁可起不来。
        if self.fps <= 0 or self.frames_per_block <= 0 or self.sample_rate <= 0:
            raise ValueError("fps / frames_per_block / sample_rate 都必须为正")

    @property
    def block_seconds(self) -> float:
        """一块对应多少秒视频。12 ÷ 25 = 0.48 s。"""
        return self.frames_per_block / self.fps

    @property
    def block_samples(self) -> int:
        """回调该返回多少个采样。

        ⚠️ 这里用**音频采样率**（16 kHz），不是音频嵌入帧率（30 Hz），
        也不是视频帧率。三个率长得都像「帧率」，拿错了差几个数量级，
        而现象只是「怎么一直不出帧」。
        """
        return int(self.sample_rate * self.block_seconds)
