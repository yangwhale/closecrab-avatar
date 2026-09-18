"""数字人那条视频轨怎么编、怎么发。

## 为什么要接管这件事

`AvatarRunner._publish_track()` 发视频轨时是这么写的：

    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)

**一个编码参数都不传。** 对一般的摄像头画面这没问题，对我们不行 ——
2026-09-18 从 LiveKit 服务端实测到的默认结果是：

    VIDEO 384x704  simulcast=True  codec=video/VP8
        层 LOW:    360x660  bitrate=450000
        层 MEDIUM: 384x704  bitrate=800000

两个问题：

**一、simulcast 在这个分辨率上是退化的。** 两层只差 6%（360×660 对 384×704）。
simulcast 的意义是「网络不好就掉到小得多的那层」，而这里掉下去几乎不省什么，
却实打实付了两份代价：编两路、发两份关键帧。**拥塞时没有有效的逃生梯，
平时还多花一倍的关键帧带宽。**

分辨率本来就小的时候，simulcast 是净亏 —— 它是为 1080p 那种量级设计的。

**二、VP8 在 iPhone 上没有硬件解码。** iOS 只硬解 H.264 / HEVC，VP8 走软解，
吃 CPU、也更容易在丢包后卡住等关键帧。我们的画面是一张会说话的脸，
帧率比分辨率重要得多，H.264 是更合适的选择。

> 症状长什么样：Chris 报「一开始黑屏，视频流没过来，只有最后的静止图出现了」。
> 全程黑、结束时突然出一张 —— 这正是**解码器一直等不到一个能解的关键帧**、
> 直到流停了带宽空出来才收到的样子。音频全程正常，所以不是链路断了。

## 都能用环境变量盖掉

因为上面那两条是**推断**，不是实测到的根因 —— 真正的判据在 Chris 的 5G 链路上，
而那条链路我这边复现不了。所以每一项都留了开关，好让我们在真机上 A/B，
而不是改一版编译一版：

    CCA_VIDEO_CODEC       H264（默认）/ VP8 / AV1 / VP9 / H265
    CCA_VIDEO_BITRATE     单层码率，默认 1_500_000
    CCA_VIDEO_SIMULCAST   1 打开（默认关）
    CCA_VIDEO_DEGRADATION framerate（默认）/ resolution / balanced

**默认值本身也是一个待验证的假设，不是结论。**
"""

from __future__ import annotations

import logging
import os

from livekit import rtc
from livekit.agents.voice.avatar import AvatarRunner

log = logging.getLogger("closecrab.avatar.publish")

_CODECS = {
    "VP8": rtc.VideoCodec.VP8,
    "H264": rtc.VideoCodec.H264,
    "AV1": rtc.VideoCodec.AV1,
    "VP9": rtc.VideoCodec.VP9,
    "H265": rtc.VideoCodec.H265,
}

# 掉带宽时先牺牲哪个。**我们要保帧率** —— 这是一张对口型的脸，
# 掉帧率立刻就「假」了，掉分辨率只是糊一点。
_DEGRADATION = {
    "framerate": rtc.DegradationPreference.MAINTAIN_FRAMERATE,
    "resolution": rtc.DegradationPreference.MAINTAIN_RESOLUTION,
    "balanced": rtc.DegradationPreference.BALANCED,
}


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip()


def video_publish_options(*, fps: int) -> rtc.TrackPublishOptions:
    """按环境变量攒一份视频发布参数，并把最终值打进日志。

    **打日志这件事不是凑数的。** 这一层的每个参数错了都不报错，只是让画面
    在某些网络上变坏 —— 而「某些网络」里不包括我测试用的那条局域网。
    不打出来，下次排障又只能靠猜它到底发的是什么。
    """
    codec_name = _env("CCA_VIDEO_CODEC", "H264").upper()
    codec = _CODECS.get(codec_name)
    if codec is None:
        # 认不出来就退回默认并说清楚，别静默用一个谁也没选的值。
        log.warning("不认识的 CCA_VIDEO_CODEC=%r，退回 H264", codec_name)
        codec_name, codec = "H264", rtc.VideoCodec.H264

    bitrate = int(_env("CCA_VIDEO_BITRATE", "1500000"))
    simulcast = _env("CCA_VIDEO_SIMULCAST", "0") in ("1", "true", "yes", "on")
    deg_name = _env("CCA_VIDEO_DEGRADATION", "framerate").lower()
    degradation = _DEGRADATION.get(deg_name)
    if degradation is None:
        log.warning("不认识的 CCA_VIDEO_DEGRADATION=%r，退回 framerate", deg_name)
        deg_name, degradation = "framerate", rtc.DegradationPreference.MAINTAIN_FRAMERATE

    log.info("视频轨发布参数：codec=%s 码率=%d simulcast=%s 降级偏好=%s 帧率上限=%d",
             codec_name, bitrate, simulcast, deg_name, fps)

    return rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_codec=codec,
        video_encoding=rtc.VideoEncoding(max_framerate=fps, max_bitrate=bitrate),
        simulcast=simulcast,
        degradation_preference=degradation,
    )


class TunedAvatarRunner(AvatarRunner):
    """`AvatarRunner`，但视频轨按我们的参数发。

    ## 为什么是覆盖私有方法

    编码参数在库里是**写死在 `_publish_track()` 里的局部变量**，既没有构造参数
    也没有钩子能改。能走的只有三条路：改库（升级就没了）、把
    `publish_track` 猴补掉（影响这个进程里所有轨，包括音频）、覆盖这一个方法。
    第三条影响面最小。

    ## 代价说清楚：这是在依赖私有实现

    库一升级，`_publish_track` 的内部字段名可能就变了。所以构造时**先检查
    依赖的字段还在不在，不在就直接抛**。

    宁可起不来也不要静默退化 —— 静默的后果是「参数没生效」，
    而那正是这个类要修的那个 bug 本身，会绕一整圈回到原点。
    """

    _NEEDS = ("_lock", "_room_connected_fut", "_audio_source", "_video_source",
              "_audio_publication", "_video_publication", "_options", "_room")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        missing = [n for n in self._NEEDS if not hasattr(self, n)]
        if missing:
            raise RuntimeError(
                f"AvatarRunner 的内部结构变了，少了 {missing}。"
                "publish_opts.TunedAvatarRunner 依赖它们来接管编码参数 —— "
                "去对一下上游 _publish_track() 的实现再改这里。"
            )

    async def _publish_track(self) -> None:
        async with self._lock:
            await self._room_connected_fut

            audio_track = rtc.LocalAudioTrack.create_audio_track(
                "avatar_audio", self._audio_source)
            self._audio_publication = await self._room.local_participant.publish_track(
                audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            # ⚠️ 这一句照抄上游，别删：音频要等订阅确认，否则第一句话的头会丢。
            await self._audio_publication.wait_for_subscription()

            video_track = rtc.LocalVideoTrack.create_video_track(
                "avatar_video", self._video_source)
            self._video_publication = await self._room.local_participant.publish_track(
                video_track, video_publish_options(fps=int(self._options.video_fps)))
