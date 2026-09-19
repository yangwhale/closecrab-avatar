"""把**递交给 LiveKit 的那一模一样的**音视频原样落盘，供离线合成 mp4。

## 为什么要有这个

> Chris 2026-09-18：「我总觉得它嘴型对不上、也不流畅。你把生成的视频和音频
> 打包一份给我，别通过 LiveKit，我本地播一下 —— 咱们再决定是生成的问题、
> 传输的问题，还是 iOS APP 的问题。」

这是个**二分法探针**：链路是「模型 → 我们这一层 → LiveKit → 手机」，
四段里出问题的是哪一段，光看手机上的画面分不出来。

落盘点选在生成器 `__aiter__` 的 yield 处 —— **不是模型出帧处**。差别要紧：

    模型出帧处   只能证明「模型画得好不好」
    yield 处     还能证明「我们的配对和节奏对不对」

后者才是完整的「服务端这一半」。如果这份 mp4 顺、口型也对，那问题一定在
LiveKit 传输或者客户端；如果这份就不对，那再往上游查。

## 为什么写裸流而不是直接编码

编码要引入 PyAV / ffmpeg 的 Python 绑定，而 worker 那五个进程跑在
一个很讲究的环境里（torch + NCCL + FP8），多一个原生依赖就多一份风险。
裸流只用内置 `open()`，合成交给命令行 ffmpeg，在 worker 之外做。

⚠️ 音频**按原样写**，不重采样。采样率从第一帧记下来，合成时告诉 ffmpeg。
自作主张重采样的话，「对不对得上」这件事就被我们自己搅浑了。
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time

log = logging.getLogger("closecrab.avatar.dump")

_ENV = "CCA_AV_DUMP"
"""设成一个目录就开始录。**默认不开** —— 一分钟视频约 1.6 GB 裸 RGBA。"""


class AVDump:
    """录一场。**任何一步失败都只记日志，绝不影响正常推流。**

    这是个排障工具：它坏了顶多是没录上，而它把会话搞挂就是本末倒置。
    """

    def __init__(self, geom_fps: int, size: tuple[int, int],
                 frames_per_block: int = 12) -> None:
        self.on = False
        self._dir: pathlib.Path | None = None
        self._v = self._a = None
        self._fps = geom_fps
        self._fpb = frames_per_block
        self._size = size
        self._rate = 0
        self._ch = 1
        self._nv = self._na = 0

        root = (os.environ.get(_ENV) or "").strip()
        if not root:
            return
        try:
            d = pathlib.Path(root) / time.strftime("%H%M%S")
            d.mkdir(parents=True, exist_ok=True)
            self._dir = d
            self._v = open(d / "video.rgba", "wb")
            self._a = open(d / "audio.pcm", "wb")
            self.on = True
            log.info("⭐ 开始录制音视频裸流 → %s（%dx%d @ %d fps）",
                     d, size[0], size[1], geom_fps)
        except Exception:                                # noqa: BLE001
            log.warning("录制开不起来，跳过", exc_info=True)
            self.on = False

    def video(self, data: bytes) -> None:
        if not self.on:
            return
        try:
            self._v.write(data)
            self._nv += 1
        except Exception:                                # noqa: BLE001
            self._fail()

    def audio(self, data: bytes, *, sample_rate: int, num_channels: int) -> None:
        if not self.on:
            return
        try:
            # ⚠️ 采样率取**第一帧**的，之后不再改。中途变了说明上游换了格式，
            #    那种情况下合出来的东西本来就没有参考价值 —— 记一条警告让人知道。
            if not self._rate:
                self._rate, self._ch = sample_rate, num_channels
            elif sample_rate != self._rate:
                log.warning("录制中采样率变了 %d → %d，这份录音不可信",
                            self._rate, sample_rate)
            self._a.write(data)
            self._na += 1
        except Exception:                                # noqa: BLE001
            self._fail()

    def close(self) -> None:
        """收尾并写一份 `meta.json`。**合成脚本只读这个文件**，不猜参数。"""
        if not self.on:
            return
        self.on = False
        try:
            for f in (self._v, self._a):
                if f:
                    f.close()
            meta = {
                "width": self._size[0], "height": self._size[1], "fps": self._fps,
                "sample_rate": self._rate or 48000, "channels": self._ch,
                "video_frames": self._nv, "audio_frames": self._na,
                "video_seconds": round(self._nv / self._fps, 2),
                # 合片脚本拿它算容差：段尾补零那一块让视频总比音频长
                # 0~一块，是确定性的，不该报警。写死 12 会在换几何时悄悄失准。
                "frames_per_block": self._fpb,
            }
            (self._dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("录制结束 → %s（视频 %d 帧 / %.1f s，音频 %d 帧）",
                     self._dir, self._nv, self._nv / self._fps, self._na)
        except Exception:                                # noqa: BLE001
            log.warning("录制收尾失败", exc_info=True)

    def _fail(self) -> None:
        log.warning("录制写盘失败，本场停止录制", exc_info=True)
        self.on = False
