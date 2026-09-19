"""量「嘴张多大」—— **取样框只有这一处定义。**

⚠️ 这个框的位置我在 2026-09-19 错过两次：一次框到手上、一次框到脖子和
衣领上，两批数字全废，而失败时数字看起来毫无异常（只是相关偏低，
而「这段本来就难量」是个太顺手的解释）。

所以两条：

1. **只有这一个定义。** 任何要量嘴的脚本都从这儿 import，
   不许复制一份坐标过去 —— 两份坐标迟早有一份是错的，而且是悄悄错。
2. **采信数字之前先看图。** `draw_box()` 把框画在真实帧上，
   做成产物链进报告，让看结果的人自己判断，不靠作者自称。

坐标是 384×704 竖版头肩像上量准的。**换形象要重调**。
"""
from __future__ import annotations

import pathlib

import numpy as np

MOUTH = dict(y0=0.38, y1=0.51, x0=0.38, x1=0.66)


def mouth_openness(vraw: pathlib.Path, w: int, h: int) -> np.ndarray:
    """逐帧的「张嘴程度」＝ 嘴部方框里的暗像素占比。

    比帧间差分好：差分会把眨眼、头微动一起算进来，而那些跟说话无关。
    口腔内部比嘴唇和皮肤暗得多，张嘴在灰度上最稳的表现就是暗区变大。
    """
    y0, y1 = int(h * MOUTH["y0"]), int(h * MOUTH["y1"])
    x0, x1 = int(w * MOUTH["x0"]), int(w * MOUTH["x1"])
    fsz, rowsz = w * h * 4, w * 4
    rows = []
    with open(vraw, "rb") as f:
        for i in range(vraw.stat().st_size // fsz):
            f.seek(i * fsz + y0 * rowsz)              # 只读嘴那几行，别整帧搬
            band = np.frombuffer(f.read((y1 - y0) * rowsz), np.uint8)
            rows.append(band.reshape(y1 - y0, w, 4)[:, x0:x1, :3].mean(axis=2))
    g = np.stack(rows)
    thr = np.percentile(g, 12)                        # 全片统一阈值，别逐帧自适应
    return (g < thr).mean(axis=(1, 2))


def draw_box(vraw: pathlib.Path, w: int, h: int, frame: int, dst: pathlib.Path) -> None:
    """把画了取样框的那一帧存出来 —— **让框对不对变成可查的**。"""
    from PIL import Image, ImageDraw
    fsz = w * h * 4
    with open(vraw, "rb") as f:
        f.seek(min(frame, vraw.stat().st_size // fsz - 1) * fsz)
        im = Image.frombytes("RGBA", (w, h), f.read(fsz)).convert("RGB")
    ImageDraw.Draw(im).rectangle(
        [int(w * MOUTH["x0"]), int(h * MOUTH["y0"]),
         int(w * MOUTH["x1"]), int(h * MOUTH["y1"])], outline=(0, 200, 0), width=4)
    im.save(dst, quality=88)
