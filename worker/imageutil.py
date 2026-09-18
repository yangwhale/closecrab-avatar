"""参考图的小处理。**只依赖 PIL** —— 这样它能在没装 livekit 的机器上跑测试。

拆出来是有具体理由的：`worker/runner.py` 一 import 就要拽进整个 livekit 栈，
而这里的逻辑（EXIF 摆正）是纯像素的事，跟 RTC 一点关系都没有。
绑在一起的后果是本机跑不了它的单测 —— 而这块逻辑恰恰只能靠单测验。
"""

from __future__ import annotations

import logging

log = logging.getLogger("closecrab.avatar.imageutil")

def upright(data: bytes) -> bytes | None:
    """按 EXIF 的方向标记把照片摆正。摆不了就返回 None，调用方用原图。

    ## 为什么必须做

    手机拍的照片**几乎都是横着存、靠一个 EXIF 标记表达「其实是竖的」**。
    `Image.open(...).convert("RGB")` 不理那个标记 —— 模型于是拿到一张躺倒的
    人像。而我们的出画是 384×704 **竖屏**，裁剪会在一张横着的图上取中间一条，
    脸多半被切掉或者压扁。

    2026-09-18 实测：Chris 传的那张自拍 `Orientation=6`（顺时针 90°），
    存的是 5712×4284，摆正后才是 4284×5712。整条换脸链路每一环都「成功」了
    —— 补丁在、文件写了、五个 rank 都打了「换脸生效」—— 只是喂进去的图是躺的。
    **又一个每层自查都正常的静默失败。**

    ⚠️ 放在这里而不是上传那一侧：网关存的是**用户的原始字节**，不该被我们
    重编码；而这里写的是给模型吃的那一份，本来就是派生物。
    """
    try:
        import io

        from PIL import Image, ImageOps

        im = Image.open(io.BytesIO(data))
        if im.getexif().get(274, 1) in (1, None):
            return None                      # 本来就是正的，别白重编码一遍
        out = io.BytesIO()
        ImageOps.exif_transpose(im).convert("RGB").save(out, format="JPEG", quality=95)
        log.info("形象图按 EXIF 摆正：%s → %s", im.size,
                 ImageOps.exif_transpose(im).size)
        return out.getvalue()
    except Exception:                        # noqa: BLE001
        log.warning("摆正形象图失败，用原图", exc_info=True)
        return None
