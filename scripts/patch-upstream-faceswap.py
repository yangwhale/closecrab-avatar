#!/usr/bin/env python3
"""给上游 `generate()` 打换脸补丁。**幂等**，可以反复跑。

## 为什么必须改上游源码

参考图派生出来的那几个张量（`ref_pixel_values` / `ref_latents` /
`motion_latents` / `videos_last_frames`）全是 `generate()` 里的**局部变量**，
monkeypatch 够不着。三条「不改上游、从外面绕」的路全部撞墙，记录见
`docs/face-swap.md`。

## 补丁做什么

在外层 `for r in range(active_nr):` 的第一行插一段：读换脸文件，图变了就把
那四个张量按 Step 1 同样的算式**就地重算**，然后照常往下跑。

不退出生成器、不重开循环、不重启进程 —— 省掉的正是「重新进入 generate()」
那一整套准备（加载权重、合 LoRA、转 FP8、编译），而那套跟换图毫无关系。

五个 rank 走的是**同一行代码**（那一行在任何按 rank 分叉之前），
所以天然一致，**不需要任何额外通信**。

## ⭐ ref_latents 也要跟着换

上游在 `r == 0` 时会把 `ref_latents` 替换成**生成出来的那一帧**当 attention
sink。换脸时如果只换像素、不换这个 sink，新脸会一点点飘回旧样子 ——
**而且不报错**。所以补丁里把 `ref_latents` 一起重算。

（Chris 2026-09-18：「换了新照片，那当然参照也要按照新照片走啊。」）

用法：
    python3 scripts/patch-upstream-faceswap.py [--file <上游 pipeline 路径>] [--revert]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_FILE = (pathlib.Path.home() / "LiveAvatar" / "liveavatar" / "models"
                / "wan" / "causal_s2v_pipeline_tpp_blockwise.py")

BEGIN = "# ─── CloseCrab 换脸补丁 begin（scripts/patch-upstream-faceswap.py）───"
END = "# ─── CloseCrab 换脸补丁 end ───"

HELPER = f'''

{BEGIN}
import os as _cc_os

_CC_FACE_FILE = _cc_os.environ.get("CCA_FACE_FILE", "/tmp/cca-current-face.txt")


def _cc_read_face():
    """换脸文件里写的是哪张图；没有 / 读不到就返回 None。

    **绝不抛异常** —— 这个函数在生成循环里每一轮都会被五个 rank 调到，
    抛出去会把整条流水线带走，而它本身只是个可选功能。
    """
    try:
        with open(_CC_FACE_FILE, encoding="utf-8") as f:
            p = f.read().strip()
        return p if p and _cc_os.path.exists(p) else None
    except Exception:
        return None
{END}
'''

ANCHOR = "            for r in range(active_nr):\n"

BODY = f'''                {BEGIN}
                # 图换了就就地重算参考图那四个张量，然后照常往下跑。
                # ⭐ ref_latents 也要重算：上游在 r==0 会把它换成生成出来的
                #    那一帧当 attention sink，不一起换的话新脸会慢慢飘回旧样子
                #    —— 而且不报错。
                _cc_new_face = _cc_read_face()
                if _cc_new_face and _cc_new_face != ref_image_path:
                    ref_image_path = _cc_new_face
                    ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
                    model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))
                    ref_pixel_values = tensor_trans(model_pic)
                    ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
                    ref_pixel_values = ref_pixel_values.to(
                        dtype=self.vae.dtype, device=self.vae.device)
                    ref_pixel_values = ref_pixel_values.repeat(1, 1, 5, 1, 1)
                    ref_latents = torch.stack(self.vae.encode(ref_pixel_values))[:, :, 1:]
                    motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
                    videos_last_frames = motion_latents.detach()
                    motion_latents = torch.stack(self.vae.encode(motion_latents))
                    # ⭐⭐ **注意力缓存也必须清掉，否则新旧两个人会混在一起。**
                    #
                    # 上游 `self.kv_cache1 = None` 只在**进循环之前**执行一次，
                    # 循环里是 `if self.kv_cache1 is None:` 才建 —— 也就是说
                    # KV cache 和 crossattn cache **只在第一轮建，之后一直沿用**。
                    #
                    # 只换参考图的话，缓存里还装着上一个人的注意力状态，模型
                    # 会把两个人**融**起来。2026-09-18 实测：从兔子换成一张
                    # 自拍，生成出来是那个人举着手，**手上长着兔耳朵**。
                    #
                    # 置 None 之后下一轮会连 crossattn cache 一起重建
                    #（两者由同一个 if 管），新身份就干净了。
                    #
                    # 代价：时序上下文断一次，画面会有一个跳切。换人本来就该跳切。
                    #
                    # 五个 rank 走的是同一行，所以不会失步（DiT rank 真的重建，
                    # VAE rank 上 kv_cache1 本来就恒为 None，行为不变）。
                    self.kv_cache1 = None
                    print(f"[CC] 换脸生效（含清缓存）：{{ref_image_path}}", flush=True)
                {END}
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    p = pathlib.Path(a.file)
    if not p.exists():
        print(f"✗ 找不到上游文件：{p}", file=sys.stderr)
        return 1
    s = p.read_text(encoding="utf-8")

    if a.revert:
        if BEGIN not in s:
            print("没打过补丁，无需回退")
            return 0
        out, keep = [], True
        for ln in s.splitlines(keepends=True):
            if BEGIN in ln:
                keep = False
            if keep:
                out.append(ln)
            if END in ln:
                keep = True
        p.write_text("".join(out), encoding="utf-8")
        print("✅ 已回退")
        return 0

    if BEGIN in s:
        print("✅ 已经打过了（幂等，什么都没做）")
        return 0

    # ⚠️ 锚点必须唯一。不唯一就停手 —— 今天已经因为「replace 打中了第一处」
    #    把一个早退插进了错误的函数，整个服务起不来。
    if s.count(ANCHOR) != 1:
        print(f"✗ 锚点出现 {s.count(ANCHOR)} 次，不是 1 次 —— 上游可能变了，停手",
              file=sys.stderr)
        return 1

    bak = p.with_suffix(p.suffix + ".bak-cc-faceswap")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")

    s = s.replace(ANCHOR, ANCHOR + BODY, 1) + HELPER
    p.write_text(s, encoding="utf-8")

    import ast
    try:
        ast.parse(s)
    except SyntaxError as e:
        p.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✗ 打完语法不过（{e}），已自动回退", file=sys.stderr)
        return 1

    print(f"✅ 已打补丁（备份 {bak.name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
