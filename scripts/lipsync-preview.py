#!/usr/bin/env python3
"""把录制合成**音画已经对好**的成品 mp4，外加一页直接能看的预览。

## 跟另外两个脚本的分工

    mux-av-dump.sh     原样合一个 mp4（不补偿）—— 看「生成得好不好」
    av-sync-page.py    音视频分开 + 滑块      —— 量「差多少毫秒」
    这个脚本           量完自动补偿再焊死      —— 只看「嘴型对不对」

> Chris 2026-09-19：「你这个弄得太复杂了，别让我去调那个前后的，
> 我也调不明白。就简简单单的：拿一段字生成音频，拿音频去生成视频，
> 把音频视频同步地合到一起给我看。」

## ⚠️ 「理应同步」在这条管线上不成立，所以必须补偿

源头就不同步：模型要攒够一段音频才画得出对应的嘴（光听一刹那分不出
「啊」还是「哈」），所以画面天生落后于声音。2026-09-19 实测**约 1.85 s，
六种条件下都是这个数**——常数，不随采样率/语速/时长/前导变。

于是「直接把两条流按零点对齐合起来」= 交付一个必然错位的东西。
这里改成：先量出滞后 L，再按 L 补偿，然后才焊死。

补偿的方式是**掐视频的头**，不是给音频垫前缀：

    输出时刻 τ  ←  视频取 v(τ + L + P)，音频取 a(τ + P)

    v(t) 画的是 a(t − L) 的嘴  ⇒  v(τ+L+P) 画的正是 a(τ+P)  ⇒  对上了

`P` 是探针为了等接收端挂上垫的静音前导，两条流一起掐掉，不然开头要干等。
垫音频前缀也能对上，但会把那段空白留在成品里 —— 他已经说过不想要。

**代价写在页面上**：末尾最后 L 秒的话没有对应画面（管线还没画到就结束了）。
所以送进来的音频末尾应该自带几秒静音，让它把最后几个字画完。

## 取样框必须可查，不能我说了算

量滞后要先框出嘴。2026-09-19 我**两次**框错 —— 一次框到手上，一次框到
脖子和衣领上，两批数字全废，而且失败时数字看起来完全正常（只是相关低）。
所以这个脚本**把画了框的那一帧存成图并链在页面上**：框对不对，看图的人
自己判断，不用信我。

用法：

    scripts/lipsync-preview.py --out /tmp/prev --prefix clips-20260919 \\
        --clip "短|7 秒|/tmp/cca-dump/041234|2.0"

`--clip` 四段：标签｜说明｜录制目录｜静音前导秒数。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

# 嘴的位置（占画幅比例）。这套数是 384×704 竖版头肩像上量准的；换形象要重调，
# **而且必须看一眼 `-box.jpg` 再采信数字**。
MOUTH = dict(y0=0.38, y1=0.51, x0=0.38, x1=0.66)
LAG_SEARCH_S = (-0.5, 3.5)          # 滞后搜索范围
MIN_CORR = 0.18                     # 低于这个就别声称量准了


def mouth_openness(vraw: pathlib.Path, w: int, h: int) -> np.ndarray:
    """逐帧的「张嘴程度」。

    取嘴部方框里的**暗像素占比** —— 口腔内部比嘴唇和皮肤暗得多，
    张嘴这个动作在灰度上最稳的表现就是暗区变大。比帧间差分好：
    差分会把眨眼、头微动一起算进来，而那些跟说话没关系。
    """
    y0, y1 = int(h * MOUTH["y0"]), int(h * MOUTH["y1"])
    x0, x1 = int(w * MOUTH["x0"]), int(w * MOUTH["x1"])
    fsz, rowsz = w * h * 4, w * 4
    out = []
    with open(vraw, "rb") as f:
        n = vraw.stat().st_size // fsz
        for i in range(n):
            f.seek(i * fsz + y0 * rowsz)          # 只读嘴那几行，别整帧搬
            band = np.frombuffer(f.read((y1 - y0) * rowsz), np.uint8)
            band = band.reshape(y1 - y0, w, 4)[:, x0:x1, :3]
            gray = band.mean(axis=2)
            out.append(gray)
    g = np.stack(out)
    thr = np.percentile(g, 12)                    # 全片统一阈值，别逐帧自适应
    return (g < thr).mean(axis=(1, 2))


def audio_envelope(apcm: pathlib.Path, rate: int, ch: int, n: int) -> np.ndarray:
    """音频响度包络，重采成每个视频帧一个点。"""
    a = np.frombuffer(apcm.read_bytes(), np.int16)
    if ch > 1:
        a = a[::ch]
    idx = np.linspace(0, len(a), n + 1).astype(int)
    return np.array([np.abs(a[idx[i]:idx[i + 1]]).max() if idx[i + 1] > idx[i] else 0
                     for i in range(n)], dtype=float)


def measure_lag(openness: np.ndarray, env: np.ndarray, fps: int) -> tuple[float, float]:
    """互相关求「画面比声音晚多少」。返回（秒, 该处相关）。"""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)          # noqa: E731
    m, a = z(openness), z(env[:len(openness)])
    best, br = 0, -2.0
    for l in range(int(LAG_SEARCH_S[0] * fps), int(LAG_SEARCH_S[1] * fps) + 1):
        x, y = (m[l:], a[:len(a) - l]) if l >= 0 else (m[:l], a[-l:])
        if len(x) < fps:                                     # 样本太少不算
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r == r and r > br:
            br, best = r, l
    return best / fps, br


def save_box_image(vraw: pathlib.Path, w: int, h: int, frame: int, dst: pathlib.Path) -> None:
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


def measure(tag: str, note: str, d: pathlib.Path, leadin: float,
            out: pathlib.Path, prefix: str) -> dict:
    """第一遍：只量，不合。

    分两遍是有原因的：滞后是**整条管线的常数**，可某一条的相关可能很低
    （画面里头动得多、说话少的段落都会拉低它）。拿一个不可信的补偿值去合，
    等于让「这一条看着不同步」变成素材本身的问题 —— 而看的人分不出那是
    模型不行还是我补偿补歪了。所以先全量测完，不可信的借用可信条目的中位数。
    """
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    w, h, fps = m["width"], m["height"], m["fps"]
    rate, ch = m["sample_rate"], m["channels"]
    vraw, apcm = d / "video.rgba", d / "audio.pcm"
    slug = f"{prefix}-{tag}"
    op = mouth_openness(vraw, w, h)
    env = audio_envelope(apcm, rate, ch, len(op))
    lag, corr = measure_lag(op, env, fps)
    save_box_image(vraw, w, h, int(len(op) * 0.6), out / f"{slug}-box.jpg")
    return dict(tag=tag, note=note, dir=d, leadin=leadin, slug=slug,
                w=w, h=h, fps=fps, rate=rate, ch=ch, meta=m,
                lag=lag, corr=corr, weak=corr < MIN_CORR)


def build(c: dict, lag: float, out: pathlib.Path) -> dict:
    """第二遍：按给定的滞后补偿并焊死。"""
    tag, d, leadin, slug = c["tag"], c["dir"], c["leadin"], c["slug"]
    w, h, fps, rate, ch, m = c["w"], c["h"], c["fps"], c["rate"], c["ch"], c["meta"]
    vraw, apcm = d / "video.rgba", d / "audio.pcm"

    # 掐头的量必须同时落在整帧和整采样上，否则自己造一个亚帧偏移出来。
    v_skip = int(round((lag + leadin) * fps))
    a_skip = int(round(leadin * rate))
    v_off, a_off = v_skip * w * h * 4, a_skip * ch * 2

    vf = open(vraw, "rb"); vf.seek(v_off)
    af = open(apcm, "rb"); af.seek(a_off)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "pipe:0",
             "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", f"/dev/fd/{af.fileno()}",
             "-map", "0:v", "-map", "1:a", "-shortest",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
             str(out / f"{slug}.mp4")],
            stdin=vf, check=True, pass_fds=(af.fileno(),))
    finally:
        vf.close(); af.close()

    return dict(c, lag_ms=round(lag * 1000), corr=round(c["corr"], 2),
                video_s=round((m["video_frames"] - v_skip) / fps, 2),
                audio_s=round((apcm.stat().st_size - a_off) / (rate * ch * 2), 2))


HTML = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>%F0%9F%A6%80</text></svg>">
<style>
 :root{--bg:#f5f5f7;--card:#fff;--ink:#1c1b1f;--mute:#5f6368;--line:#e3e3e6;--accent:#1a73e8;--warn:#b3261e}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.65 -apple-system,"Noto Sans CJK SC","PingFang SC",system-ui,sans-serif}
 .wrap{max-width:880px;margin:0 auto;padding:22px 16px 64px}
 h1{font-size:21px;font-weight:600;margin:0 0 4px}
 .sub{color:var(--mute);font-size:14px;margin:0 0 18px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
 h2{font-size:17px;margin:0 0 2px}
 .meta{color:var(--mute);font-size:13px;margin:0 0 10px}
 video{width:100%;max-height:70vh;background:#000;border-radius:8px;display:block}
 .warn{color:var(--warn)}
 details{margin-top:10px}
 summary{cursor:pointer;color:var(--accent);font-size:13px}
 details img{max-width:220px;border-radius:6px;margin-top:8px;display:block}
 .verdict{border-left:3px solid var(--accent);padding:10px 14px;background:#f8f9fe;
          border-radius:0 8px 8px 0;font-size:14px}
 .verdict b{color:var(--accent)}
</style>
<div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">三段长短不同的话，音画<b>已经对好</b>，直接播就行 —— 只看嘴型准不准。</p>
<div class="card"><div class="verdict">
<b>为什么需要「对好」这一步。</b>这条管线在源头就不同步：模型要攒够一段声音
才画得出对应的嘴（光听一刹那分不出「啊」还是「哈」），所以画面天生落后。
实测这个滞后约 <b>1.85 秒</b>，而且六种条件下都一样 —— 是个常数。
下面每段都按各自量到的值补偿过，补了多少写在标题下面。
<b>那个延迟本身是要单独修的 bug，这一页只回答「嘴型对不对」。</b><br>
代价：每段末尾最后一点话没有画面（管线还没画到就结束了），所以录的时候
音频尾巴留了 3 秒静音。
</div></div>
__CLIPS__
</div></html>
"""

CLIP = """<div class="card">
<h2>__TAG__ · __NOTE__</h2>
<p class="meta">补偿 __LAG__ ms（相关 __CORR__）__WEAK__ ｜ 画面 __VS__ s ｜ 声音 __AS__ s ｜ __RATE__ Hz</p>
<video controls playsinline preload="metadata" src="__SLUG__.mp4"></video>
<details><summary>量嘴用的取样框长什么样（点开自己看对不对）</summary>
<img src="__SLUG__-box.jpg" alt="取样框"></details>
</div>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True,
                    help="「标签|说明|录制目录|静音前导秒数」，可重复")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--title", default="数字人口型预览")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    measured = []
    for spec in a.clip:
        parts = [x.strip() for x in spec.split("|")]
        if len(parts) != 4:
            raise SystemExit(f"✗ --clip 要四段「标签|说明|目录|前导秒数」：{spec}")
        mm = measure(parts[0], parts[1], pathlib.Path(parts[2]), float(parts[3]),
                     out, a.prefix)
        print(f"   量：{mm['tag']}  滞后 {mm['lag']*1000:.0f} ms  相关 {mm['corr']:.2f}"
              + ("  ⚠️ 偏低，改用可信条目的中位数" if mm["weak"] else ""))
        measured.append(mm)

    good = [m["lag"] for m in measured if not m["weak"]]
    if not good:
        raise SystemExit("✗ 没有一条量得可信 —— 先去 -box.jpg 看取样框是不是框错了")
    fallback = float(np.median(good))

    cards = []
    for mm in measured:
        lag = fallback if mm["weak"] else mm["lag"]
        c = build(mm, lag, out)
        flag = (f"　⚠️ 这条相关只有 {c['corr']}，量不准，用的是另外两条的中位数 "
                f"{fallback*1000:.0f} ms" if mm["weak"] else "")
        print(f"── {c['tag']}  补偿 {lag*1000:.0f} ms")
        cards.append(CLIP
                     .replace("__TAG__", c["tag"]).replace("__NOTE__", c["note"])
                     .replace("__LAG__", f"{lag*1000:.0f}").replace("__CORR__", str(c["corr"]))
                     .replace("__WEAK__", f'<span class="warn">{flag}</span>')
                     .replace("__VS__", str(c["video_s"])).replace("__AS__", str(c["audio_s"]))
                     .replace("__RATE__", str(c["rate"])).replace("__SLUG__", c["slug"]))

    html = HTML.replace("__TITLE__", a.title).replace("__CLIPS__", "\n".join(cards))
    for must in ("<video controls", "-box.jpg", "补偿"):
        if must not in html:
            raise SystemExit(f"✗ 生成的页面里没有 {must}，模板八成被改坏了")
    (out / f"{a.prefix}.html").write_text(html, encoding="utf-8")
    print(f"✅ {out}/{a.prefix}.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
