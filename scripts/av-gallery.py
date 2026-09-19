#!/usr/bin/env python3
"""把一批端到端跑出来的录制，做成**一条一条往上加**的画廊页。

> Chris 2026-09-19：「测一个加一条，测一个加一条，让我能看见他们。」

所以它是**从清单渲染**的，不是手写 HTML：加一条测试 = 往清单 JSON 里
追一个条目再跑一次这个脚本。已经合过的不重复合（看 mp4 在不在），
所以重跑很便宜。

## 这里的片子**一毫秒都不补**

跟 `lipsync-preview.py` 正相反 —— 那个是为了让人能专心看嘴型，
先把偏移补掉。这里要看的就是**偏移本身**，补了就什么都看不见了。

每条下面写三个数，都是量出来的：

    每句「音频起 → 嘴起」   正数 = 嘴慢，负数 = 嘴快
    两遍之间的差            同一形状跑两遍，差多少
    帧数 / 时长             对不上就是漏帧

## 清单格式

```json
[{"tag": "s01-single-8s", "note": "单句 8 秒", "pass": 1,
  "dump": "/tmp/cca-dump/171203", "leadin": 2.0,
  "marks": {"A": 0.0}}]
```

`marks` 是每句在**真音频内部**的起点秒数（不含 leadin）。

用法：`scripts/av-gallery.py --manifest m.json --out /tmp/g --prefix gal-20260919`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from worker.mouth_metric import mouth_openness  # noqa: E402


def measure(dump: pathlib.Path, leadin: float, marks: dict[str, float]) -> dict:
    m = json.loads((dump / "meta.json").read_text(encoding="utf-8"))
    w, h, fps, rate = m["width"], m["height"], m["fps"], m["sample_rate"]
    a = np.frombuffer((dump / "audio.pcm").read_bytes(), np.int16)
    op = mouth_openness(dump / "video.rgba", w, h)

    hop = max(1, rate // 100)                       # 10 ms
    pk = np.array([np.abs(a[i:i + hop]).max() for i in range(0, len(a) - hop, hop)],
                  dtype=float)
    base = op[:int(fps * 1.5)]
    med, noise = float(np.median(base)), float(base.std())
    dev = np.abs(op - med)

    out = {}
    for tag, off in sorted(marks.items(), key=lambda kv: kv[1]):
        at = leadin + off
        i0 = max(0, int((at - 0.3) * 100))
        seg = pk[i0:i0 + 80]
        if not len(seg) or pk.max() <= 0:
            continue
        a_on = (i0 + int(np.argmax(seg > pk.max() * 0.06))) / 100
        f0 = int(a_on * fps)
        # 连续 3 帧超过基线 5σ 才算「嘴动了」—— 单帧会被压缩噪声骗到
        on = next((i for i in range(f0, min(len(op) - 3, f0 + int(fps * 4)))
                   if (dev[i:i + 3] > noise * 5).all()), None)
        out[tag] = None if on is None else round((on / fps - a_on) * 1000)
    return {"lags": out, "frames": m["video_frames"], "fps": fps,
            "video_s": round(m["video_frames"] / fps, 2),
            "audio_s": round((dump / "audio.pcm").stat().st_size
                             / (rate * m["channels"] * 2), 2)}


def mux(dump: pathlib.Path, dst: pathlib.Path, head: float) -> None:
    """**零补偿**合成：音视频切同一段时间，原样焊死。"""
    m = json.loads((dump / "meta.json").read_text(encoding="utf-8"))
    w, h, fps = m["width"], m["height"], m["fps"]
    rate, ch = m["sample_rate"], m["channels"]
    v_off = int(round(head * fps)) * w * h * 4
    a_off = int(round(head * rate)) * ch * 2
    tmp = dst.with_suffix(".pcm.tmp")
    with open(dump / "audio.pcm", "rb") as src, open(tmp, "wb") as g:
        src.seek(a_off)
        while chunk := src.read(1 << 20):
            g.write(chunk)
    vf = open(dump / "video.rgba", "rb"); vf.seek(v_off)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "pipe:0",
             "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", str(tmp),
             "-map", "0:v", "-map", "1:a", "-shortest",
             "-c:v", "libx264", "-preset", "medium", "-crf", "22",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(dst)],
            stdin=vf, check=True)
    finally:
        vf.close(); tmp.unlink(missing_ok=True)


CARD = """<div class="card">
<h2>__TAG__ <span class="pass">第 __PASS__ 遍</span></h2>
<p class="meta">__NOTE__ ｜ 画面 __VS__ s / __FR__ 帧 ｜ 声音 __AS__ s</p>
<p class="lags">__LAGS__</p>
<video controls playsinline preload="none" poster="" src="__SLUG__.mp4"></video>
</div>"""

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>%F0%9F%A6%80</text></svg>">
<style>
:root{--bg:#f5f5f7;--card:#fff;--ink:#1c1b1f;--mute:#5f6368;--line:#e3e3e6;
      --accent:#1a73e8;--warn:#b3261e;--ok:#146c2e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 -apple-system,"Noto Sans CJK SC","PingFang SC",system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:22px 16px 64px}
h1{font-size:21px;font-weight:600;margin:0 0 4px}
.sub{color:var(--mute);font-size:14px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:16px;margin-bottom:16px}
h2{font-size:17px;margin:0 0 2px}
.pass{font-size:12px;font-weight:400;color:var(--mute);border:1px solid var(--line);
      border-radius:999px;padding:1px 8px;margin-left:6px}
.meta{color:var(--mute);font-size:13px;margin:0 0 6px}
.lags{font-size:14px;margin:0 0 10px;font-variant-numeric:tabular-nums}
.lags b{font-weight:600}.big{color:var(--warn)}.small{color:var(--ok)}
video{width:100%;max-height:70vh;background:#000;border-radius:8px;display:block}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:6px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mute);font-weight:500;font-size:13px}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.verdict{border-left:3px solid var(--accent);padding:10px 14px;background:#f8f9fe;
         border-radius:0 8px 8px 0;font-size:14px}
.verdict b{color:var(--accent)}
</style><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">每条都是**零补偿**原样合成的 —— 要看的就是偏移本身，补了就看不见了。
共 __N__ 条，每种形状跑两遍。</p>
<div class="card"><div class="verdict">
<b>怎么读。</b>每句下面那个数是「音频第一次出声 → 嘴第一次动」，
正数表示嘴慢、负数表示嘴快。一帧 = 40 ms，所以<b>绝对值小于 40 就是对齐的</b>。<br>
同一形状的两遍放在一起看：<b>两遍差得多，就说明这个偏移不是固定值</b> ——
那正是不能用一个常数去补偿的理由。
</div></div>
<div class="card"><h2>总表</h2><table>__TABLE__</table></div>
__CARDS__
</div></html>"""


def fmt_lags(lags: dict[str, int | None]) -> str:
    if not lags:
        return '<span class="mute">没量到</span>'
    bits = []
    for k, v in lags.items():
        if v is None:
            bits.append(f"{k} <b>?</b>")
        else:
            cls = "small" if abs(v) < 40 else "big"
            bits.append(f'{k} <b class="{cls}">{v:+d} ms</b>')
    return " ｜ ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--title", default="数字人音画配对 · 实测画廊")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))

    cards, rows = [], ["<tr><th>形状</th><th>遍</th><th>每句偏移</th>"
                       "<th class='n'>帧</th></tr>"]
    for e in entries:
        slug = f"{a.prefix}-{e['tag']}-p{e['pass']}"
        mp4 = out / f"{slug}.mp4"
        dump = pathlib.Path(e["dump"])
        if not dump.exists():
            print(f"  跳过 {slug}：{dump} 不在")
            continue
        r = measure(dump, e.get("leadin", 2.0), e.get("marks", {}))
        if not mp4.exists():                       # 合过的不重合，重跑很便宜
            mux(dump, mp4, e.get("leadin", 2.0))
        print(f"  {slug}: {r['lags']}")
        cards.append(CARD.replace("__TAG__", e["tag"]).replace("__PASS__", str(e["pass"]))
                     .replace("__NOTE__", e.get("note", "")).replace("__SLUG__", slug)
                     .replace("__VS__", str(r["video_s"])).replace("__AS__", str(r["audio_s"]))
                     .replace("__FR__", str(r["frames"]))
                     .replace("__LAGS__", fmt_lags(r["lags"])))
        rows.append(f"<tr><td>{e['tag']}</td><td>{e['pass']}</td>"
                    f"<td>{fmt_lags(r['lags'])}</td>"
                    f"<td class='n'>{r['frames']}</td></tr>")

    page = (PAGE.replace("__TITLE__", a.title).replace("__N__", str(len(cards)))
            .replace("__TABLE__", "\n".join(rows)).replace("__CARDS__", "\n".join(cards)))
    page = page.replace("**零补偿**", "<b>零补偿</b>").replace(
        "**两遍差得多，就说明这个偏移不是固定值**",
        "<b>两遍差得多，就说明这个偏移不是固定值</b>")
    (out / f"{a.prefix}.html").write_text(page, encoding="utf-8")
    print(f"✅ {out}/{a.prefix}.html（{len(cards)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
