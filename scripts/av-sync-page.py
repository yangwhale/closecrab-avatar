#!/usr/bin/env python3
"""把一场录制做成**可以拖偏移的**音画同步检查页。

## 跟 `mux-av-dump.sh` 的分工

    mux-av-dump.sh   合成一个 mp4 —— 回答「生成得好不好」
    这个脚本         音视频分开 + 一个滑块 —— 回答「差多少毫秒」

差别不是花哨程度，是**能不能量**。焊死在一个 mp4 里的音画只能让人说
「感觉不太对」；分开之后拖到对上为止，读数就是答案，而且是被试者自己
给出的，不掺我的判断。

> Chris 2026-09-19：「我觉得可能还差个 0.25 秒左右，就是音画不同步的原因，
> 我猜的。你给我做一个 HTML，把合成的音频和视频同步的给我看一下。」

## 这个数出来之后怎么用（页面上也写了）

录的是**递交给 LiveKit 之前**的东西，所以：

    页面上要拖才对得上  →  服务端这一半就错了，去查生成/配对
    页面上 0 就是对的   →  服务端没问题，错在传输或客户端

**两种结论指向完全不同的代码**，所以这一步不能跳过直接去调模型。

用法：

    scripts/av-sync-page.py <录制目录> --out <输出目录> [--title 标题]

产物：`video.mp4`（无声）、`audio.m4a`、`index.html`（自包含，只依赖同目录那两个文件）。
"""
from __future__ import annotations

import argparse
import array
import json
import pathlib
import shutil
import subprocess
import sys

# 波形取样点数。太多画不动、太少看不出起伏；1200 在手机上也够细。
ENVELOPE_POINTS = 1200


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def envelope(pcm: pathlib.Path, *, rate: int, channels: int, points: int) -> list[float]:
    """把 PCM 压成一条 0..1 的响度包络。

    用**每段绝对值的最大值**而不是 RMS：判断「嘴该不该张」看的是有没有
    冲击，RMS 会把清辅音那种短促高频抹平，而那恰恰是口型最明显的时刻。
    """
    raw = pcm.read_bytes()
    n = len(raw) // 2
    if n == 0:
        return []
    samples = array.array("h")
    samples.frombytes(raw[: n * 2])
    if channels > 1:                      # 只取第一声道，够用
        samples = samples[::channels]
        n = len(samples)
    step = max(1, n // points)
    out: list[float] = []
    for i in range(0, n - step + 1, step):
        peak = 0
        for v in samples[i:i + step]:
            av = -v if v < 0 else v
            if av > peak:
                peak = av
        out.append(round(peak / 32768.0, 4))
    return out


HTML = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>%F0%9F%A6%80</text></svg>">
<style>
  :root {
    --bg:#f5f5f7; --card:#fff; --ink:#1c1b1f; --mute:#5f6368;
    --line:#e3e3e6; --accent:#1a73e8; --warn:#b3261e; --ok:#146c2e;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.65 -apple-system,"Noto Sans CJK SC","PingFang SC",system-ui,sans-serif; }
  .wrap { max-width:880px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--mute); font-size:14px; margin:0 0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px; margin-bottom:16px; }
  video { width:100%; max-height:62vh; background:#000; border-radius:8px; display:block; }
  .readout { font-size:34px; font-weight:600; font-variant-numeric:tabular-nums;
             text-align:center; margin:4px 0 2px; }
  .readout small { font-size:14px; font-weight:400; color:var(--mute); }
  input[type=range] { width:100%; accent-color:var(--accent); height:28px; }
  .ticks { display:flex; justify-content:space-between; color:var(--mute);
           font-size:12px; font-variant-numeric:tabular-nums; margin-top:-4px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:12px; }
  button { font:inherit; padding:8px 14px; border-radius:999px; cursor:pointer;
           border:1px solid var(--line); background:#fff; color:var(--ink); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:active { transform:translateY(1px); }
  canvas { width:100%; height:84px; display:block; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  td { padding:5px 0; border-bottom:1px solid var(--line); }
  td:first-child { color:var(--mute); width:44%; }
  td:last-child { font-variant-numeric:tabular-nums; }
  .verdict { border-left:3px solid var(--accent); padding:10px 14px; background:#f8f9fe;
             border-radius:0 8px 8px 0; font-size:14px; }
  .verdict b { color:var(--accent); }
  kbd { background:#eceff1; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:1px 6px; font-size:12px; font-family:inherit; }
</style>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">录制点在<b>递交给 LiveKit 之前</b> —— 这里看到的是服务端这一半的原貌，不含传输和手机端。</p>

  <div class="card">
    <video id="v" playsinline muted preload="auto" src="video.mp4"></video>
    <audio id="a" preload="auto" src="audio.m4a"></audio>
    <canvas id="wave" height="84"></canvas>
  </div>

  <div class="card">
    <div class="readout"><span id="val">0</span> ms<br><small id="hint">音频与视频同步</small></div>
    <input type="range" id="off" min="-600" max="600" step="10" value="0">
    <div class="ticks"><span>−600（声音提前）</span><span>0</span><span>+600（声音延后）</span></div>
    <div class="row">
      <button id="play" class="primary">播放</button>
      <button data-d="-10">−10 ms</button>
      <button data-d="10">+10 ms</button>
      <button id="zero">归零</button>
      <button id="back">回到开头</button>
    </div>
    <div class="row">
      <button data-f="-1">◀ 上一帧</button>
      <button data-f="1">下一帧 ▶</button>
      <span style="align-self:center;color:var(--mute);font-size:13px">
        一帧 = <span id="fms"></span> ms ｜ <kbd>←</kbd><kbd>→</kbd> 逐帧 <kbd>空格</kbd> 播放
      </span>
    </div>
  </div>

  <div class="card">
    <div class="verdict">
      <b>怎么读这个数。</b>拖到嘴型和声音对上为止，那个读数就是服务端这一侧的音画偏移。<br>
      · 需要拖才对得上 → <b>服务端就错了</b>，去查生成和音视频配对。<br>
      · 0 就是对的、但手机上仍然不对 → <b>服务端没问题</b>，错在 LiveKit 传输或 App。<br>
      两种结论指向完全不同的代码，所以先量这一步。
    </div>
  </div>

  <div class="card">
    <table id="meta"></table>
  </div>
</div>
<script>
const META = __META__;
const ENV  = __ENV__;

const v = document.getElementById('v'), a = document.getElementById('a');
const slider = document.getElementById('off'), val = document.getElementById('val');
const hint = document.getElementById('hint'), playBtn = document.getElementById('play');
const cv = document.getElementById('wave'), ctx = cv.getContext('2d');
const frameMs = 1000 / META.fps;
document.getElementById('fms').textContent = frameMs.toFixed(0);

let offset = 0;   // 毫秒。正 = 声音相对画面**延后**播。

// ⚠️ 视频是时钟，音频跟着它对。反过来（音频当时钟）在 seek 时会打架 ——
//    video 的 seek 是关键帧对齐的，audio 不是，互相追会来回抖。
function targetAudioTime() { return v.currentTime - offset / 1000; }

function syncAudio(force) {
  const t = targetAudioTime();
  if (t < 0 || t > (a.duration || 1e9)) { if (!a.paused) a.pause(); return; }
  if (force || Math.abs(a.currentTime - t) > 0.035) a.currentTime = t;
  if (!v.paused && a.paused) a.play().catch(() => {});
}

slider.addEventListener('input', () => {
  offset = +slider.value;
  val.textContent = (offset > 0 ? '+' : '') + offset;
  hint.textContent = offset === 0 ? '音频与视频同步'
    : offset > 0 ? '声音比画面晚 ' + offset + ' ms' : '声音比画面早 ' + (-offset) + ' ms';
  syncAudio(true);
});
document.querySelectorAll('[data-d]').forEach(b => b.onclick = () => {
  slider.value = Math.max(-600, Math.min(600, +slider.value + (+b.dataset.d)));
  slider.dispatchEvent(new Event('input'));
});
document.getElementById('zero').onclick = () => { slider.value = 0; slider.dispatchEvent(new Event('input')); };
document.getElementById('back').onclick = () => { v.currentTime = 0; syncAudio(true); };

function toggle() {
  if (v.paused) { v.play(); syncAudio(true); a.play().catch(() => {}); playBtn.textContent = '暂停'; }
  else { v.pause(); a.pause(); playBtn.textContent = '播放'; }
}
playBtn.onclick = toggle;
v.addEventListener('pause', () => { a.pause(); playBtn.textContent = '播放'; });

function step(n) {
  v.pause(); a.pause(); playBtn.textContent = '播放';
  v.currentTime = Math.max(0, v.currentTime + n * frameMs / 1000);
  setTimeout(() => syncAudio(true), 30);
}
document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => step(+b.dataset.f));
addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
  if (e.key === ' ') { toggle(); e.preventDefault(); }
});

// 每帧校一次音频，别用 timeupdate —— 它大约 4 Hz，肉眼能看出漂。
(function loop() { if (!v.paused) syncAudio(false); requestAnimationFrame(loop); })();

function drawWave() {
  const w = cv.clientWidth, h = cv.height, dpr = devicePixelRatio || 1;
  cv.width = w * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#dbe4f3';
  const n = ENV.length;
  for (let i = 0; i < n; i++) {
    const x = i / n * w, bh = Math.max(1, ENV[i] * (h - 22));
    ctx.fillRect(x, (h - 18 - bh) , Math.max(1, w / n), bh);
  }
  // 播放头按**音频自己的时间轴**画 —— 拖偏移时它会相对波形移动，
  // 这正是我们想让人看见的东西。
  const dur = META.audio_seconds || 1;
  const x = Math.max(0, Math.min(1, targetAudioTime() / dur)) * w;
  ctx.fillStyle = '#b3261e';
  ctx.fillRect(x - 1, 0, 2, h - 14);
  ctx.fillStyle = '#5f6368'; ctx.font = '11px system-ui';
  ctx.fillText('音频包络（红线 = 当前听到的位置）', 4, h - 3);
  requestAnimationFrame(drawWave);
}
drawWave();

const rows = [
  ['分辨率', META.width + ' × ' + META.height],
  ['帧率', META.fps + ' fps（一帧 ' + frameMs.toFixed(1) + ' ms）'],
  ['视频时长', META.video_seconds + ' s（' + META.video_frames + ' 帧）'],
  ['音频时长', (META.audio_seconds || 0).toFixed(2) + ' s'],
  ['两者之差', Math.round(((META.video_seconds || 0) - (META.audio_seconds || 0)) * 1000) + ' ms'],
  ['音频格式', META.sample_rate + ' Hz / ' + META.channels + ' 声道'],
  ['录制场次', META.session || '—'],
];
document.getElementById('meta').innerHTML =
  rows.map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('');
</script>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="数字人音画同步检查")
    args = ap.parse_args()

    d = pathlib.Path(args.dump_dir)
    meta_p = d / "meta.json"
    if not meta_p.exists():
        print(f"✗ 没有 {meta_p} —— 这一场没正常收尾，参数不可信", file=sys.stderr)
        return 1
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    w, h, fps = meta["width"], meta["height"], meta["fps"]
    rate, ch = meta["sample_rate"], meta["channels"]
    apcm = d / "audio.pcm"
    meta["audio_seconds"] = apcm.stat().st_size / (rate * ch * 2)
    meta["session"] = d.name

    # 视频**不带声音**：偏移要在浏览器里调，焊在一个文件里就没得调了。
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", str(d / "video.rgba"), "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "video.mp4")])
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
         "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", str(apcm),
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out / "audio.m4a")])

    env = envelope(apcm, rate=rate, channels=ch, points=ENVELOPE_POINTS)
    (out / "index.html").write_text(
        HTML.replace("__TITLE__", args.title)
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__ENV__", json.dumps(env)),
        encoding="utf-8")

    for f in ("video.mp4", "audio.m4a", "index.html"):
        print(f"  {f}  {(out / f).stat().st_size / 1e6:.1f} MB")
    if not shutil.which("ffmpeg"):
        print("（本机没有 ffmpeg —— 上面那两步应该已经报错了）")
    print(f"✅ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
