#!/usr/bin/env python3
"""把若干场录制做成**可以逐条拖偏移的**音画同步检查页。

## 跟 `mux-av-dump.sh` 的分工

    mux-av-dump.sh   合成一个 mp4 —— 回答「生成得好不好」
    这个脚本         音视频分开 + 滑块 —— 回答「差多少毫秒」

差别不是花哨程度，是**能不能量**。焊死在一个 mp4 里的音画只能让人说
「感觉不太对」；分开之后拖到对上为止，读数就是答案，而且是被试者自己
给出的，不掺我的判断。

> Chris 2026-09-19：「我觉得可能还差个 0.25 秒左右……你给我做一个 HTML，
> 把合成的音频和视频同步的给我看一下。」
> 看完之后：「HTML 里边的嘴型也对不齐。你再给我多生成几个……我来听一听，
> 看有什么规律。」

## 为什么要支持多条，而不是发多个页面

**「有什么规律」这件事，只有把读数放在同一张表里才看得见。**
一条一条发过去，人得自己记住六个数再心算 —— 那一步一定会丢。
所以页面自己记（localStorage），底下那张表实时排出来：

    偏移随时长涨      ⇒ 比例误差（帧率 / 重采样比不对）
    偏移是个常数      ⇒ 固定延迟（对齐点差了几帧）
    只有某个采样率错  ⇒ 错在重采样那一段

## 录制点在哪，决定了结论指向哪

采集点在**递交给 LiveKit 之前**，所以这一页只能判服务端这一半：

    要拖才对得上  →  服务端就错了，去查生成/配对
    0 就是对的    →  服务端没问题，错在传输或客户端

用法：

    scripts/av-sync-page.py --out /tmp/lab --prefix lab-20260919 \\
        --clip "A|基准 48 kHz|/tmp/cca-dump/032238" \\
        --clip "B|同一句 16 kHz|/tmp/cca-dump/032410"

`--clip` 的三段是「短名｜这条改了什么｜录制目录」。
产物全部平铺在 `--out` 里，文件名带 `--prefix`，可以直接扔进同一个
CC Pages 目录而不打架。
"""
from __future__ import annotations

import argparse
import array
import json
import pathlib
import subprocess
import sys

ENVELOPE_POINTS = 900
"""波形取样点数。太多手机上画不动，太少看不出起伏。"""


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def envelope(pcm: pathlib.Path, *, rate: int, channels: int, points: int,
             skip_bytes: int = 0) -> list[float]:
    """把 PCM 压成一条 0..1 的响度包络。

    用**每段绝对值的最大值**而不是 RMS：判断「嘴该不该张」看的是有没有
    冲击，RMS 会把清辅音那种短促高频抹平，而那恰恰是口型最明显的时刻。
    """
    raw = pcm.read_bytes()[skip_bytes:]
    n = len(raw) // 2
    if n == 0:
        return []
    samples = array.array("h")
    samples.frombytes(raw[: n * 2])
    if channels > 1:
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


def _trimmed(path: pathlib.Path, skip_bytes: int):
    """从 `skip_bytes` 处开始把文件喂给 ffmpeg 的 stdin。

    **不用 ffmpeg 的 `-ss`** —— 裸流没有时间信息，`-ss` 只能按码率估位置，
    而这一页量的就是毫秒，估出来的头等于把待测量本身搞脏。按字节切是精确的：
    调用方保证 `skip_bytes` 落在整帧 / 整采样的边界上。
    """
    f = open(path, "rb")
    f.seek(skip_bytes)
    return f


def build_clip(tag: str, note: str, d: pathlib.Path, out: pathlib.Path,
               prefix: str, trim_head: float = 0.0) -> dict:
    meta_p = d / "meta.json"
    if not meta_p.exists():
        raise SystemExit(f"✗ 没有 {meta_p} —— 这一场没正常收尾，参数不可信")
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    w, h, fps = m["width"], m["height"], m["fps"]
    rate, ch = m["sample_rate"], m["channels"]
    apcm = d / "audio.pcm"
    vraw = d / "video.rgba"
    slug = f"{prefix}-{tag}"

    # ⭐ 掐头：**两条流必须掐掉完全相同的时长**，否则就是自己造一个偏移出来。
    #
    #    探针为了等接收端挂上，会在真音频前面垫几秒静音（`--leadin`）。
    #    那几秒原样进了录制：开头一段静音 + 一张不动的脸。
    #    Chris 2026-09-19：「视频老是先播半秒一秒才出声，感觉视频被拉长了，
    #    多出来的部分放到音频前面。」—— 那就是这段前导，不是管线的毛病。
    #    但它让判口型变难（要先干等），所以做成页面之前掐掉。
    #
    #    帧数和采样数都取整：25 fps 与 16k/48k 下，0.1 s 的整数倍一定同时落在
    #    整帧和整采样上。不取整的话两条流会差出零点几帧 —— 而那正是被测量。
    nv_skip = int(round(trim_head * fps))
    na_skip = int(round(trim_head * rate))
    v_off, a_off = nv_skip * w * h * 4, na_skip * ch * 2
    real_trim = nv_skip / fps
    if abs(real_trim - na_skip / rate) > 1e-9:
        raise SystemExit(f"✗ {tag}: 掐头 {trim_head}s 在 {fps}fps/{rate}Hz 下对不齐")

    # 视频**不带声音**：偏移要在浏览器里调，焊在一个文件里就没得调了。
    with _trimmed(vraw, v_off) as vf:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                        "-r", str(fps), "-i", "pipe:0", "-an",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out / f"{slug}.mp4")], stdin=vf, check=True)
    with _trimmed(apcm, a_off) as af:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "s16le", "-ar", str(rate), "-ac", str(ch),
                        "-i", "pipe:0", "-c:a", "aac", "-b:a", "160k",
                        "-movflags", "+faststart",
                        str(out / f"{slug}.m4a")], stdin=af, check=True)

    return {
        "tag": tag, "note": note, "slug": slug, "session": d.name,
        "width": w, "height": h, "fps": fps,
        "sample_rate": rate, "channels": ch,
        "video_frames": m["video_frames"] - nv_skip,
        "video_seconds": round((m["video_frames"] - nv_skip) / fps, 3),
        "audio_seconds": round((apcm.stat().st_size - a_off) / (rate * ch * 2), 3),
        "trim_head": round(real_trim, 3),
        "env": envelope(apcm, rate=rate, channels=ch,
                        points=ENVELOPE_POINTS, skip_bytes=a_off),
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>%F0%9F%A6%80</text></svg>">
<style>
  :root { --bg:#f5f5f7; --card:#fff; --ink:#1c1b1f; --mute:#5f6368;
          --line:#e3e3e6; --accent:#1a73e8; --warn:#b3261e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.65 -apple-system,"Noto Sans CJK SC","PingFang SC",system-ui,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:22px 16px 72px; }
  h1 { font-size:21px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--mute); font-size:14px; margin:0 0 18px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px; margin-bottom:14px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; }
  .chip { border:1px solid var(--line); background:#fff; border-radius:999px;
          padding:7px 14px; cursor:pointer; font:inherit; font-size:14px; }
  .chip.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  .chip .done { color:#146c2e; font-weight:600; }
  .chip.on .done { color:#cfe3ff; }
  .note { color:var(--mute); font-size:13px; margin-top:10px; }
  video { width:100%; max-height:56vh; background:#000; border-radius:8px; display:block; }
  canvas { width:100%; height:80px; display:block; margin-top:6px; }
  .readout { font-size:32px; font-weight:600; font-variant-numeric:tabular-nums;
             text-align:center; margin:2px 0; }
  .readout small { font-size:13px; font-weight:400; color:var(--mute); }
  input[type=range] { width:100%; accent-color:var(--accent); height:28px; }
  .ticks { display:flex; justify-content:space-between; color:var(--mute);
           font-size:12px; margin-top:-4px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:10px; }
  button { font:inherit; padding:8px 14px; border-radius:999px; cursor:pointer;
           border:1px solid var(--line); background:#fff; color:var(--ink); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th,td { padding:7px 6px; border-bottom:1px solid var(--line); text-align:left; }
  th { color:var(--mute); font-weight:500; font-size:13px; }
  td.num { font-variant-numeric:tabular-nums; text-align:right; }
  .verdict { border-left:3px solid var(--accent); padding:10px 14px; background:#f8f9fe;
             border-radius:0 8px 8px 0; font-size:14px; }
  .verdict b { color:var(--accent); }
  kbd { background:#eceff1; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:1px 6px; font-size:12px; font-family:inherit; }
  #pattern { font-size:14px; }
</style>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">采集点在<b>递交给 LiveKit 之前</b> —— 这一页只看服务端这一半，不含传输和手机端。
     六条样本<b>彼此只差一个变量</b>，逐条拖到对齐，底下那张表会把规律排出来。</p>

  <div class="card">
    <div class="chips" id="chips"></div>
    <div class="note" id="note"></div>
  </div>

  <div class="card">
    <video id="v" playsinline muted preload="auto"></video>
    <audio id="a" preload="auto" style="display:none"></audio>
    <canvas id="wave" height="80"></canvas>
    <div class="note" id="astate">声音：还没开始</div>
  </div>

  <div class="card">
    <div class="readout"><span id="val">0</span> ms<br><small id="hint">音频与视频同步</small></div>
    <input type="range" id="off" min="-800" max="800" step="10" value="0">
    <div class="ticks"><span>−800（声音提前）</span><span>0</span><span>+800（声音延后）</span></div>
    <div class="row">
      <button id="play" class="primary">播放</button>
      <button data-d="-10">−10</button><button data-d="10">+10</button>
      <button data-f="-1">◀ 帧</button><button data-f="1">帧 ▶</button>
      <button id="zero">归零</button><button id="back">回开头</button>
    </div>
    <div class="note" style="text-align:center">
      一帧 <span id="fms"></span> ms ｜ <kbd>←</kbd><kbd>→</kbd> 逐帧 <kbd>空格</kbd> 播放
      ｜ 读数会自动记住，换条再回来还在
    </div>
  </div>

  <div class="card">
    <table id="tbl"></table>
    <div class="note" id="pattern" style="margin-top:12px"></div>
  </div>

  <div class="card">
    <div class="verdict">
      <b>这些数怎么用。</b>
      偏移<b>随时长变大</b> ⇒ 比例误差，去查帧率和重采样比；
      偏移是个<b>常数</b> ⇒ 固定延迟，对齐点差了几帧；
      只有某个<b>采样率</b>那条不一样 ⇒ 错在重采样那一段；
      改<b>前导静音</b>就变 ⇒ 流起始那一刻没对齐。<br>
      全都拖不动就得回头 —— 那说明服务端没问题，错在传输或 App。
    </div>
  </div>
</div>
<script>
const CLIPS = __CLIPS__;
const KEY = "__PREFIX__";

const v=document.getElementById('v'), a=document.getElementById('a');
const slider=document.getElementById('off'), val=document.getElementById('val');
const hint=document.getElementById('hint'), playBtn=document.getElementById('play');
const cv=document.getElementById('wave'), ctx=cv.getContext('2d');
const astate=document.getElementById('astate');
let cur = 0;

const store = {
  get(tag){ const x = localStorage.getItem(KEY+':'+tag); return x===null?null:+x; },
  set(tag,o){ localStorage.setItem(KEY+':'+tag, String(o)); },
};
function clip(){ return CLIPS[cur]; }
function offset(){ const o = store.get(clip().tag); return o===null?0:o; }
function targetAudioTime(){ return v.currentTime - offset()/1000; }

/* ── 声音走 WebAudio，不走 <audio> 元素 ────────────────────────────
   Chris 2026-09-19：「你忘了放音轨了吧？我播放视频的时候没有声。」
   音轨在、文件也是好的 —— 是 `<audio>` 元素在他那边没起来，而上一版
   把 `a.play()` 的失败 `.catch(()=>{})` 吞了，**连报都不报**。

   `<audio>` 在各家 webview 里太不可靠：飞书/微信内置浏览器、iOS 静音
   开关、第二个媒体元素拿不到播放许可 —— 任何一条中招都长成「视频在动、
   没有声音」。WebAudio 顺带还给了采样级定位，而这一页量的就是毫秒。

   解不开就退回 `<audio>`，但**把原因写在页面上**。失败要么别发生，
   要么看得见，不能两头落空。
------------------------------------------------------------------ */
let actx=null, abuf=null, asrc=null, aT0=0, vT0=0, useEl=false;
function say(t, bad){ astate.textContent='声音：'+t; astate.style.color = bad?'var(--warn)':'var(--mute)'; }

async function ensureAudio(){
  if (useEl) return;
  const c = clip();
  try {
    if (!actx) actx = new (window.AudioContext||window.webkitAudioContext)();
    if (actx.state !== 'running') await actx.resume();
    if (abuf && abuf._slug === c.slug) return;
    say('正在解码…');
    const buf = await (await fetch(c.slug+'.m4a')).arrayBuffer();
    abuf = await actx.decodeAudioData(buf);
    abuf._slug = c.slug;
    say('已就绪（WebAudio）');
  } catch (e) {
    useEl = true;
    say('WebAudio 起不来（'+e.message+'），退回 <audio> 元素', true);
  }
}
function stopSrc(){ if (asrc){ try{asrc.stop();}catch(_){ } asrc=null; } }
function startSrc(){
  if (useEl || !abuf) return;
  stopSrc();
  const at = targetAudioTime();
  if (at < 0 || at >= abuf.duration) { say('偏移把音频推到片外了'); return; }
  asrc = actx.createBufferSource();
  asrc.buffer = abuf; asrc.connect(actx.destination); asrc.start(0, at);
  aT0 = actx.currentTime - at; vT0 = v.currentTime - at;
  say('播放中（WebAudio）');
}
// ⚠️ 视频是时钟。反过来在 seek 时会打架 —— video 的 seek 是关键帧对齐的。
function syncAudio(force){
  if (useEl){
    const t = targetAudioTime();
    if (t<0 || t>(a.duration||1e9)){ if(!a.paused) a.pause(); return; }
    if (force || Math.abs(a.currentTime-t) > 0.035) a.currentTime = t;
    if (!v.paused && a.paused) a.play().then(()=>say('播放中（<audio>）'))
                                       .catch(e=>say('播不出来：'+e.message, true));
    return;
  }
  if (!asrc || v.paused) return;
  const drift = (actx.currentTime - aT0) - (v.currentTime - vT0);
  if (Math.abs(drift) > 0.05) startSrc();
}

function renderChips(){
  document.getElementById('chips').innerHTML = CLIPS.map((c,i)=>{
    const o = store.get(c.tag);
    const mark = o===null ? '' : ' <span class="done">'+(o>0?'+':'')+o+'</span>';
    return '<button class="chip'+(i===cur?' on':'')+'" data-i="'+i+'">'+c.tag+mark+'</button>';
  }).join('');
  document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>select(+b.dataset.i));
}

function select(i){
  cur = i; const c = clip();
  v.pause(); a.pause(); stopSrc(); playBtn.textContent='播放';
  abuf = null;                                  // 换样本就得重新解码
  v.src = c.slug+'.mp4'; a.src = c.slug+'.m4a';
  document.getElementById('note').textContent =
    c.note+' ｜ '+c.sample_rate+' Hz ｜ 视频 '+c.video_seconds.toFixed(2)+
    ' s ｜ 音频 '+c.audio_seconds.toFixed(2)+' s ｜ '+c.width+'×'+c.height+' @ '+c.fps+' fps';
  document.getElementById('fms').textContent = (1000/c.fps).toFixed(0);
  say('还没开始（点播放）');
  slider.value = offset(); applyOffset(false);
  renderChips(); renderTable();
}

function applyOffset(save){
  const o = +slider.value;
  val.textContent = (o>0?'+':'')+o;
  hint.textContent = o===0 ? '音频与视频同步'
    : o>0 ? '声音比画面晚 '+o+' ms' : '声音比画面早 '+(-o)+' ms';
  if (save){ store.set(clip().tag, o); renderChips(); renderTable(); }
  if (!v.paused && !useEl) startSrc(); else syncAudio(true);
}

async function toggle(){
  if (v.paused){
    await ensureAudio();        // ⚠️ resume 必须在用户手势里，否则 iOS 不放行
    v.play(); playBtn.textContent='暂停';
    if (useEl) a.play().then(()=>say('播放中（<audio>）'))
                       .catch(e=>say('播不出来：'+e.message, true));
    else startSrc();
  } else {
    v.pause(); a.pause(); stopSrc(); playBtn.textContent='播放'; say('已暂停');
  }
}

function step(n){
  v.pause(); a.pause(); stopSrc(); playBtn.textContent='播放';
  v.currentTime = Math.max(0, v.currentTime + n*(1/clip().fps));
  setTimeout(()=>syncAudio(true), 30);
}

function drawWave(){
  const c = clip(), w = cv.clientWidth, h = cv.height, dpr = devicePixelRatio||1;
  cv.width = w*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#dbe4f3';
  const n = c.env.length;
  for (let i=0;i<n;i++){
    const x = i/n*w, bh = Math.max(1, c.env[i]*(h-22));
    ctx.fillRect(x, h-18-bh, Math.max(1, w/n), bh);
  }
  const x = Math.max(0, Math.min(1, targetAudioTime()/(c.audio_seconds||1)))*w;
  ctx.fillStyle='#b3261e'; ctx.fillRect(x-1, 0, 2, h-14);
  ctx.fillStyle='#5f6368'; ctx.font='11px system-ui';
  ctx.fillText('音频包络（红线 = 此刻听到的位置）', 4, h-3);
  requestAnimationFrame(drawWave);
}

function renderTable(){
  const head = '<tr><th>样本</th><th>变量</th><th class="num">采样率</th>'
             + '<th class="num">时长 s</th><th class="num">你量到的偏移</th></tr>';
  const rows = CLIPS.map(c=>{
    const o = store.get(c.tag);
    return '<tr><td><b>'+c.tag+'</b></td><td>'+c.note+'</td>'
      +'<td class="num">'+(c.sample_rate/1000)+'k</td>'
      +'<td class="num">'+c.audio_seconds.toFixed(1)+'</td>'
      +'<td class="num">'+(o===null?'—':((o>0?'+':'')+o+' ms'))+'</td></tr>';
  }).join('');
  document.getElementById('tbl').innerHTML = head+rows;

  const got = CLIPS.map(c=>({c, o:store.get(c.tag)})).filter(x=>x.o!==null);
  const p = document.getElementById('pattern');
  if (got.length < 2){ p.textContent='把两条以上拖齐之后，这里会给出初步判断。'; return; }
  const vals = got.map(x=>x.o), spread = Math.max(...vals)-Math.min(...vals);
  const mean = Math.round(vals.reduce((s,x)=>s+x,0)/vals.length);
  let t = '已量 '+got.length+'/'+CLIPS.length+' 条：均值 '+mean+' ms，极差 '+spread+' ms。';
  // 直接写 <b>，不要在这儿摆一个 markdown 小解析器 —— 那个正则里的反斜杠
  // 在 Python 模板字符串里还要再转一层，两层转义栈起来必出事（已出过）。
  t += spread <= 40
    ? ' 各条基本一致 ⇒ 像是<b>固定延迟</b>，去查对齐点差了几帧，不是比例问题。'
    : ' 各条差得多 ⇒ 不是单一常数，看看是随时长涨（比例误差）还是只有某个采样率不一样。';
  p.innerHTML = t;
}

slider.addEventListener('input', ()=>applyOffset(true));
document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{
  slider.value = Math.max(-800, Math.min(800, +slider.value + (+b.dataset.d)));
  applyOffset(true);
});
document.querySelectorAll('[data-f]').forEach(b=>b.onclick=()=>step(+b.dataset.f));
document.getElementById('zero').onclick=()=>{ slider.value=0; applyOffset(true); };
document.getElementById('back').onclick=()=>{ v.currentTime=0; syncAudio(true); };
playBtn.onclick=toggle;
v.addEventListener('pause', ()=>{ a.pause(); stopSrc(); playBtn.textContent='播放'; });
addEventListener('keydown', e=>{
  if(e.key==='ArrowLeft'){ step(-1); e.preventDefault(); }
  if(e.key==='ArrowRight'){ step(1); e.preventDefault(); }
  if(e.key===' '){ toggle(); e.preventDefault(); }
});
(function loop(){ if(!v.paused) syncAudio(false); requestAnimationFrame(loop); })();

renderChips(); select(0); drawWave();
</script>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True,
                    help='「短名|变量说明|录制目录[|掐头秒数]」，可重复')
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True, help="产物文件名前缀，保证同目录不打架")
    ap.add_argument("--title", default="数字人音画同步检查")
    ap.add_argument("--trim-head", type=float, default=0.0,
                    help="音视频各掐掉开头这么多秒（去掉探针垫的静音前导）")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for spec in a.clip:
        parts = [x.strip() for x in spec.split("|")]
        if len(parts) not in (3, 4):
            raise SystemExit(
                f"✗ --clip 要「短名|说明|目录」，可选第四段是这一条单独的掐头秒数，"
                f"收到：{spec}")
        tag, note, d = parts[0], parts[1], parts[2]
        # 每条的静音前导长度可能不一样（探针的 --leadin 是按条给的），
        # 所以掐头也得按条给；不给就用全局那个。
        trim = float(parts[3]) if len(parts) == 4 else a.trim_head
        print(f"── {tag}  {note}"
              + (f"（掐头 {trim:g}s）" if trim else ""))
        clips.append(build_clip(tag, note, pathlib.Path(d), out, a.prefix,
                                trim_head=trim))

    html = (HTML.replace("__TITLE__", a.title)
                .replace("__PREFIX__", a.prefix)
                .replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)))
    # ⚠️ **生成完必须自检。** 2026-09-19 我用一串 str.replace 给这段 JS 打补丁，
    #    其中几处没匹配上 —— Python 的 replace **匹配不到就静默返回原串**，
    #    于是九个函数被前一步连带删掉、后面的补丁又全部落空，发出去是一个
    #    只剩标题的白页。Chris：「你是不是把页面改坏了？」
    #    模板这种东西没有编译器兜着，那道防线只能自己加。
    need = ["function clip(", "function offset(", "function renderChips(",
            "function select(", "function applyOffset(", "async function toggle(",
            "function step(", "function drawWave(", "function renderTable(",
            "const store", "renderChips(); select(0); drawWave();"]
    missing = [n for n in need if n not in html]
    if missing:
        raise SystemExit("✗ 生成的页面缺了这些，八成是模板被改坏了：\n  "
                         + "\n  ".join(missing))
    (out / f"{a.prefix}.html").write_text(
        html, encoding="utf-8")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"✅ {out}/{a.prefix}.html  （共 {total/1e6:.1f} MB，{len(clips)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
