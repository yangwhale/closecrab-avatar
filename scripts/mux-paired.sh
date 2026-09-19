#!/usr/bin/env bash
# 把「配对落盘」出来的东西合成 mp4。
#
# ⭐ 这里**没有任何偏移参数**，而且不可能有 —— 文件里每一帧就写在生成它
#    的那段音频后面，同步是构造出来的。要是这个片子还不同步，那说明
#    账本的配对本身错了，不是"补偿量没调好"。
#
# 用法：scripts/mux-paired.sh <配对目录> [输出.mp4]
set -euo pipefail
DIR="${1:?用法: mux-paired.sh <配对目录> [输出.mp4]}"
OUT="${2:-$DIR/paired.mp4}"
read -r W H FPS RATE < <(python3 -c "
import json; m=json.load(open('$DIR/meta.json'))
print(m['width'], m['height'], m['fps'], m['sample_rate'])")
python3 -c "
import json, os
m=json.load(open('$DIR/meta.json'))
v=os.path.getsize('$DIR/paired.rgba')/(m['width']*m['height']*4)
a=os.path.getsize('$DIR/paired.pcm')/(m['sample_rate']*2)
print(f'视频 {v:.0f} 帧 = {v/m[\"fps\"]:.2f}s ; 音频 {a:.2f}s ; 差 {abs(v/m[\"fps\"]-a)*1000:.0f} ms')
print('⚠️ 差超过一帧说明配对本身有问题' if abs(v/m['fps']-a) > 1/m['fps'] else '✅ 逐块配对，时长自洽')"
ffmpeg -hide_banner -loglevel warning -y \
  -f rawvideo -pix_fmt rgba -s "${W}x${H}" -r "$FPS" -i "$DIR/paired.rgba" \
  -f s16le -ar "$RATE" -ac 1 -i "$DIR/paired.pcm" \
  -map 0:v -map 1:a -shortest \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -movflags +faststart "$OUT"
echo "✅ $OUT ($(du -h "$OUT" | cut -f1))"
