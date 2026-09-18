#!/usr/bin/env bash
# 把 `CCA_AV_DUMP` 录下来的裸流合成一个 mp4。
#
# 用法：scripts/mux-av-dump.sh <录制目录> [输出.mp4]
#
# ## 为什么参数全从 meta.json 读，一个都不许在命令行里猜
#
# 尺寸、帧率、采样率任何一个填错，合出来的东西**照样能播**，只是画面拉伸
# 或者声音变调 —— 而我们正是拿这份 mp4 去判断「生成质量到底行不行」。
# 一个猜错的参数会让整场判断作废，而且看起来像模型的锅。
set -euo pipefail

DIR="${1:?用法: mux-av-dump.sh <录制目录> [输出.mp4]}"
OUT="${2:-$DIR/preview.mp4}"
META="$DIR/meta.json"

[ -f "$META" ] || { echo "✗ 没有 $META —— 这一场没正常收尾，参数不可信" >&2; exit 1; }

read -r W H FPS RATE CH < <(python3 -c "
import json,sys
m=json.load(open('$META'))
print(m['width'], m['height'], m['fps'], m['sample_rate'], m['channels'])
")

echo "录制信息：${W}x${H} @ ${FPS}fps，音频 ${RATE}Hz ${CH}声道"
python3 -c "
import json,os
m=json.load(open('$META'))
v=os.path.getsize('$DIR/video.rgba'); a=os.path.getsize('$DIR/audio.pcm')
vs=v/(m['width']*m['height']*4)/m['fps']
as_=a/(m['sample_rate']*m['channels']*2)
print(f'视频 {vs:.2f} 秒 / 音频 {as_:.2f} 秒 —— 差 {abs(vs-as_)*1000:.0f} ms')
print('⚠️ 两者相差超过一帧，说明服务端这一侧就没配平' if abs(vs-as_) > 1/m['fps'] else '✅ 时长对得上')
"

# ⚠️ 视频用 `-r` 而不是 `-framerate`：裸流没有时间信息，全靠这个数定节奏。
#    填错的话画面会整体快放或慢放，而口型对不上看起来跟模型没画好一模一样。
ffmpeg -hide_banner -loglevel warning -y \
  -f rawvideo -pix_fmt rgba -s "${W}x${H}" -r "$FPS" -i "$DIR/video.rgba" \
  -f s16le -ar "$RATE" -ac "$CH" -i "$DIR/audio.pcm" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -shortest "$OUT"

echo "✅ $OUT  ($(du -h "$OUT" | cut -f1))"
