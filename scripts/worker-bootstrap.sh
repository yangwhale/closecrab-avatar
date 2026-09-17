#!/usr/bin/env bash
# 在一台干净的 GPU 机器上把 worker 环境装好。**幂等，可以反复跑。**
#
# 为什么要有这个脚本而不是手敲：worker 跑在 Spot 上，**实例说没就没**，
# 回来的是一台全新的机器（家目录一并清空）。2026-09-17 就撞上一次：
# 之前测出 1.357× 实时的那台被回收了，`~/LiveAvatar` 连同环境全没了，
# 而 SSH 别名里的 IP 也失效。手敲的东西在这种机器上等于没装。
#
#   bash worker-bootstrap.sh            # 装环境 + 拉权重（权重在后台拉）
#   bash worker-bootstrap.sh --check    # 只报状态，什么都不动
set -euo pipefail

ROOT="${LA_WORKER_ROOT:-$HOME/LiveAvatar}"
REPO="${LA_WORKER_REPO:-https://github.com/yangwhale/LiveAvatar.git}"
BRANCH="${LA_WORKER_BRANCH:-b200-realtime}"
VENV="$ROOT/.venv"
CKPT="$ROOT/ckpt"
LOG="${LA_WORKER_LOG:-/tmp/la-worker-bootstrap.log}"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

say() { echo "[$(date '+%F %T')] $*"; }

# ── 状态 ────────────────────────────────────────────────────────────
say "GPU: $(nvidia-smi -L 2>/dev/null | wc -l) 张"
say "仓库: $([[ -d $ROOT/.git ]] && git -C "$ROOT" rev-parse --short HEAD || echo '未克隆')"
say "venv: $([[ -x $VENV/bin/python ]] && "$VENV/bin/python" -V || echo '未建')"
for d in Wan2.2-S2V-14B LiveAvatar; do
    # 判据用**体积**不用「目录在不在」——半截的下载同样会建出目录，
    # 按存在与否判断会把一个下到一半的权重当成装好了。
    sz=$(du -sh "$CKPT/$d" 2>/dev/null | cut -f1 || true)
    say "权重 $d: ${sz:-无}"
done
[[ $CHECK_ONLY -eq 1 ]] && exit 0

# ── 1. 代码 ─────────────────────────────────────────────────────────
if [[ -d "$ROOT/.git" ]]; then
    say "仓库已在，拉一下"
    git -C "$ROOT" fetch --depth 1 origin "$BRANCH" && git -C "$ROOT" reset --hard FETCH_HEAD
else
    say "克隆 $BRANCH"
    git clone --depth 1 -b "$BRANCH" "$REPO" "$ROOT"
fi

# ── 2. venv ─────────────────────────────────────────────────────────
# 不碰系统 Python：这台机器上还有别的东西在用。
#
# ⚠️⚠️ **绝对不要加 `--system-site-packages`。** 2026-09-17 被它咬了两次：
#
#   1. `pip install huggingface_hub` → 系统里已有，pip 判「已满足」什么都不装，
#      **入口脚本 `hf` 从来没进 venv/bin**。包能 import，命令不存在。
#   2. `pip install torch==2.8.0 --index-url .../cu128` → torch 本体装进来了，
#      但配套的 `nvidia-*-cu12` 一堆库，pip 看到系统里有 cu129 版就判「已满足」，
#      **cu128 那套压根没装** → `import torch` 直接
#      `ImportError: libcusparseLt.so.0: cannot open shared object file`。
#
# 两次都是 **pip 返回 0**，只有真用到才炸。规矩：**要某个包的特定版本、而系统里
# 也有它时，就别共享 site-packages。** 省那几个 G 换来一个说不清装了什么的环境，
# 在 Spot 上尤其亏。
if [[ ! -x "$VENV/bin/python" ]]; then
    say "建干净 venv（不共享 system site-packages，见上方注释）"
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
# ⚠️ **不要依赖 `hf` / `huggingface-cli` 这个命令行入口。**
#    venv 建在 --system-site-packages 上时，系统里已经有 huggingface_hub，
#    pip 判定「已满足」于是什么都不装 —— 包能 import，但**入口脚本从来没进
#    venv 的 bin**。现象很骗人：pip 返回 0、import 正常，只有可执行文件不在。
#    下面直接用 Python API，绕开整件事。
"$VENV/bin/pip" install -q huggingface_hub
"$VENV/bin/python" -c "import huggingface_hub" || { echo "huggingface_hub 装不上"; exit 1; }

# ── 2b. torch：**必须是 fork 验证过的那一版** ───────────────────────
# README:125 写的是 torch 2.8.0 + cu128，flash_attn_3 的 wheel 也是
# `cu128torch280` 版。机器自带的 2.9.1+cu129 看着更新，实际会让
# transformers 4.51.3（requirements 的上限）在 modeling_opt 那里循环导入：
#     from transformers import PreTrainedModel  → RuntimeError
# 不是 peft 的锅，是 transformers 本身在那个 torch 上就起不来。
if ! "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8.0') else 1)" 2>/dev/null; then
    say "装 torch 2.8.0 + cu128（fork 验证过的栈，几个 G，慢）"
    "$VENV/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# ── 3. 权重（后台，很大）────────────────────────────────────────────
mkdir -p "$CKPT"
pull() {   # $1=HF repo  $2=本地目录名
    local repo="$1" dir="$2"
    if [[ -f "$CKPT/$dir/.complete" ]]; then
        say "权重 $dir 已完整，跳过"
        return
    fi
    say "开始拉 $repo → $CKPT/$dir（后台，看 $LOG）"
    # ⚠️ setsid + nohup：ssh 断开不能把下载带走。
    #    远端起的长进程必须自带死期，所以套 timeout ——
    #    不套的话 ssh 一断就成孤儿，没人知道它还在不在。
    setsid nohup timeout 21600 bash -c "
        '$VENV/bin/python' -c \"
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$CKPT/$dir', max_workers=16)
\" >>'$LOG' 2>&1 && touch '$CKPT/$dir/.complete'
    " >/dev/null 2>&1 &
}
pull Wan-AI/Wan2.2-S2V-14B Wan2.2-S2V-14B
pull Quark-Vision/Live-Avatar LiveAvatar

say "环境就绪；权重在后台拉。进度： tail -f $LOG ；状态： bash $0 --check"
