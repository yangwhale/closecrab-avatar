#!/usr/bin/env bash
# CloseCrab Avatar 一键部署。
#
#   install.sh --role all                # 控制面 + worker 同机（单机玩法）
#   install.sh --role control            # 只装控制面（CPU 小机器）
#   install.sh --role worker             # 只装 GPU worker
#   install.sh --check                   # 只体检，什么都不动
#
# **幂等**：装过的跳过，可以反复跑。
#
# ## 为什么一定要有这个脚本
#
# worker 跑在 Spot 上，**实例说没就没**，回来的是一台全新机器。手敲的东西
# 在这种机器上等于没装。控制面同理 —— 换台机器要能十分钟内起来。
#
# ## 它不会替你做的事
#
# **不生成、不猜测任何凭据。** LiveKit 的 key/secret 必须由你提供
# （`--livekit-key/--livekit-secret`，或事先写好 env 文件）。
# 脚本宁可停下来问，也不会塞一个默认值进去 —— 那种默认值会一路跑到生产。
set -euo pipefail

ROLE=""
CHECK=0
GATEWAY_URL="${LA_GATEWAY_URL:-http://127.0.0.1:8080}"
PORT="${LA_PORT:-8080}"
LK_KEY="${LIVEKIT_API_KEY:-}"
LK_SECRET="${LIVEKIT_API_SECRET:-}"
WORKER_CAPACITY="${LA_WORKER_CAPACITY:-1}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ETC=/etc/closecrab-avatar
VAR=/var/lib/closecrab-avatar
UNIT_CTL=closecrab-avatar.service
UNIT_WRK=closecrab-avatar-worker@.service

say()  { echo "[$(date '+%F %T')] $*"; }
die()  { echo "❌ $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role) ROLE="$2"; shift 2 ;;
        --check) CHECK=1; shift ;;
        --gateway) GATEWAY_URL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --livekit-key) LK_KEY="$2"; shift 2 ;;
        --livekit-secret) LK_SECRET="$2"; shift 2 ;;
        --capacity) WORKER_CAPACITY="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) die "未知参数 $1（--help 看用法）" ;;
    esac
done

# ── 体检 ──────────────────────────────────────────────────────────────

status() {
    # ⚠️ 不要写 `$(cmd || echo 默认)` —— `systemctl is-active` 在服务没跑时
    #    **本来就会打印 "inactive"**，只是退出码非零，于是 `|| echo` 又追加一行，
    #    整个字段变成两行、后面的输出跟着错位。`grep -c` 同理。
    #    正确做法：`x=$(cmd) || true`，再用 ${VAR:-默认} 兜底。
    #
    # ⚠️ 那个 `|| true` 也不能省：`set -e` 下**裸赋值的退出码就是命令替换的
    #    退出码**，而 `is-active` 在服务没跑时返回 4 —— 整个脚本会一声不吭
    #    地退出（连一行输出都没有，只剩 rc=4）。
    #    注意它挂在赋值后面、不在 `$( )` 里面，所以不会追加任何东西。
    local act ena
    act=$(systemctl is-active  $UNIT_CTL 2>/dev/null) || true; act=${act:-未安装}
    ena=$(systemctl is-enabled $UNIT_CTL 2>/dev/null) || true; ena=${ena:--}
    echo "── CloseCrab Avatar 状态 ──"
    printf "  控制面服务   %s\n" "$act"
    printf "  开机自启     %s\n" "$ena"
    printf "  凭据文件     %s\n" "$([[ -f $ETC/env ]] && echo "$ETC/env" || echo 无)"
    printf "  状态库       %s\n" "$([[ -f $VAR/state.db ]] && du -h "$VAR/state.db" | cut -f1 || echo 无)"
    local hz
    hz=$(curl -s -m 3 "http://127.0.0.1:$PORT/healthz" 2>/dev/null || true)
    printf "  /healthz     %s\n" "${hz:-打不通}"
    printf "  GPU          %s 张\n" "$(nvidia-smi -L 2>/dev/null | wc -l)"
    printf "  worker 环境  %s\n" \
        "$([[ -x $HOME/LiveAvatar/.venv/bin/python ]] && echo 已装 || echo 未装)"
    # ⚠️ 权重按**体积**判，不按目录在不在 —— 下到一半同样会建出目录。
    for d in Wan2.2-S2V-14B LiveAvatar; do
        printf "  权重 %-16s %s\n" "$d" \
            "$(du -sh "$HOME/LiveAvatar/ckpt/$d" 2>/dev/null | cut -f1 || echo 无)"
    done
}

[[ $CHECK -eq 1 ]] && { status; exit 0; }
[[ -n "$ROLE" ]] || die "要指定 --role control|worker|all（或 --check）"
[[ "$ROLE" =~ ^(control|worker|all)$ ]] || die "--role 只能是 control / worker / all"

# ── 控制面 ────────────────────────────────────────────────────────────

install_control() {
    say "装控制面（端口 $PORT）"
    have python3 || die "没有 python3"

    # ⚠️ **不往系统 Python 装。** Ubuntu 24.04 起是 externally-managed（PEP 668），
    #    `pip install` 直接被拒；就算加 --break-system-packages 能过，那也是在
    #    往发行版管理的目录里塞东西，升级时会互相打架。
    #    控制面用自己的 venv —— 依赖就那几个，代价可以忽略。
    local VENV="$REPO_DIR/.venv"
    [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt" || die "依赖装不上"
    # 装完真的 import 一遍 —— 「pip 返回 0」不等于能跑。
    "$VENV/bin/python" -c "import fastapi, uvicorn, jwt, livekit.api" \
        || die "依赖装了但 import 不了"

    sudo mkdir -p "$ETC" "$VAR"
    sudo chown "$USER":"$USER" "$VAR"

    if [[ ! -f "$ETC/env" ]]; then
        # ⚠️ **绝不生成假凭据。** 一个能起来但铸不出有效 token 的网关，
        #    症状是客户端连不上房间 —— 看起来像 LiveKit 坏了。宁可现在停。
        [[ -n "$LK_KEY" && -n "$LK_SECRET" ]] || die \
"缺 LiveKit 凭据。两种给法：
   install.sh --role $ROLE --livekit-key <k> --livekit-secret <s>
   或先自己写好 $ETC/env（见 docs/deploy.md）"

        local api_key api_secret
        api_key="default"
        api_secret="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"

        # 0600 在创建那一刻定死 —— 先 open 后 chmod 中间有一瞬是 0644。
        sudo install -m 600 /dev/null "$ETC/env"
        sudo tee "$ETC/env" >/dev/null <<EOF
# 由 scripts/install.sh 生成。改完 systemctl restart $UNIT_CTL 生效。
LIVEKIT_API_KEY=$LK_KEY
LIVEKIT_API_SECRET=$LK_SECRET
LA_API_KEYS=[{"key_id":"$api_key","secret":"$api_secret","max_concurrency":8}]
LA_GATEWAY_DB=$VAR/state.db
LA_IDLE_TIMEOUT_S=120
LA_MAX_SESSION_S=3600
EOF
        say "已生成 $ETC/env（0600）"
        echo
        echo "  ⚠️ 调用方要用的 API key（**只显示这一次**）："
        echo "     key_id = $api_key"
        echo "     secret = $api_secret"
        echo
    else
        say "$ETC/env 已存在，不覆盖"
    fi

    sudo tee /etc/systemd/system/$UNIT_CTL >/dev/null <<EOF
[Unit]
Description=CloseCrab Avatar 控制面（调度 + 铸票 + 槽位回收）
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ETC/env
ExecStart=$VENV/bin/python -m uvicorn --factory closecrab_avatar.entry:build \\
          --host 0.0.0.0 --port $PORT --log-level info
Restart=always
RestartSec=3
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$VAR

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now $UNIT_CTL
    sleep 3
    curl -sf -m 5 "http://127.0.0.1:$PORT/healthz" >/dev/null \
        || die "控制面起来了但 /healthz 打不通，看 journalctl -u $UNIT_CTL"
    say "✅ 控制面就绪：$(curl -s -m 3 http://127.0.0.1:$PORT/healthz)"
}

# ── GPU worker ────────────────────────────────────────────────────────

install_worker() {
    say "装 GPU worker（控制面 $GATEWAY_URL，容量 $WORKER_CAPACITY）"
    [[ $(nvidia-smi -L 2>/dev/null | wc -l) -gt 0 ]] || die "这台机器上看不到 GPU"

    # 模型环境：版本坑全在这个脚本里（干净 venv / torch 2.8.0 / 卸 deepspeed /
    # FA2 而不是 FA3），装完会自己 import 一遍验证。见 docs/gpu-setup.md。
    bash "$REPO_DIR/scripts/worker-bootstrap.sh" || die "worker 环境没装成"

    sudo tee /etc/systemd/system/$UNIT_WRK >/dev/null <<EOF
[Unit]
Description=CloseCrab Avatar GPU worker（%i 号卡）
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$REPO_DIR
Environment=CUDA_VISIBLE_DEVICES=%i
Environment=LA_GATEWAY_URL=$GATEWAY_URL
ExecStart=$HOME/LiveAvatar/.venv/bin/python -m worker.runner \\
          --gateway $GATEWAY_URL --worker-id %H-gpu%i --capacity $WORKER_CAPACITY
# Spot 会被回收、模型会偶发崩 —— 一律拉起来重新注册。
# 控制面那边靠心跳超时把旧的摘掉，不会重复计数。
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    say "✅ worker 环境就绪。"
    say ""
    say "   下一步要先选路（见 docs/benchmarks.md 第一节）："
    say "   · 多路并发 —— 一卡一进程，8 卡 = 8 路："
    say "       sudo systemctl enable --now closecrab-avatar-worker@0   # …@7"
    say "   · 单路流式 —— 五卡一组（4 DiT + 1 VAE），8 卡机只能开 1 组："
    say "       见 docs/deploy.md「五卡流式怎么起」"
}

case "$ROLE" in
    control) install_control ;;
    worker)  install_worker ;;
    all)     install_control; install_worker ;;
esac

echo
status
