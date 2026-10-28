#!/usr/bin/env bash
# ============================================================================
#  AI2050 平台一键部署（CentOS 7/8/9 Stream）
#  域名：server.ai2050.com
#
#  作用：把本仓库部署为
#     nginx  :80/:443  →  静态前端 dist + /api 反代 127.0.0.1:5001（同源，无 CORS）
#     systemd ai2050-server → FastAPI :5001（AI2050_SKIP_MODEL_LOAD=1，无需 GPU/模型）
#     systemd ai2050-distributed（可选）→ FastAPI :5010（分布式平台）
#
#  用法（在仓库根目录执行）：
#     sudo bash deploy/install_centos.sh                    # 交互式（默认）
#     sudo bash deploy/install_centos.sh --yes              # 无人值守
#     sudo bash deploy/install_centos.sh --yes --no-https   # 跳过 HTTPS
#     sudo bash deploy/install_centos.sh --yes --skip-dist   # 跳过 :5010
#
#  幂等：可重复执行。已存在的目录/venv/unit 只更新不重建；nginx 配置先备份后覆盖。
#  不幂等的部分（会明确提示）：--fresh 重建 venv、--reset-admin-token 换令牌。
#
#  前置：域名 server.ai2050.com 的 A 记录已指向本机公网 IP。
# ============================================================================
set -euo pipefail

# ── 可配置项（环境变量可覆盖）────────────────────────────────
DOMAIN="${DOMAIN:-server.ai2050.com}"
REPO_SRC="${REPO_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/ai2050}"
REPO_DIR="${INSTALL_ROOT}/Ai2050-OpenOne"
VENV="${INSTALL_ROOT}/venv"
DATA_DIR="${DATA_DIR:-/var/lib/ai2050/distributed}"
WEB_ROOT="${WEB_ROOT:-/var/www/ai2050}"
SERVICE_USER="${SERVICE_USER:-ai2050}"
SERVER_PORT="${SERVER_PORT:-5001}"
DIST_PORT="${DIST_PORT:-5010}"
PYBIN="${PYBIN:-python3.11}"

ASSUME_YES=0
DO_HTTPS=1
DO_DIST=1
DO_FRESH=0
RESET_TOKEN=0

for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --no-https) DO_HTTPS=0 ;;
    --skip-dist) DO_DIST=0 ;;
    --fresh) DO_FRESH=1 ;;
    --reset-admin-token) RESET_TOKEN=1 ;;
    --help|-h) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（--help 查看用法）"; exit 2 ;;
  esac
done

# ── 输出helpers ───────────────────────────────────────────────
BOLD=$'\033[1m'; GRN=$'\033[32m'; YLW=$'\033[33m'; RED=$'\033[31m'; RST=$'\033[0m'
step() { echo "${BOLD}==>${RST} $*"; }
ok()   { echo "  ${GRN}✓${RST} $*"; }
warn() { echo "  ${YLW}!${RST} $*"; }
die()  { echo "  ${RED}✗${RST} $*" >&2; exit 1; }
ask()  { # ask <提示> <默认值>
  if [ "$ASSUME_YES" = "1" ]; then echo "$2"; return; fi
  local reply; read -r -p "  $1 [$2] " reply </dev/tty || true
  echo "${reply:-$2}"
}

# ── 0. 权限与环境检查 ─────────────────────────────────────────
step "检查权限与工具"
[ "$(id -u)" = "0" ] || die "请用 root/sudo 执行（需要 systemctl / nginx / useradd）"
command -v systemctl >/dev/null || die "无 systemctl —— 这不是 systemd 系统"
command -v nginx    >/dev/null || warn "未安装 nginx，稍后安装"
[ -d "$REPO_SRC/server" ] || die "REPO_SRC 不是仓库根（含 server/ 目录）：$REPO_SRC"
[ -f "$REPO_SRC/server/server.py" ] || die "缺 server/server.py：$REPO_SRC"
command -v "$PYBIN" >/dev/null || die "缺 $PYBIN，请先 dnf install -y python3.11 python3.11-pip"
ok "仓库源：$REPO_SRC"

# ── 1. 系统用户与目录 ─────────────────────────────────────────
step "准备用户与目录"
if id "$SERVICE_USER" >/dev/null 2>&1; then
  ok "用户已存在：$SERVICE_USER"
else
  useradd -r -s /sbin/nologin "$SERVICE_USER"
  ok "已创建系统用户：$SERVICE_USER"
fi
mkdir -p "$INSTALL_ROOT" "$DATA_DIR" "$WEB_ROOT"
if [ ! -d "$REPO_DIR" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  cp -a "$REPO_SRC" "$REPO_DIR"
  ok "已复制仓库 → $REPO_DIR"
else
  # 已存在：同步代码但保留 venv/数据（rsync 优先，退化 cp）
  if command -v rsync >/dev/null; then
    rsync -a --delete --exclude '.git' --exclude 'node_modules' --exclude 'tests' \
          --exclude 'frontend/node_modules' --exclude '.venv' "$REPO_SRC/" "$REPO_DIR/"
    ok "已同步代码（rsync，保留 .venv/node_modules/tests）"
  else
    warn "无 rsync，跳过代码同步（如需更新请手动 cp）"
  fi
fi
mkdir -p "$REPO_DIR/ai2050_research_os/generated"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_ROOT" "$DATA_DIR"
ok "目录就绪"

# ── 2. 前端构建产物 ───────────────────────────────────────────
step "准备前端 dist"
if [ -d "$REPO_SRC/frontend/dist" ] && [ -n "$(ls -A "$REPO_SRC/frontend/dist" 2>/dev/null)" ]; then
  rsync -a --delete "$REPO_SRC/frontend/dist/" "$WEB_ROOT/"
  ok "dist 已同步 → $WEB_ROOT（$(du -sh "$WEB_ROOT" | cut -f1)）"
else
  warn "frontend/dist 为空。服务器部署请在本机先执行：cd frontend && npx vite build"
  warn "然后重跑本脚本，或手动把 dist/ 上传到 $WEB_ROOT"
fi

# ── 3. Python venv 与依赖 ─────────────────────────────────────
step "准备 Python 环境"
if [ "$DO_FRESH" = "1" ] && [ -d "$VENV" ]; then
  rm -rf "$VENV"; warn "已按 --fresh 删除旧 venv"
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$PYBIN" -m venv "$VENV"
  ok "已创建 venv：$VENV"
else
  ok "venv 已存在（--fresh 可重建）"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip setuptools wheel >/dev/null
ok "pip 已升级"

# torch 单独处理：CPU wheel 索引（服务器通常无 GPU）
if ! "$VPY" -c "import torch" >/dev/null 2>&1; then
  echo "  安装 torch（CPU wheel）…"
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  ok "torch 已安装（CPU 版）"
else
  ok "torch 已存在：$("$VPY" -c 'import torch;print(torch.__version__)')"
fi

REQ="$REPO_DIR/requirements-server.txt"
if [ -f "$REQ" ]; then
  "$VPY" -m pip install -r "$REQ"
  ok "requirements-server.txt 安装完成"
else
  "$VPY" -m pip install fastapi "uvicorn[standard]" pydantic numpy scipy transformers safetensors
  warn "未找到 requirements-server.txt，安装了基础包"
fi

# 关键校验：必须在仓库根可导入 transformer_lens（源码目录）
( cd "$REPO_DIR" && "$VPY" -c "import transformer_lens, fastapi, torch; print('  import 校验通过')" ) \
  || die "import 校验失败（transformer_lens 需在仓库根启动）"

# ── 4. 管理令牌 ────────────────────────────────────────────────
step "管理令牌"
TOKEN_FILE="/etc/ai2050/admin_token"
if [ "$RESET_TOKEN" = "1" ] || [ ! -f "$TOKEN_FILE" ]; then
  NEW_TOKEN="$("$VPY" -c 'import secrets;print(secrets.token_hex(24))')"
  mkdir -p /etc/ai2050
  printf '%s' "$NEW_TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  ok "新令牌已写入 $TOKEN_FILE"
else
  NEW_TOKEN="$(cat "$TOKEN_FILE")"
  ok "沿用现有令牌（--reset-admin-token 可更换）"
fi

# ── 5. systemd：中心服务 :5001 ────────────────────────────────
step "安装 systemd 单元（ai2050-server :$SERVER_PORT）"
UNIT_SRC="$REPO_DIR/deploy/ai2050-server.service"
[ -f "$UNIT_SRC" ] || die "缺 $UNIT_SRC（deploy/ 不入 git，需从本机拷贝）"
sed -e "s#change-me-to-random-hex#${NEW_TOKEN}#" "$UNIT_SRC" > /etc/systemd/system/ai2050-server.service
systemctl daemon-reload
systemctl enable ai2050-server >/dev/null 2>&1 || true
systemctl restart ai2050-server
sleep 6
if systemctl is-active --quiet ai2050-server; then
  ok "ai2050-server 运行中"
else
  echo "  ---- journalctl 尾部 ----"; journalctl -u ai2050-server -n 30 --no-pager || true
  die "ai2050-server 启动失败（见上方日志）"
fi

# ── 6. systemd：分布式平台 :5010（可选）──────────────────────
if [ "$DO_DIST" = "1" ] && [ -f "$REPO_DIR/deploy/distributed_service.py" ]; then
  step "安装 systemd 单元（ai2050-distributed :$DIST_PORT）"
  sed -e "s#change-me-to-random-hex#${NEW_TOKEN}#" \
      -e "s#--port 5010#--port ${DIST_PORT}#" \
      "$REPO_DIR/deploy/ai2050-distributed.service" > /etc/systemd/system/ai2050-distributed.service
  systemctl daemon-reload
  systemctl enable ai2050-distributed >/dev/null 2>&1 || true
  systemctl restart ai2050-distributed
  sleep 4
  systemctl is-active --quiet ai2050-distributed \
    && ok "ai2050-distributed 运行中" \
    || warn "ai2050-distributed 未运行（不影响主站；journalctl -u ai2050-distributed 查看）"
else
  step "跳过分布式平台（:5010）"
fi

# ── 7. nginx ─────────────────────────────────────────────────
step "配置 nginx"
if command -v nginx >/dev/null; then
  if ! rpm -q nginx >/dev/null 2>&1; then
    dnf install -y nginx || warn "nginx 安装失败，请手动处理"
  fi
  CONF_SRC="$REPO_DIR/deploy/nginx-ai2050.conf"
  [ -f "$CONF_SRC" ] || die "缺 $CONF_SRC"
  if [ -f /etc/nginx/conf.d/ai2050.conf ]; then
    cp /etc/nginx/conf.d/ai2050.conf "/etc/nginx/conf.d/ai2050.conf.bak.$(date +%Y%m%d%H%M%S)"
    ok "已备份旧配置"
  fi
  sed "s#platform\.ai2050\.com#${DOMAIN}#g" "$CONF_SRC" > /etc/nginx/conf.d/ai2050.conf
  nginx -t && systemctl reload nginx && ok "nginx 配置生效"
  systemctl enable nginx >/dev/null 2>&1 || true
else
  die "无 nginx，无法托管前端"
fi

# ── 8. 防火墙 ───────────────────────────────────────────────
step "开放端口"
if command -v firewall-cmd >/dev/null; then
  firewall-cmd --permanent --add-port=80/tcp >/dev/null 2>&1 || true
  [ "$DO_HTTPS" = "1" ] && firewall-cmd --permanent --add-port=443/tcp >/dev/null 2>&1 || true
  firewall-cmd --reload >/dev/null 2>&1 || true
  ok "firewalld: 80/443 已开放"
fi
if command -v semanage >/dev/null; then
  semanage port -a -t http_port_t -p tcp 80 >/dev/null 2>&1 || true
fi

# ── 9. HTTPS（certbot）────────────────────────────────────────
TLS_OK=0
if [ "$DO_HTTPS" = "1" ]; then
  step "HTTPS"
  if command -v certbot >/dev/null; then
    PUBLIC_IP="$(curl -fsS --max-time 10 https://api.ipify.org 2>/dev/null || true)"
    if [ -n "$PUBLIC_IP" ]; then
      warn "公网 IP：$PUBLIC_IP —— 请确认 ${DOMAIN} 的 A 记录已指向该 IP"
    fi
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@${DOMAIN}" --redirect \
      || warn "certbot 失败（可稍后手动重跑：certbot --nginx -d ${DOMAIN}）"
    systemctl is-active --quiet nginx && TLS_OK=1
  else
    warn "未安装 certbot；如需 HTTPS：dnf install -y certbot python3-certbot-nginx && certbot --nginx -d ${DOMAIN}"
  fi
else
  warn "已跳过 HTTPS（--no-https）"
fi

# ── 10. 验证 ─────────────────────────────────────────────────
step "验证"
LOCAL_FAIL=0
check() { # check <名称> <URL> <期望码>
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$2" || echo 000)"
  if [ "$code" = "$3" ]; then ok "$1 → $code"; else warn "$1 → $code（期望 $3）"; LOCAL_FAIL=1; fi
}
check "本机 /health"        "http://127.0.0.1:${SERVER_PORT}/health" 200
check "本机 /healthz(nginx)" "http://127.0.0.1/healthz"              200
check "本机 前端 /"          "http://127.0.0.1/"                     200
check "本机 图谱 registry"   "http://127.0.0.1/api/ai-rnd/atlas/registry" 200

PUBLIC_HOST="$([ "$TLS_OK" = "1" ] && echo "https://${DOMAIN}" || echo "http://${DOMAIN}")"
check "公网 前端 /"          "${PUBLIC_HOST}/" 200
check "公网 /healthz"        "${PUBLIC_HOST}/healthz" 200

echo
echo "${BOLD}════════════════════════════════════════════════════${RST}"
echo "${BOLD} 部署完成${RST}"
echo "${BOLD}════════════════════════════════════════════════════${RST}"
echo "  站点      ${PUBLIC_HOST}"
echo "  中心服务  http://127.0.0.1:${SERVER_PORT}   (systemd: ai2050-server)"
echo "  前端文件  ${WEB_ROOT}"
echo "  数据目录  ${DATA_DIR}"
echo "  管理令牌  /etc/ai2050/admin_token"
echo "  查看日志  journalctl -u ai2050-server -f"
echo
echo "${BOLD}后续建议${RST}"
echo "  ① 确认 ${DOMAIN} 的 A 记录指向本机公网 IP"
echo "  ② 图谱数据为空？实验数据被 .gitignore 排除，需另行同步："
echo "     rsync -a 本机/research/gpt5/atlas/ ${REPO_DIR}/research/gpt5/atlas/"
echo "     （不同步时界面显示「未接入」空态，属设计行为，不是故障）"
echo "  ③ SELinux 拒绝时：semanage port -a -t http_port_t -p tcp 80"
echo "  ④ 节点接入：python deploy/node_agent.py --server http://${DOMAIN}:${DIST_PORT} register --name <节点名>"
exit 0
