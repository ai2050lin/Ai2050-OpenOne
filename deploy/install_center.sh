#!/usr/bin/env bash
# ============================================================================
#  AI2050 开放平台中心 · 一键安装（CentOS / Rocky / Alma 8+，Ubuntu / Debian 22+）
#
#  在解压后的发布包（或仓库根目录）中执行：
#     sudo DOMAIN=platform.example.org bash deploy/install_center.sh            # 交互确认
#     sudo DOMAIN=platform.example.org bash deploy/install_center.sh --yes      # 无人值守
#     sudo DOMAIN=platform.example.org bash deploy/install_center.sh --yes --https   # 另用 certbot 申请证书
#
#  结果：
#     /opt/ai2050/app            程序（发布包内容）       /opt/ai2050/venv   Python 环境
#     /var/lib/ai2050/center     中心数据（SQLite、结果、普查作业）
#     /var/lib/ai2050/native     普查语料与合并后的普查结果
#     /etc/ai2050/admin_token    管理令牌（600）           /etc/ai2050/platform_config.json  站点配置
#     systemd ai2050-center → 127.0.0.1:5010；nginx :80 反代（同源托管客户端与 API）
#
#  幂等：可重复执行（升级时覆盖程序，保留数据、令牌与站点配置）。
# ============================================================================
set -euo pipefail

DOMAIN="${DOMAIN:-}"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT=/opt/ai2050
APP="$ROOT/app"
VENV="$ROOT/venv"
DATA=/var/lib/ai2050
ETC=/etc/ai2050
USER_NAME=ai2050
PORT=5010
ASSUME_YES=0
DO_HTTPS=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --https) DO_HTTPS=1 ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg"; exit 2 ;;
  esac
done

step() { echo -e "\033[1m==>\033[0m $*"; }
ok()   { echo -e "  \033[32m✓\033[0m $*"; }
die()  { echo -e "  \033[31m✗\033[0m $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "请用 root 运行（sudo）"
[ -n "$DOMAIN" ] || die "请设置 DOMAIN=你的域名（或服务器 IP）"
[ -f "$SRC/deploy/center_app.py" ] || die "在发布包根目录运行（找不到 deploy/center_app.py）"
[ -f "$SRC/frontend/dist/index.html" ] || die "发布包缺少 frontend/dist（用 scripts/build_center_release.py 生成发布包）"
if [ "$ASSUME_YES" != 1 ]; then
  read -r -p "将安装到 $APP，域名 $DOMAIN，继续？[y/N] " a; [ "$a" = y ] || exit 1
fi

step "系统依赖"
if command -v dnf >/dev/null; then
  dnf install -y python3.11 python3.11-pip nginx rsync >/dev/null || dnf install -y python3 python3-pip nginx rsync >/dev/null
  PY=$(command -v python3.11 || command -v python3)
elif command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip nginx rsync >/dev/null
  PY=$(command -v python3)
else
  die "不支持的发行版（需要 dnf 或 apt）"
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 10)' || die "需要 Python 3.10+"
ok "Python $("$PY" -V)"

step "用户与目录"
id "$USER_NAME" >/dev/null 2>&1 || useradd --system --home "$ROOT" --shell /sbin/nologin "$USER_NAME"
mkdir -p "$APP" "$DATA/center" "$DATA/native/_census/corpus" "$ETC"
rsync -a --delete --exclude '.git' --exclude '__pycache__' "$SRC/" "$APP/"
chown -R "$USER_NAME:$USER_NAME" "$ROOT" "$DATA"
ok "程序 → $APP"

step "Python 环境"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$APP/deploy/requirements-center.txt"
ok "依赖已安装"

step "管理令牌与站点配置"
if [ ! -f "$ETC/admin_token" ]; then
  "$VENV/bin/python" -c 'import secrets;print(secrets.token_hex(24))' | tr -d '\n' > "$ETC/admin_token"
  ok "新令牌 → $ETC/admin_token"
fi
chmod 600 "$ETC/admin_token"
if [ ! -f "$ETC/platform_config.json" ]; then
  cp "$APP/deploy/platform_config.example.json" "$ETC/platform_config.json"
  ok "站点配置 → $ETC/platform_config.json（按需修改标题、联系方式、需要的模型）"
fi
chown root:"$USER_NAME" "$ETC/platform_config.json"; chmod 640 "$ETC/platform_config.json"

step "systemd"
TOKEN=$(cat "$ETC/admin_token")
SCHEME=http; [ "$DO_HTTPS" = 1 ] && SCHEME=https
sed -e "s#change-me-to-random-hex#${TOKEN}#" -e "s#https://platform.example.org#${SCHEME}://${DOMAIN}#" \
  "$APP/deploy/ai2050-center.service" > /etc/systemd/system/ai2050-center.service
chmod 600 /etc/systemd/system/ai2050-center.service
systemctl daemon-reload
systemctl enable --now ai2050-center >/dev/null
systemctl restart ai2050-center
ok "ai2050-center 已启动（127.0.0.1:$PORT）"

step "nginx"
sed -e "s#platform.example.org#${DOMAIN}#" "$APP/deploy/nginx-ai2050-center.conf" > /etc/nginx/conf.d/ai2050-center.conf
if command -v setsebool >/dev/null; then setsebool -P httpd_can_network_connect 1 || true; fi
nginx -t && systemctl enable --now nginx >/dev/null && systemctl reload nginx
ok "nginx → $DOMAIN"
if [ "$DO_HTTPS" = 1 ]; then
  command -v certbot >/dev/null || { command -v dnf >/dev/null && dnf install -y certbot python3-certbot-nginx >/dev/null || apt-get install -y -qq certbot python3-certbot-nginx >/dev/null; }
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect || echo "  证书申请失败：确认域名已解析到本机后重跑 certbot --nginx -d $DOMAIN"
fi
if command -v firewall-cmd >/dev/null; then firewall-cmd --permanent --add-service=http --add-service=https >/dev/null 2>&1 && firewall-cmd --reload >/dev/null 2>&1 || true; fi

step "自检"
for i in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/api/health" && echo || die "中心服务未响应：journalctl -u ai2050-center -n 100"
curl -sf -o /dev/null "http://127.0.0.1/" -H "Host: $DOMAIN" && ok "客户端首页 200" || echo "  ! nginx 未通：检查 /var/log/nginx/ai2050-center.error.log"
cat <<EOF

完成。下一步：
  1. 浏览器打开 ${SCHEME}://${DOMAIN} ，注册第一个账号（自动成为管理员）。
  2. 上传普查语料到 $DATA/native/_census/corpus/<指纹>/ 后建普查作业：
       sudo $VENV/bin/python $APP/deploy/admin_tool.py census-create --help
  3. 管理令牌：$ETC/admin_token（admin_tool.py 自动读取）；日志：journalctl -u ai2050-center -f
EOF
