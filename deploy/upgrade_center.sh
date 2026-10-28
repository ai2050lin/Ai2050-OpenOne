#!/usr/bin/env bash
# AI2050 中心一键升级：把发布包和本脚本放在同一目录（如 /data），以 root 运行：
#
#     sudo bash upgrade_center.sh                     # 自动选同目录下版本号最高的 ai2050-center-*.tar.gz
#     sudo bash upgrade_center.sh ai2050-center-1.0.14.tar.gz
#
# 依次：核对 sha256 → 从 systemd 读出当前程序目录、数据目录与端口 → 备份数据库 → 部署新版本 → 重启 →
# 等 /api/health 报告新版本。任何一步失败都自动回退到旧版本（程序与 systemd 配置；数据库没有被替换）。
#
# 两种布局自动识别：
#   版本目录  WorkingDirectory=/data/ai2050-center-<旧版本>：新版本解压到同一父目录，systemd 改指新目录，旧目录保留以便回退
#   固定目录  WorkingDirectory=/opt/ai2050/app（install_center.sh 的布局）：旧程序先复制为 app.bak-<旧版本>，再同步新程序
#
# 环境变量：SERVICE（缺省 ai2050-center）、BACKUP_DIR（缺省 /root/ai2050-backups）、HEALTH_TIMEOUT（秒，缺省 90）。
set -euo pipefail

SERVICE=${SERVICE:-ai2050-center}
BACKUP_DIR=${BACKUP_DIR:-/root/ai2050-backups}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-90}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TS=$(date +%Y%m%d-%H%M%S)

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "请用 root 运行：sudo bash $0"
command -v systemctl >/dev/null || die "需要 systemd"
command -v curl >/dev/null || die "需要 curl"

# ---------- 1. 发布包 ----------
step "发布包"
if [ $# -ge 1 ]; then
  PKG=$(readlink -f "$1")
else
  PKG=$(ls -1 "$HERE"/ai2050-center-*.tar.gz 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$PKG" ] || die "在 $HERE 没找到 ai2050-center-*.tar.gz；把发布包放在本脚本旁边，或把路径作为参数传入"
fi
[ -f "$PKG" ] || die "找不到 $PKG"
NAME=$(basename "$PKG" .tar.gz)
NEW_VER=${NAME#ai2050-center-}
[[ "$NEW_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "文件名应为 ai2050-center-<x.y.z>.tar.gz：$PKG"
if [ -f "$PKG.sha256" ]; then
  want=$(awk '{print $1}' "$PKG.sha256")
  got=$(sha256sum "$PKG" | awk '{print $1}')
  [ "$want" = "$got" ] || die "sha256 不一致：发布包可能没传完整（期望 ${want:0:16}…，实际 ${got:0:16}…）"
  ok "$NAME（sha256 已核对）"
else
  warn "$NAME（没有 .sha256 文件，跳过核对）"
fi

# ---------- 2. 当前部署 ----------
step "当前部署（systemd: $SERVICE）"
UNIT=$(systemctl show -p FragmentPath --value "$SERVICE")
[ -n "$UNIT" ] && [ -f "$UNIT" ] || die "找不到 systemd 服务 $SERVICE（可用 SERVICE=<名称> 指定）"
OLD_DIR=$(systemctl show -p WorkingDirectory --value "$SERVICE")
[ -n "$OLD_DIR" ] && [ -d "$OLD_DIR" ] || die "服务的 WorkingDirectory 无效：'$OLD_DIR'"
OLD_DIR_RAW=${OLD_DIR%/}                     # systemd 配置里的写法（可能经过符号链接），改配置时按它替换
OLD_DIR=$(readlink -f "$OLD_DIR")
ENVS=$(systemctl show -p Environment --value "$SERVICE")
envval() { tr ' ' '\n' <<<"$ENVS" | sed -n "s/^$1=//p" | tail -1; }
DATA_DIR=$(envval AI2050_DIST_DIR)
[ -n "$DATA_DIR" ] || DATA_DIR="$OLD_DIR/distributed_data"
DATA_DIR=$(readlink -f "$DATA_DIR")
EXEC=$(systemctl cat "$SERVICE" | sed -n 's/^ExecStart=//p' | tail -1)
PY=$(awk '{print $1}' <<<"$EXEC")
PORT=$(grep -oE -- '--port[ =][0-9]+' <<<"$EXEC" | grep -oE '[0-9]+' || true)
PORT=${PORT:-5010}
HEALTH="http://127.0.0.1:$PORT/api/health"
OLD_VER=$(curl -fsS -m 5 "$HEALTH" 2>/dev/null | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' || true)
[ -n "$OLD_VER" ] || OLD_VER=$(tr -d '[:space:]' < "$OLD_DIR/VERSION" 2>/dev/null || echo unknown)
SVC_USER=$(systemctl show -p User --value "$SERVICE"); SVC_USER=${SVC_USER:-root}
SVC_GROUP=$(systemctl show -p Group --value "$SERVICE"); SVC_GROUP=${SVC_GROUP:-$SVC_USER}
ok "程序目录 $OLD_DIR（版本 $OLD_VER）"
ok "数据目录 $DATA_DIR"
ok "解释器 $PY，端口 $PORT，运行用户 $SVC_USER"

[ "$OLD_VER" != "$NEW_VER" ] || die "当前已经是 $NEW_VER，无需升级"
[ -x "$PY" ] || die "ExecStart 的解释器不存在：$PY"
case "$DATA_DIR/" in
  "$OLD_DIR"/*) die "数据目录 $DATA_DIR 在程序目录之内，换目录会丢数据。先在 systemd 里设置 AI2050_DIST_DIR 指向程序目录之外并迁移数据（见 deploy/README_DEPLOY.md 2.1）";;
esac
[ -f "$DATA_DIR/dist.sqlite3" ] || warn "数据目录里还没有 dist.sqlite3（新中心？）"

LAYOUT=fixed
if [[ "$(basename "$OLD_DIR")" =~ ^ai2050-center-[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  LAYOUT=versioned
  NEW_DIR="$(dirname "$OLD_DIR")/$NAME"
  NEW_DIR_RAW="$(dirname "$OLD_DIR_RAW")/$NAME"
  case "$PY" in "$OLD_DIR"/*) die "解释器在旧版本目录里（$PY），换目录后会失效；请先把 Python 环境放到版本目录之外";; esac
  [ ! -e "$NEW_DIR" ] || die "$NEW_DIR 已存在：确认后删除或改名再运行"
else
  NEW_DIR="$OLD_DIR"
fi
ok "布局：$([ $LAYOUT = versioned ] && echo "版本目录 → $NEW_DIR" || echo "固定目录（原地同步，旧程序备份为 $OLD_DIR.bak-$OLD_VER）")"

# ---------- 3. 备份 ----------
step "备份"
mkdir -p "$BACKUP_DIR"; chmod 700 "$BACKUP_DIR"
if [ -f "$DATA_DIR/dist.sqlite3" ]; then
  DB_BAK="$BACKUP_DIR/dist-$OLD_VER-$TS.sqlite3"
  "$PY" - "$DATA_DIR/dist.sqlite3" "$DB_BAK" <<'EOF'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1], timeout=60)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
dst.close(); src.close()
EOF
  ok "数据库 → $DB_BAK（$(du -h "$DB_BAK" | cut -f1)，完整性已检查）"
fi
UNIT_FILES=("$UNIT")
for f in /etc/systemd/system/"$SERVICE".service.d/*.conf; do [ -f "$f" ] && UNIT_FILES+=("$f"); done
UNIT_BAK="$BACKUP_DIR/systemd-$OLD_VER-$TS"
mkdir -p "$UNIT_BAK"
for f in "${UNIT_FILES[@]}"; do cp -p "$f" "$UNIT_BAK/$(echo "$f" | tr '/' '_')"; done
ok "systemd 配置 → $UNIT_BAK"

# ---------- 4. 部署 ----------
step "部署 $NEW_VER"
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/ai2050-upgrade.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
tar xzf "$PKG" -C "$STAGE"
SRC="$STAGE/$NAME"
[ -f "$SRC/deploy/center_app.py" ] || die "发布包内容不完整：缺 deploy/center_app.py"
PKG_VER=$(tr -d '[:space:]' < "$SRC/VERSION" 2>/dev/null || true)      # 容忍 CRLF 与行尾空白
[ "$PKG_VER" = "$NEW_VER" ] || die "发布包里的 VERSION 是 '$PKG_VER'，与文件名的 $NEW_VER 不符"

rollback() {
  printf '\n\033[31m✗ %s，回退到 %s\033[0m\n' "$1" "$OLD_VER" >&2
  for f in "${UNIT_FILES[@]}"; do cp -p "$UNIT_BAK/$(echo "$f" | tr '/' '_')" "$f"; done
  if [ $LAYOUT = fixed ] && [ -d "$OLD_DIR.bak-$OLD_VER" ]; then
    rsync -a --delete "$OLD_DIR.bak-$OLD_VER/" "$OLD_DIR/"
  fi
  systemctl daemon-reload
  systemctl restart "$SERVICE" || true
  sleep 3
  echo "--- 新版本启动日志（最后 40 行）---" >&2
  journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
  if curl -fsS -m 5 "$HEALTH" >/dev/null 2>&1; then
    echo "已回退，服务正常（$OLD_VER）。新版本目录与备份保留，便于排查。" >&2
  else
    echo "回退后服务仍未响应：systemctl status $SERVICE；数据库备份在 ${DB_BAK:-（无）}" >&2
  fi
  exit 1
}

if ! cmp -s "$OLD_DIR/deploy/requirements-center.txt" "$SRC/deploy/requirements-center.txt"; then
  "$(dirname "$PY")/pip" install -q -r "$SRC/deploy/requirements-center.txt" || die "依赖安装失败（尚未改动服务）"
  ok "依赖已按新 requirements-center.txt 更新"
fi

if [ $LAYOUT = versioned ]; then
  mv "$SRC" "$NEW_DIR"
  chown -R "$SVC_USER:$SVC_GROUP" "$NEW_DIR"
  for f in "${UNIT_FILES[@]}"; do sed -i "s#$OLD_DIR_RAW#$NEW_DIR_RAW#g" "$f"; done
  grep -q "$NEW_DIR_RAW" "${UNIT_FILES[@]}" || rollback "systemd 配置里没找到 $OLD_DIR_RAW，无法改指新目录"
  ok "程序 → $NEW_DIR；systemd 已改指新目录"
else
  command -v rsync >/dev/null || die "需要 rsync"
  rm -rf "$OLD_DIR.bak-$OLD_VER"
  cp -a "$OLD_DIR" "$OLD_DIR.bak-$OLD_VER"
  rsync -a --delete --exclude '__pycache__' "$SRC/" "$OLD_DIR/"
  chown -R "$SVC_USER:$SVC_GROUP" "$OLD_DIR"
  ok "程序已同步到 $OLD_DIR（旧程序 → $OLD_DIR.bak-$OLD_VER）"
fi

# ---------- 5. 重启与检查 ----------
step "重启"
systemctl daemon-reload
systemctl restart "$SERVICE" || rollback "重启失败"
deadline=$((SECONDS + HEALTH_TIMEOUT))
body=''
while [ $SECONDS -lt $deadline ]; do
  body=$(curl -fsS -m 5 "$HEALTH" 2>/dev/null || true)
  [ -n "$body" ] && break
  systemctl is-failed -q "$SERVICE" && rollback "服务启动后退出"
  sleep 2
done
[ -n "$body" ] || rollback "${HEALTH_TIMEOUT} 秒内 $HEALTH 没有响应"
grep -q "\"version\": *\"$NEW_VER\"" <<<"$body" || rollback "health 报告的版本不是 $NEW_VER：$body"
grep -q '"data_dir_in_release": *false' <<<"$body" || warn "health 报告 data_dir_in_release 不是 false：$body"
ok "$HEALTH → $NEW_VER"

step "完成：$OLD_VER → $NEW_VER"
echo "  数据库备份：${DB_BAK:-（无）}"
echo "  systemd 备份：$UNIT_BAK"
if [ $LAYOUT = versioned ]; then
  echo "  旧版本目录保留：$OLD_DIR"
  echo "  回退：把 $UNIT_BAK 里的配置复制回 ${UNIT_FILES[*]}，再 systemctl daemon-reload && systemctl restart $SERVICE"
else
  echo "  旧程序备份：$OLD_DIR.bak-$OLD_VER"
  echo "  回退：rsync -a --delete $OLD_DIR.bak-$OLD_VER/ $OLD_DIR/ && systemctl restart $SERVICE"
fi
