#!/usr/bin/env bash
set -euo pipefail

SERVICE=${SERVICE:-ai2050-center}
HOST=${HOST:-server.ai2050.com}
PORT=${PORT:-5010}
LOCAL="http://127.0.0.1:$PORT"
PUBLIC="https://$HOST"

die() { printf '%s\n' "$*" >&2; exit 1; }
probe() { printf '%-58s ' "$1"; curl -sS --max-time 8 -o /dev/null -w 'HTTP %{http_code}\n' "$@" || true; }

if [[ ${1:-diagnose} == diagnose ]]; then
    systemctl show "$SERVICE" -p LoadState -p ActiveState -p SubState -p Result -p ExecMainStatus -p MainPID -p WorkingDirectory -p KillMode || true
    ss -ltnp "sport = :$PORT" || true
    probe "$LOCAL/api/health"
    probe "$LOCAL/api/lang/map"
    probe "$PUBLIC/api/health"
    probe "$PUBLIC/api/health" --resolve "$HOST:443:127.0.0.1"
    nginx -t || true
    journalctl -u "$SERVICE" -n 60 --no-pager || true
    for file in /var/log/nginx/ai2050-center.error.log /var/log/nginx/error.log; do
        if [[ -r $file ]]; then printf '\n%s\n' "$file"; tail -n 30 "$file"; fi
    done
    exit 0
fi

[[ $1 == refresh ]] || die "Usage: $0 [diagnose|refresh]"
[[ $(id -u) == 0 ]] || die 'Run refresh with sudo.'
curl -fsS --max-time 8 "$LOCAL/api/health" >/dev/null || die 'Restore the local center health endpoint before refreshing.'
curl -fsS --max-time 8 "$LOCAL/api/lang/map" >/dev/null || die 'The running center must expose the language map endpoint.'

app=$(systemctl show "$SERVICE" -p WorkingDirectory --value)
pid=$(systemctl show "$SERVICE" -p MainPID --value)
user=$(systemctl show "$SERVICE" -p User --value)
kill_mode=$(systemctl show "$SERVICE" -p KillMode --value)
[[ $kill_mode == control-group || $kill_mode == mixed ]] || die 'Use systemd KillMode=control-group or mixed to stop all package writers.'
[[ $pid =~ ^[1-9][0-9]+$ && -r /proc/$pid/environ ]] || die 'Cannot read the running service environment.'
read -r -d '' python < "/proc/$pid/cmdline" || die 'Cannot read the running service Python command.'
[[ -x $python && -r $app/deploy/snapshot_worker.py ]] || die 'Cannot find the Python environment or snapshot worker.'
[[ -n $user && $user != root ]] || die 'Refresh requires a non-root systemd service user.'

data=''
settings=()
while IFS= read -r -d '' item; do
    case $item in
        AI2050_DIST_DIR=*) data=${item#*=} ;;
        AI2050_NATIVE_DIR=*|AI2050_RESEARCH_DIR=*|AI2050_PLATFORM_CONFIG=*|AI2050_LANG_BENCH=*|AI2050_PUBLIC_URL=*) settings+=("$item") ;;
    esac
done < "/proc/$pid/environ"
[[ $data == /* && -f $data/dist.sqlite3 ]] || die 'Cannot confirm the existing center data directory from AI2050_DIST_DIR.'
data=$(readlink -f -- "$data")
backup=$(mktemp -d "$data/lang-map-pack-backup.XXXXXXXX")
pack_dir="$data/datapack"
files=(datapack.json.gz datapack.meta.json)

restore_and_start() {
    local status=$?
    trap - EXIT
    if [[ $rebuilding == 1 ]]; then
        for file in "${files[@]}"; do
            if [[ -f $backup/$file ]]; then
                cp -p -- "$backup/$file" "$pack_dir/$file" || status=1
            else
                rm -f -- "$pack_dir/$file" || status=1
            fi
        done
        printf 'Restored the previous main package files from %s\n' "$backup" >&2
    fi
    systemctl start "$SERVICE" || status=1
    exit "$status"
}
rebuilding=0
trap restore_and_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl stop "$SERVICE"
for file in "${files[@]}"; do
    if [[ -f $pack_dir/$file ]]; then cp -p -- "$pack_dir/$file" "$backup/$file"; fi
done
printf 'Existing main package backup: %s\n' "$backup"
rebuilding=1

cd "$app"
runuser -u "$user" -- env -i PATH=/usr/bin:/bin PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    AI2050_DIST_ADMIN_TOKEN=datapack-maintenance-local "AI2050_DIST_DIR=$data" "${settings[@]}" \
    "$python" deploy/snapshot_worker.py --datapack-only

"$python" - "$pack_dir" <<'PY'
import gzip, json, sys
from pathlib import Path
from deploy.lang_map_service import item_ids, map_ep, words_ep

path = Path(sys.argv[1])
with gzip.open(path / 'datapack.json.gz', 'rt', encoding='utf-8') as stream:
    pack = json.load(stream)
meta = json.loads((path / 'datapack.meta.json').read_text(encoding='utf-8'))
entries = pack['entries']
if pack['version'] != meta['version'] or entries.get('/api/lang/map') != map_ep():
    raise SystemExit('Main package version or language map is invalid.')
ids = item_ids()
if any(entries.get(f'/api/lang/map/words/{item}') != words_ep(item) for item in ids):
    raise SystemExit('The main package has missing or inconsistent word entries.')
print(f'Validated {len(ids)} word entries. Main package version: {pack["version"]}')
PY
rebuilding=0
systemctl start "$SERVICE"
trap - EXIT INT TERM
for _ in {1..30}; do
    if curl -fsS --max-time 4 "$LOCAL/api/health" >/dev/null 2>&1; then
        probe "$PUBLIC/api/health"
        probe "$PUBLIC/api/platform/config"
        exit 0
    fi
    sleep 2
done
die "Local center did not return HTTP 200 after restart. Run $0 diagnose."
