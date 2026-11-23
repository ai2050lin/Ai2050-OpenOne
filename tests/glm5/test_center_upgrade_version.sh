#!/usr/bin/env bash
# Exercise both release scripts on the real tarballs with local rsync and simulated systemd.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
old="$repo/release/model-search-3d-center/ai2050-center-1.0.45.tar.gz"
new="$repo/release/goodfire-native-replay-center/ai2050-center-1.0.46.tar.gz"
corrected="$repo/release/goodfire-native-replay-center/upgrade_center.sh"
root="$repo/tests/glm5_temp/center_upgrade_version"
result="$repo/tests/glm5/result/center_upgrade_version"
mkdir -p "$root" "$result"
scratch=$(mktemp -d "$root/replay.XXXXXX")
mkdir "$scratch/bin"
python=${1:?provide an absolute path to the test Python interpreter}
command -v rsync >/dev/null

cat > "$scratch/bin/systemctl" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  show)
    case "$3" in
      FragmentPath) printf '%s\n' "$TEST_UNIT";;
      WorkingDirectory) printf '%s\n' "$TEST_APP";;
      Environment) printf 'AI2050_DIST_DIR=%s\n' "$TEST_DATA";;
      User|Group) printf 'root\n';;
      *) exit 3;;
    esac;;
  cat) printf 'ExecStart=%s deploy/center_app.py --port 5010\n' "$TEST_PY";;
  restart|daemon-reload) exit 0;;
  is-failed) exit 1;;
  *) exit 3;;
esac
EOF
cat > "$scratch/bin/curl" <<'EOF'
#!/usr/bin/env bash
version=$(tr -d '[:space:]' < "$TEST_APP/VERSION")
if [ "$TEST_MODE" = bad_health ] && [ "$version" = 1.0.46 ]; then version=1.0.44; fi
printf '{"service":"ai2050-center","version":"%s","data_dir_in_release":false}\n' "$version"
EOF
printf '#!/usr/bin/env bash\necho 0\n' > "$scratch/bin/id"
for cmd in chown journalctl sleep; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$scratch/bin/$cmd"
done
chmod +x "$scratch/bin/"*
export PATH="$scratch/bin:$PATH" TEST_PY="$python"
tar -xOzf "$new" 'ai2050-center-1.0.46/deploy/upgrade_center.sh' > "$scratch/original.sh"
tar -xOzf "$new" 'ai2050-center-1.0.46/VERSION' > "$scratch/new-version"
tar -xOzf "$new" 'ai2050-center-1.0.46/deploy/platform_service.py' > "$scratch/new-platform-service"
tar -xOzf "$old" 'ai2050-center-1.0.45/deploy/platform_service.py' > "$scratch/old-platform-service"

prepare() {
  local sandbox="$scratch/$1"
  mkdir -p "$sandbox/data" "$sandbox/tmp"
  tar -xzf "$old" -C "$sandbox"
  mv "$sandbox/ai2050-center-1.0.45" "$sandbox/app"
  export TEST_APP="$sandbox/app" TEST_UNIT="$sandbox/unit" TEST_DATA="$sandbox/data"
  export BACKUP_DIR="$sandbox/backup" TMPDIR="$sandbox/tmp" SERVICE=ai2050-center-qa-upgrade HEALTH_TIMEOUT=5
  printf 'fixture unit\n' > "$TEST_UNIT"
  printf 'preserve\n' > "$TEST_DATA/marker"
  mkdir -p "$TEST_APP/data/lang_bench"
  printf 'bench\n' > "$TEST_APP/data/lang_bench/questions.qa.jsonl"
}

prepare original
export TEST_MODE=normal
if bash "$scratch/original.sh" "$new" > "$result/original.log" 2>&1; then
  echo 'Expected the original release upgrade to fail' >&2
  exit 1
fi
grep -q 'health 报告的版本不是 1.0.46' "$result/original.log"
test "$(cat "$TEST_APP/VERSION")" = '1.0.45'
cmp -s "$scratch/old-platform-service" "$TEST_APP/deploy/platform_service.py"
echo 'Original upgrade reproduced the version mismatch and returned to 1.0.45.'

prepare corrected
bash "$corrected" "$new" > "$result/corrected.log" 2>&1
cmp -s "$scratch/new-version" "$TEST_APP/VERSION"
cmp -s "$scratch/new-platform-service" "$TEST_APP/deploy/platform_service.py"
test "$(cat "$TEST_DATA/marker")" = preserve
test "$(cat "$TEST_APP/data/lang_bench/questions.qa.jsonl")" = bench
echo 'Corrected upgrade served 1.0.46 and preserved external data and test questions.'

prepare rollback
export TEST_MODE=bad_health
if bash "$corrected" "$new" > "$result/rollback.log" 2>&1; then
  echo 'Expected failure when the new service reports the wrong version' >&2
  exit 1
fi
grep -q '已回退，服务正常（1.0.45）' "$result/rollback.log"
test "$(cat "$TEST_APP/VERSION")" = '1.0.45'
cmp -s "$scratch/old-platform-service" "$TEST_APP/deploy/platform_service.py"
test "$(cat "$TEST_DATA/marker")" = preserve
echo 'Corrected rollback restored the exact old version file and program.'
