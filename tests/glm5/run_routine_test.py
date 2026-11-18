"""Run a new, scratch-aware daily test and keep a small replay record."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import (GIB, StorageBlocked, check_write, checked_dir,
                                   process_env, stop_with_notice, storage_dir)


def digest(file: Path) -> str:
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkout_revision() -> str | None:
    try:
        git = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'],
                             cwd=ROOT, capture_output=True, text=True, timeout=5)
        return git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run(args) -> int:
    script, questions = args.script.resolve(strict=True), args.questions.resolve(strict=True)
    if script.suffix != '.py' or not script.is_relative_to(ROOT / 'tests' / 'glm5'):
        raise ValueError('脚本应位于 tests/glm5，使用 AI2050_TEST_WORK_DIR 写入本轮产物')
    if not math.isfinite(args.peak_gib) or args.peak_gib * GIB < 1:
        raise ValueError('--peak-gib 必须大于零')
    budget = int(args.peak_gib * GIB)
    keep_bytes = script.stat().st_size + questions.stat().st_size + 1024 * 1024
    keep_root = checked_dir(ROOT / 'tests/glm5/result/routine', keep_bytes, operation='复核记录')
    scratch = checked_dir(storage_dir('scratch', 'routine'), budget, operation='日常测试预算')
    work = Path(tempfile.mkdtemp(prefix='run-', dir=scratch)).resolve()
    record = keep_root / work.name
    record.mkdir()
    files = {'script': record / 'test.py', 'questions': record / ('questions' + questions.suffix)}
    shutil.copyfile(script, files['script'])
    shutil.copyfile(questions, files['questions'])
    env = process_env(work)
    env.update(AI2050_TEST_WORK_DIR=str(work), AI2050_TEST_QUESTIONS=str(files['questions']),
               AI2050_NATIVE_DIR='off', AI2050_OFFLOAD_DIR=str(work / 'offload'),
               AI2050_REUSABLE_DIR=str(storage_dir('reusable')),
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(script.parent), env.get('PYTHONPATH')])))
    result = {'schema': 'ai2050_routine_test.v1', 'script': str(script.relative_to(ROOT)),
              'sha256': {key: digest(file) for key, file in files.items()}, 'args': args.script_args,
              'started_at': datetime.now().astimezone().isoformat(), 'python': sys.version,
              'checkout_head': checkout_revision(), 'retained_files': {key: file.name for key, file in files.items()},
              'model_snapshot': args.model_snapshot, 'tokenizer_sha256': args.tokenizer_sha256,
              'seed': args.seed, 'peak_budget_bytes': budget, 'scratch': str(work)}
    result_file = record / 'review.json'

    def save_review():
        draft = result_file.with_suffix('.tmp')
        check_write(draft, operation='复核记录')
        draft.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        draft.replace(result_file)

    save_review()
    log = work / 'output.log'
    start = time.monotonic()
    blocked = None
    with log.open('wb') as output:
        proc = subprocess.Popen([sys.executable, '-B', '-u', str(files['script']), *args.script_args],
                                cwd=work, env=env, stdout=output, stderr=subprocess.STDOUT)
        print(f'日常测试已启动 · {record} · 临时目录 {work}', flush=True)
        while True:
            try:
                proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                try:
                    check_write(work, operation='日常测试继续写入')
                    current = sum(file.stat().st_size for file in work.rglob('*') if file.is_file())
                    if current > budget:
                        raise StorageBlocked('budget', work, 0, '本轮临时文件超过声明预算')
                except (StorageBlocked, OSError) as exc:
                    blocked = str(exc)
                    proc.terminate()
                    proc.wait()
                    break
            except KeyboardInterrupt:
                blocked = '用户中止'
                proc.terminate()
                proc.wait()
                break

    conclusion = work / 'conclusion.json'
    if conclusion.is_file() and conclusion.resolve().parent == work and conclusion.stat().st_size <= 512 * 1024:
        try:
            result['conclusion'] = json.loads(conclusion.read_text(encoding='utf-8-sig'))
        except ValueError:
            result['conclusion'] = None
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 16384))
        result['log_tail'] = stream.read().decode('utf-8', errors='replace')
    actual_bytes = sum(file.stat().st_size for file in work.rglob('*') if file.is_file())
    if actual_bytes > budget and blocked is None:
        blocked = f'本轮临时文件超过声明预算（{actual_bytes} > {budget} bytes）'
    result.update(returncode=proc.returncode, elapsed_seconds=round(time.monotonic() - start, 2),
                  stop_reason=blocked, status='finished' if proc.returncode == 0 and 'conclusion' in result and
                  result['conclusion'] is not None and blocked is None else 'incomplete')
    save_review()
    confirmed = json.loads(result_file.read_text(encoding='utf-8'))
    if confirmed != result or confirmed['sha256'] != {key: digest(file) for key, file in files.items()}:
        raise RuntimeError('复核记录校验失败，保留本轮临时文件')
    # Only the exact directory created for this invocation is disposable.
    if work.resolve() != work or work.parent != scratch.resolve():
        raise RuntimeError('本轮临时目录身份已改变，保留供核查')
    shutil.rmtree(work)
    print(f"{result['status']} · {result_file}", flush=True)
    return 0 if result['status'] == 'finished' else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--peak-gib', type=float, required=True)
    parser.add_argument('--model-snapshot')
    parser.add_argument('--tokenizer-sha256')
    parser.add_argument('--seed', type=int)
    args, script_args = parser.parse_known_args()
    if script_args and script_args[0] != '--':
        parser.error('用 -- 分隔传给测试脚本的参数')
    args.script_args = script_args[1:]
    try:
        return run(args)
    except StorageBlocked as error:
        stop_with_notice(error)


if __name__ == '__main__':
    raise SystemExit(main())
