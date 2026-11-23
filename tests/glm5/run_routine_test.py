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
from deploy.test_archive_client import Client, save as save_archive, cleanup_scratch, runtime_identity
from shared.dist_protocol.test_archive import validate_plan, digest as spec_digest


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


def run(args, *, archive_client=None, on_preflight=None, on_registered=None, registered_record=None) -> int:
    if not args.archive_plan:
        raise ValueError('科研测试必须先查询服务器：请提供 --archive-plan、--archive-reason；未查询不得启动脚本')
    if args.archive_decision == 'analysis_only':
        raise ValueError('analysis_only 请使用已归档产物和分析工具；模型执行入口不得借此运行前向')
    if not args.archive_reason.strip():
        raise ValueError('必须提供 --archive-reason，说明新增价值、独立复现或重采理由')
    script, questions = args.script.resolve(strict=True), args.questions.resolve(strict=True)
    if script.suffix != '.py' or not script.is_relative_to(ROOT / 'tests' / 'glm5'):
        raise ValueError('脚本应位于 tests/glm5，使用 AI2050_TEST_WORK_DIR 写入本轮产物')
    if not math.isfinite(args.peak_gib) or args.peak_gib * GIB < 1:
        raise ValueError('--peak-gib 必须大于零')
    plan = json.loads(args.archive_plan.read_text(encoding='utf-8-sig'))
    validate_plan(plan)
    if plan.get('entrypoint_sha256') != digest(script) or plan['dataset'].get('questions_file_sha256') != digest(questions):
        raise ValueError('归档方案的脚本或题目文件哈希不匹配')
    if plan.get('execution_args') != args.script_args:
        raise ValueError('归档方案必须包含完全一致的 execution_args')
    runtime = runtime_identity()
    if plan['acquisition'].get('entrypoint_sha256') != digest(script) or plan['acquisition'].get('questions_file_sha256') != digest(questions):
        raise ValueError('采集键必须包含实际 entrypoint_sha256 和 questions_file_sha256')
    if plan['acquisition']['generation'].get('script_args') != args.script_args or plan['acquisition']['environment_sha256'] != spec_digest(runtime):
        raise ValueError('采集键的参数或实际 Python/依赖环境指纹不匹配')
    for flag, key in [('model_snapshot', 'model_snapshot'), ('tokenizer_sha256', 'tokenizer_sha256'), ('seed', 'seed')]:
        value = getattr(args, flag)
        if value is not None and value != plan['acquisition'][key]:
            raise ValueError('运行参数与冻结方案不一致：' + flag)
        setattr(args, flag, plan['acquisition'][key])
    client = archive_client if archive_client is not None else Client(args.center, args.token_file)
    # A saved run can resume here only if its child has never produced an output log.
    if registered_record is not None:
        record = Path(registered_record).resolve(strict=True)
        records = checked_dir(ROOT / 'tests/glm5/result/routine', operation='复核记录')
        if not record.is_relative_to(records.resolve()):
            raise ValueError('恢复记录超出本机科研记录目录')
        prior = json.loads((record / 'review.json').read_text(encoding='utf-8'))
        stored = json.loads((record / 'protocol.json').read_text(encoding='utf-8'))
        preflight = json.loads((record / 'preflight.json').read_text(encoding='utf-8'))
        work = Path(prior['scratch']).resolve(strict=True)
        run_id = prior['server_run_id']
        files = {'script': record / 'test.py', 'questions': record / ('questions' + questions.suffix)}
        if (stored != plan or preflight.get('plan_sha256') != spec_digest(plan)
                or not work.is_relative_to(storage_dir('scratch', 'routine').resolve())
                or (work / 'output.log').exists() or prior['script'] != str(script.relative_to(ROOT))
                or digest(files['script']) != digest(script) or digest(files['questions']) != digest(questions)):
            raise ValueError('无法恢复：登记材料不一致或模型已经启动')
    else:
        # Network/auth/preflight failure happens before creating scratch or starting any child.
        preflight = client.preflight(plan)
    if on_preflight is not None:
        on_preflight(preflight)
    print(json.dumps({'preflight_id': preflight['preflight_id'], 'matches': preflight['matches']}, ensure_ascii=False), flush=True)
    budget = int(args.peak_gib * GIB)
    if registered_record is None:
        keep_bytes = script.stat().st_size + questions.stat().st_size + 1024 * 1024
        keep_root = checked_dir(ROOT / 'tests/glm5/result/routine', keep_bytes, operation='复核记录')
        scratch = checked_dir(storage_dir('scratch', 'routine'), budget, operation='日常测试预算')
        work = Path(tempfile.mkdtemp(prefix='run-', dir=scratch)).resolve()
        record = keep_root / work.name
        record.mkdir()
        files = {'script': record / 'test.py', 'questions': record / ('questions' + questions.suffix)}
        shutil.copyfile(script, files['script'])
        shutil.copyfile(questions, files['questions'])
        run_id = 'routine-' + work.name + '-' + str(time.time_ns())
    else:
        check_write(work, budget, operation='恢复已登记的科研测试')
    registration = client.start(preflight, run_id, args.archive_decision, args.archive_reason, args.archive_source_run_id)
    if (registration.get('id') != run_id or registration.get('preflight_id') != preflight['preflight_id']
            or registration.get('state') != 'running'):
        raise ValueError('中心运行登记身份与预检不一致')
    save_archive(record / 'preflight.json', preflight)
    save_archive(record / 'protocol.json', plan)
    archive_secrets = {'AI2050_RESEARCH_ADMIN_TOKEN', 'AI2050_DIST_ADMIN_TOKEN', 'AI2050_RESEARCH_SESSION'}
    env = process_env(work, base={k:v for k,v in os.environ.items() if k not in archive_secrets})
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
    result.update(server_run_id=run_id, archive_state='pending')
    result['runtime'] = runtime
    result_file = record / 'review.json'

    def save_review():
        draft = result_file.with_suffix('.tmp')
        check_write(draft, operation='复核记录')
        draft.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        draft.replace(result_file)

    save_review()
    if on_registered is not None:
        on_registered(registration, record)
    log = work / 'output.log'
    start = time.monotonic()
    last_progress = start
    blocked = None
    proc = None
    with log.open('xb') as output:
        try:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(files['script']), *args.script_args],
                                    cwd=work, env=env, stdout=output, stderr=subprocess.STDOUT)
            print(f'日常测试已启动 · {record} · 临时目录 {work}', flush=True)
        except OSError as exc:
            blocked = '脚本启动失败：' + str(exc)
            output.write((blocked + '\n').encode('utf-8'))
        while proc is not None:
            try:
                proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                try:
                    if time.monotonic() - last_progress >= 60:
                        with log.open('ab') as progress:
                            progress.write(f'\n[wrapper] running elapsed={time.monotonic()-start:.0f}s; child must report batch progress/GPU usage\n'.encode())
                        last_progress = time.monotonic()
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
    exit_code = proc.returncode if proc is not None else -1
    result.update(returncode=exit_code, elapsed_seconds=round(time.monotonic() - start, 2),
                  stop_reason=blocked, status='finished' if exit_code == 0 and 'conclusion' in result and
                  result['conclusion'] is not None and blocked is None else 'incomplete')
    save_review()
    confirmed = json.loads(result_file.read_text(encoding='utf-8'))
    if confirmed != result or confirmed['sha256'] != {key: digest(file) for key, file in files.items()}:
        raise RuntimeError('复核记录校验失败，保留本轮临时文件')
    # Upload full logs and outputs, including failed/interrupted runs. Unknown files stay local.
    for name, source in [('test.py', files['script']), (files['questions'].name, files['questions'])]:
        check_write(work / name, source.stat().st_size, operation='核心归档文件')
        shutil.copyfile(source, work / name)
    save_archive(work / 'protocol.json', plan)
    save_archive(work / 'reproducibility.json', result)
    entries = [
        {'path': 'test.py', 'role': 'script', 'retention': 'core'},
        {'path': files['questions'].name, 'role': 'questions', 'retention': 'core'},
        {'path': 'protocol.json', 'role': 'protocol', 'retention': 'core'},
        {'path': 'reproducibility.json', 'role': 'reproducibility', 'retention': 'core'},
        {'path': 'output.log', 'role': 'log', 'retention': 'core'},
    ]
    for name, role in [('conclusion.json', 'results'), ('case_results.jsonl', 'case_results')]:
        if (work / name).is_file():
            entries.append({'path': name, 'role': role, 'retention': 'core'})
    extras = work / 'artifacts.json'
    if extras.is_file():
        extra = json.loads(extras.read_text(encoding='utf-8-sig'))
        if extra.get('schema') != 'research_test_files.v1':
            raise ValueError('artifacts.json 格式错误，保留全部本地文件')
        entries.extend(extra['files'])
    manifest = work / 'archive-files.json'
    save_archive(manifest, {'schema': 'research_test_files.v1', 'files': entries})
    try:
        receipt = client.submit(run_id, manifest,
                                'interrupted' if blocked else 'completed' if result['status'] == 'finished' else 'failed',
                                json.dumps(result.get('conclusion') or {'stop_reason': blocked, 'returncode': exit_code}, ensure_ascii=False))
        save_archive(record / 'server-receipt.json', receipt)
        result['archive_state'] = receipt['archive_state']
        if receipt.get('cleanup_allowed'):
            result['cleanup'] = cleanup_scratch(manifest, receipt)
        else:
            result['status'] = 'incomplete'
    except (ValueError, OSError, RuntimeError) as exc:
        result.update(archive_state='pending_upload', archive_error=str(exc), status='incomplete')
    save_review()
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
    parser.add_argument('--archive-plan', type=Path)
    parser.add_argument('--archive-decision', choices=['new_acquisition', 'analysis_only', 'independent_replication', 'recapture'], default='new_acquisition')
    parser.add_argument('--archive-reason', default='')
    parser.add_argument('--archive-source-run-id', default='')
    parser.add_argument('--center', default='https://server.ai2050.com')
    parser.add_argument('--token-file', type=Path)
    args, script_args = parser.parse_known_args()
    if script_args and script_args[0] != '--':
        parser.error('用 -- 分隔传给测试脚本的参数')
    args.script_args = script_args[1:]
    try:
        return run(args)
    except StorageBlocked as error:
        stop_with_notice(error)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    raise SystemExit(main())
