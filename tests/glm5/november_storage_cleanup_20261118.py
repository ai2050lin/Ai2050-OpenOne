"""Audited, explicitly scoped retirement of November A1/A2 recapturable fields.

No model execution, directory deletion, wildcard deletion, or model-file changes.
Plan and execute are separate commands. Historical manifests remain immutable.
"""
from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write, StorageBlocked

BASE = ROOT / 'tests/glm5/result'
OUT = BASE / 'november_research_storage_audit_20261118'
CSV = ROOT / 'tests/gpt5/result/d_drive_cleanup_plan_v2_20261117.csv'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
GIB = 2**30
ALLOWANCE = 10 * 2**20
APPROVAL = {
    'scope': 'Only A1-COLD-NEW and A2-COLD-SOURCE directories from the cited CSV',
    'choice': 'Keep scripts, questions, results and review records; delete recapturable raw fields',
    'maintenance_exception': 'User explicitly allowed <=10 MiB of maintenance records below the 20 GiB reserve; no model runs',
    'prior_retention': 'User choice supersedes the earlier raw-field review dates for this cleanup only',
}
A1 = (
    [f'recursive_branch14_20261113_v{i}' for i in range(1, 7)]
    + [f'branch_reuse14_20261113_v{i}' for i in range(1, 4)]
    + [f'b36_composition14_20261113_v{i}' for i in range(1, 4)]
    + ['b36_transition14_20261114_v1', 'own_state_explicit_20261115_v2',
       'goodfire_middle_20261113_v1', 'source_binding_langparts_v1',
       'source_binding_qkv_followup_v1', 'source_binding_enrelay_v1',
       'source_binding_downstream_v1', 'mlp_condition_v2']
)
A2 = ['source_formation_v1', 'source_binding_v1', 'source_relay_v2',
      'source_relay_v1', 'common_bench_b1_repair_v2', 'qwen14_mlp_b28_b39_v2']
NAMES = set(A1 + A2)
NATIVE_NAMES = {
    'attention', 'early', 'final_norm', 'gate', 'head_context', 'hidden',
    'key', 'kpre', 'mass', 'mlp_input', 'mlp_output', 'postattn', 'qpre',
    'query', 'source_mass', 'trace_postattn', 'trace_state', 'value', 'vproj',
}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp() -> str:
    # Host timezone is deliberately not used for research timestamps.
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-6))).isoformat(timespec='seconds')


def validated(path: Path, parent: Path = BASE) -> Path:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_relative_to(parent.resolve()):
        raise RuntimeError(f'Out of scope: {absolute}')
    probe = absolute
    while probe != parent.parent:
        st = probe.lstat()
        if probe.is_symlink() or getattr(st, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError(f'Reparse point rejected: {probe}')
        if probe == parent:
            break
        probe = probe.parent
    real = absolute.resolve(strict=True)
    if real != absolute or real.drive.upper() != ROOT.drive.upper():
        raise RuntimeError(f'Unexpected resolved path: {absolute} -> {real}')
    return real


def guard_write(path: Path, size: int) -> None:
    try:
        check_write(path, size, operation='本次授权清理审计')
    except StorageBlocked as exc:
        if exc.reason != 'space' or not ARGS.maintenance_exception_10mib:
            raise
        used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
        used += Path(__file__).stat().st_size
        if path.resolve().drive.upper() != 'D:' or used + size > ALLOWANCE:
            raise RuntimeError('Task-specific maintenance allowance exceeded') from exc


def save(name: str, value) -> None:
    target = OUT / name
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    guard_write(target, len(payload))
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def log(value: dict) -> None:
    line = json.dumps({'time': stamp(), **value}, ensure_ascii=False)
    print(line, flush=True)
    target = OUT / 'progress.jsonl'
    guard_write(target, len(line.encode('utf-8')) + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
        stream.flush()


def allocated(path: Path) -> int:
    if os.name != 'nt':
        return path.stat().st_blocks * 512
    fn = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    fn.restype = ctypes.c_ulong
    high = ctypes.c_ulong()
    ctypes.set_last_error(0)
    low = fn(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return int(low) + (high.value << 32)


def role(run: str, rel: Path) -> tuple[bool, str]:
    """Classify by known producer and data role, never extension alone."""
    name = rel.name
    parts = rel.parts
    if run.startswith('recursive_branch14_') and len(parts) == 2:
        if parts[0] in {'pilot', 'discovery', 'new_words', 'new_frames', 'new_domains'}:
            if re.fullmatch(r'B\d+-(?:G|V|A)\.npy', name) or name in {'hidden-all.npy', 'final-hidden.npy'} or re.fullmatch(r'(?:procedure|structure)-(?:all-token-GVA|replay-input)\.npy', name):
                return True, 'Recapturable native G/V/A, hidden field, token trajectory or replay input; frozen inputs and source archives retained'
    if run.startswith(('branch_reuse14_', 'b36_composition14_')) and len(parts) == 3:
        if parts[0] in {'pilot', 'discovery', 'confirm', 'format'} and re.fullmatch(r'chunk-\d+', parts[1]):
            if re.fullmatch(r'B\d+\.npz', name) or name == 'states.npz':
                return True, 'Recapturable natural-forward chunk; row identities, behavior, predictions, fits and intervention outputs retained'
    if run in {'source_binding_v1', 'source_formation_v1', 'source_relay_v1', 'source_relay_v2'}:
        if len(parts) == 3 and re.fullmatch(r'(?:discovery|confirm|lockbox_frame)-native', parts[1]) and rel.suffix == '.npy' and rel.stem in NATIVE_NAMES:
            return True, 'Recapturable model-native field; fitted and per-row prediction/causal results, diagnostic failures and parameter ledgers retained'
    if run == 'common_bench_b1_repair_v2' and len(parts) == 2:
        if parts[0] in {'discovery-native', 'lockbox-native', 'lockbox_frame-native', 'lockbox-path-account'} and name in {'hidden.npy', 'oin.npy', 'postattn.npy', 'raw.npy'}:
            return True, 'Recapturable native/path-account state; final output states, rollout predictions and numerical error results retained'
    return False, 'Retained: inputs, source, result, fitted predictor, diagnostic/minimal evidence, or client/reuse dependency'


def header_sample(path: Path) -> dict:
    """Read only the header and first bytes, without importing a model or loading the field."""
    import numpy as np

    def one(stream):
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY header: {version}')
        return {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': bool(fortran),
                'first_64_payload_bytes_hex': stream.read(64).hex()}

    if path.suffix == '.npy':
        with path.open('rb') as stream:
            return one(stream)
    with zipfile.ZipFile(path) as archive:
        result = {}
        for name in archive.namelist():
            if not name.endswith('.npy'):
                raise ValueError(f'Unexpected archive member {path}:{name}')
            with archive.open(name) as stream:
                result[name] = one(stream)
        return result


def originals():
    rows = list(csv.DictReader(CSV.open(encoding='utf-8-sig')))
    selected = [r for r in rows if r['tier'] in {'A1-COLD-NEW', 'A2-COLD-SOURCE'}]
    if {r['label'] for r in selected} != NAMES or len(selected) != len(NAMES):
        raise RuntimeError('CSV target names differ from the reviewed 26-directory scope')
    for row in selected:
        folder = validated(Path(row['absolute_path']))
        if folder != BASE / row['label']:
            raise RuntimeError('CSV directory identity mismatch')
        yield row, folder


def do_plan():
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Immutable cleanup plan already exists')
    start = time.monotonic()
    log({'stage': 'inventory_start', 'done': 0, 'total_directories': len(NAMES),
         'free_bytes': shutil.disk_usage(BASE).free, 'approval': APPROVAL})
    files = []
    for row, folder in originals():
        for path in sorted(folder.rglob('*')):
            validated(path)
            if path.is_file():
                files.append((row, folder, path))
    total_bytes = sum(p.stat().st_size for _, _, p in files)
    records = []
    done_bytes = 0
    last = time.monotonic()
    for i, (row, folder, path) in enumerate(files, 1):
        st = path.stat()
        delete, reason = role(folder.name, path.relative_to(folder))
        record = {'path': str(path), 'run': folder.name, 'tier': row['tier'],
                  'relative': path.relative_to(folder).as_posix(), 'bytes': st.st_size,
                  'mtime_ns': st.st_mtime_ns, 'device': st.st_dev, 'inode': st.st_ino,
                  'nlink': st.st_nlink, 'allocated_bytes': allocated(path),
                  'sha256': sha(path), 'action': 'delete_raw' if delete else 'keep', 'reason': reason}
        if delete:
            record['header_and_sample'] = header_sample(path)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError(f'Invalid source archive: {path}')
                record['archive_members_crc_verified'] = len(archive.namelist())
        records.append(record)
        done_bytes += st.st_size
        if time.monotonic() - last >= 25 or i == len(files):
            elapsed = time.monotonic() - start
            log({'stage': 'hash_inventory', 'done': i, 'total': len(files),
                 'bytes_done': done_bytes, 'bytes_total': total_bytes,
                 'elapsed_seconds': round(elapsed, 1),
                 'eta_seconds': round(elapsed * (total_bytes - done_bytes) / max(done_bytes, 1), 1)})
            last = time.monotonic()
    # Refuse deletion if a selected file has an alias outside the selected set.
    counts = collections.Counter((r['device'], r['inode']) for r in records if r['action'] == 'delete_raw')
    for record in records:
        if record['action'] == 'delete_raw' and record['nlink'] > counts[(record['device'], record['inode'])]:
            raise RuntimeError(f'External hardlink on deletion candidate: {record["path"]}')
    raw = [r for r in records if r['action'] == 'delete_raw']
    kept = [r for r in records if r['action'] == 'keep']
    unique = {(r['device'], r['inode']): r for r in raw}
    memo_bytes = MEMO.read_bytes()
    text = memo_bytes.decode('utf-8')
    heads = list(re.finditer(r'^## Phase [^\n]+', text, re.M))
    monthly = []
    for i, match in enumerate(heads):
        if re.search(r'\[2026-11-\d\d \d\d:\d\d\]', match[0]):
            end = heads[i+1].start() if i+1 < len(heads) else len(text)
            body = text[match.start():end]
            monthly.append({'header': match[0], 'line': text.count('\n', 0, match.start()) + 1,
                            'characters': len(body), 'sha256': hashlib.sha256(body.encode('utf-8')).hexdigest()})
    plan = {'schema': 'november_raw_retirement.v1', 'created_at': stamp(), 'root': str(ROOT),
            'approval': APPROVAL, 'source_csv': str(CSV), 'source_csv_sha256': sha(CSV),
            'script_sha256': sha(Path(__file__)), 'monthly_index': monthly,
            'memo_before_bytes': len(memo_bytes), 'memo_before_sha256': hashlib.sha256(memo_bytes).hexdigest(),
            'delete_files': len(raw), 'delete_logical_bytes': sum(r['bytes'] for r in raw),
            'delete_unique_file_bytes': sum(r['bytes'] for r in unique.values()),
            'keep_files': len(kept), 'keep_bytes': sum(r['bytes'] for r in kept),
            'records': records}
    save('plan.json', plan)
    by_run = []
    for name in sorted(NAMES):
        rr = [r for r in records if r['run'] == name]
        by_run.append({'run': name, 'delete_files': sum(r['action'] == 'delete_raw' for r in rr),
                       'delete_logical_bytes': sum(r['bytes'] for r in rr if r['action'] == 'delete_raw'),
                       'keep_files': sum(r['action'] == 'keep' for r in rr),
                       'keep_bytes': sum(r['bytes'] for r in rr if r['action'] == 'keep')})
    save('directory-summary.json', by_run)
    log({'stage': 'plan_ready', 'delete_files': len(raw),
         'logical_GiB': plan['delete_logical_bytes'] / GIB,
         'unique_GiB': plan['delete_unique_file_bytes'] / GIB,
         'keep_files': len(kept), 'monthly_entries': len(monthly), 'elapsed_seconds': time.monotonic() - start})


def do_execute():
    plan_path = OUT / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['root'] != str(ROOT) or plan['script_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Cleanup plan/root/script identity mismatch')
    if (OUT / 'deletion-receipt.json').exists():
        raise RuntimeError('Deletion receipt already exists')
    # All preflight checks precede the first unlink. No changes outside this allowlist.
    records = plan['records']
    for record in records:
        path = validated(Path(record['path']))
        st = path.stat()
        if (st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino, st.st_nlink) != (
                record['bytes'], record['mtime_ns'], record['device'], record['inode'], record['nlink']):
            raise RuntimeError(f'File changed since hash inventory: {path}')
        expected, _ = role(record['run'], Path(record['relative']))
        if expected != (record['action'] == 'delete_raw'):
            raise RuntimeError(f'Candidate role mismatch: {path}')
    raw = [r for r in records if r['action'] == 'delete_raw']
    before = shutil.disk_usage(BASE).free
    log({'stage': 'deletion_start', 'done': 0, 'total': len(raw), 'free_bytes': before,
         'plan_sha256': sha(plan_path)})
    start = time.monotonic()
    deleted = []
    current = None
    for record in raw:
        path = validated(Path(record['path']))
        st = path.stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (record['bytes'], record['mtime_ns'], record['inode']):
            raise RuntimeError(f'Concurrent change, stopped: {path}')
        # Individual file unlink only; no recursive directory deletion or cross-shell commands.
        path.unlink()
        deleted.append(record['path'])
        if record['run'] != current or len(deleted) % 100 == 0:
            log({'stage': 'delete_raw', 'done': len(deleted), 'total': len(raw),
                 'run': record['run'], 'elapsed_seconds': round(time.monotonic() - start, 2),
                 'eta_seconds': round((time.monotonic() - start) * (len(raw) - len(deleted)) / len(deleted), 2),
                 'free_bytes': shutil.disk_usage(BASE).free})
            current = record['run']
    after = shutil.disk_usage(BASE).free
    save('deletion-receipt.json', {'schema': 'november_raw_retirement_receipt.v1', 'completed_at': stamp(),
                                 'plan_sha256': sha(plan_path), 'deleted_paths': deleted,
                                 'deleted_files': len(deleted), 'logical_bytes': plan['delete_logical_bytes'],
                                 'unique_file_bytes': plan['delete_unique_file_bytes'],
                                 'free_before_bytes': before, 'free_after_bytes': after,
                                 'observed_free_increase_bytes': after - before,
                                 'scope': 'Only listed recapturable raw fields; original manifests are historical, not present-file inventories'})
    log({'stage': 'deletion_finished', 'deleted_files': len(deleted), 'free_gain_GiB': (after-before)/GIB,
         'free_after_GiB': after/GIB})


def do_verify():
    start = time.monotonic()
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUT/'deletion-receipt.json').read_text(encoding='utf-8'))
    if sha(OUT/'plan.json') != receipt['plan_sha256']:
        raise RuntimeError('Receipt/plan digest mismatch')
    failures = []
    kept = [r for r in plan['records'] if r['action'] == 'keep']
    deleted = [r for r in plan['records'] if r['action'] == 'delete_raw']
    for r in deleted:
        if Path(r['path']).exists():
            failures.append({'path': r['path'], 'reason': 'deletion candidate still exists'})
    last = start
    for i, r in enumerate(kept, 1):
        p = validated(Path(r['path']))
        if sha(p) != r['sha256']:
            failures.append({'path': r['path'], 'reason': 'retained digest mismatch'})
        if time.monotonic() - last >= 25 or i == len(kept):
            log({'stage': 'verify_retained', 'done': i, 'total': len(kept),
                 'elapsed_seconds': round(time.monotonic()-start, 1),
                 'eta_seconds': round((time.monotonic()-start) * (len(kept)-i)/i, 1)})
            last = time.monotonic()
    # Additive marker, preserving every historical manifest and result byte.
    for name in sorted(NAMES):
        retired = [r for r in deleted if r['run'] == name]
        if not retired:
            continue
        marker = BASE/name/'RAW_STORAGE_STATUS_20261118.json'
        check_write(marker, 16384, operation='原场清理状态登记')
        body = {'schema': 'raw_storage_status.v1', 'created_at': stamp(),
                'status': 'raw_fields_retired_by_user_choice', 'deleted_files': len(retired),
                'deleted_logical_bytes': sum(r['bytes'] for r in retired),
                'scope': 'Only raw files listed in the external cleanup plan. Retained fields/results are unchanged.',
                'audit_plan': str(OUT/'plan.json'), 'plan_sha256': sha(OUT/'plan.json'),
                'receipt': str(OUT/'deletion-receipt.json'),
                'recovery': 'No independent raw backup created. Recapture with retained frozen source, materials, model/tokenizer snapshot and original runtime; use a fresh output directory, never silently overwrite old results.',
                'historical_manifests': 'Original manifests/checks describe the pre-cleanup experiment; use the cleanup plan and this status for present availability.',
                'conclusion_status': 'Scientific claims, positive/negative results and admission failures unchanged. No new experiment or evidence promotion.',
                'approval': APPROVAL}
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(body, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    result = {'schema': 'november_raw_retirement_verification.v1', 'completed_at': stamp(),
              'passed': not failures, 'retained_files_rehashed': len(kept),
              'retained_bytes_rehashed': sum(r['bytes'] for r in kept),
              'deleted_paths_absent': len(deleted), 'failures': failures,
              'model_runs': 0, 'directories_recursively_deleted': 0,
              'free_bytes': shutil.disk_usage(BASE).free,
              'limitations': ['No model recapture performed; exact reproducibility after runtime changes is unverified',
                             'Per-user choice, full-field reanalysis now requires recapture for retired arrays',
                             'External client/native hardlinks and source archives preserved',
                             'Whole-drive free-space changes can include unrelated concurrent activity']}
    save('verification.json', result)
    log({'stage': 'verification_finished', **{k: result[k] for k in ['passed', 'retained_files_rehashed', 'deleted_paths_absent', 'free_bytes']}})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'execute', 'verify'])
    parser.add_argument('--maintenance-exception-10mib', action='store_true',
                        help='Use only for the explicit user-authorized maintenance exception in this chat')
    ARGS = parser.parse_args()
    {'plan': do_plan, 'execute': do_execute, 'verify': do_verify}[ARGS.command]()
