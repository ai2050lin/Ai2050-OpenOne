"""B1 historical Qwen3-4B candidate repair/confirmation. Never edits v1.

All new thresholds are frozen before the first new CUDA run. Full-ID keyed
randomisation, full behaviour admission, observed-norm checks and explicit
descriptive-only status are shared by execution and reporting.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import common_bench_b1 as B
from deploy.storage_policy import check_write
from tools import provenance

OUT = ROOT / 'tests/glm5/result/common_bench_b1_repair_v2'
OLD = B.OUT
MODEL = B.MODEL
RUN = 'COMMON-BENCH-B1-REPAIR-20261110-v2'
SEED = 202611100392
LAYER = 32
ALPHA = 1.0  # retained from the old discovery freeze, not reselected
BASE_ARMS = ['pooled', 'cross_language', 'cross_source', 'unembed',
             'random0', 'random1', 'random2', 'wrong_donor',
             'direct_pooled', 'direct_cross_language', 'direct_unembed',
             'donor_q', 'donor_L']
PATH_ARMS = ['heads_document', 'heads_document_random', 'heads_parametric', 'heads_parametric_random']
CAP = ROOT / 'tests/glm5/result/mlp_puzzle_v1/capability-materials.json'
SOURCES = ['common_bench_b1_repair.py', 'common_bench_b1_repair_run.py']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT,
                          capture_output=True, text=True, check=True, timeout=20).stdout.strip()


def prov():
    return {'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'model_path': str(MODEL), 'model_revision': read(OLD / 'plan.json')['model_snapshot'],
            'seed': SEED, 'tool': RUN, 'created_at': datetime.now().astimezone().isoformat()}


def save(name, body, immutable=False):
    p = OUT / name
    check_write(p, 1 << 20, operation='B1修订复验产物')
    p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        raise RuntimeError(f'Immutable artifact exists: {p}')
    body = {**body, 'provenance': prov()}
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item()), encoding='utf-8')
    t.replace(p)
    return p


def log(stage, done, total, start, **extra):
    import psutil
    elapsed = time.monotonic() - start
    row = {'time': datetime.now().astimezone().isoformat(), 'stage': stage, 'done': done, 'total': total,
           'elapsed_seconds': round(elapsed, 1), 'eta_seconds': round(elapsed / done * (total-done), 1) if done else None,
           'available_ram_bytes': psutil.virtual_memory().available, **extra}
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated=t.cuda.memory_allocated(), cuda_peak=t.cuda.max_memory_allocated())
        try:
            cmd = "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object {$_.InstanceName -like 'pid_" + str(__import__('os').getpid()) + "_*'} | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            r = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, timeout=8)
            row['shared_gpu_bytes'] = int(float(r.stdout.strip())) if r.returncode == 0 else None
        except Exception:
            row['shared_gpu_bytes'] = None
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def rng_for(item_id, purpose):
    raw = hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{item_id}'.encode()).digest()
    return np.random.default_rng(np.frombuffer(raw, dtype='<u4'))


def seed_fingerprint(item_id, purpose):
    return hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{item_id}'.encode()).hexdigest()


def target_map(rows):
    """Balance all four alternative colours within each source/language/colour stratum."""
    groups = {}
    for r in rows:
        groups.setdefault((r['source'], r['language'], r['gold_color']), []).append(r)
    result = {}
    for (_, _, color), group in groups.items():
        k = B.COLORS.index(color)
        alternatives = [j for j in range(5) if j != k]
        ordered = sorted(group, key=lambda r: seed_fingerprint(r['id'], 'target-order'))
        shift = int(rng_for(ordered[0]['id'], 'target-cycle').integers(4))
        for i, r in enumerate(ordered):
            result[r['id']] = alternatives[(i + shift) % 4]
    return result


def behavior_gate(rows, records):
    assert [r['id'] for r in rows] == [x['id'] for x in records]
    by_entity = {}
    for r, x in zip(rows, records):
        if r['source'] == -1:
            name = r['entities'][0 if r['query'] == 1 else 1]
            by_entity.setdefault(name, []).append(x['base'] == x['gold'])
    bad = {k for k, v in by_entity.items() if np.mean(v) < .75}
    keep = np.array([not (set(r['entities']) & bad) for r in rows])
    correct = np.array([x['base'] == x['gold'] for x in records])
    top = np.array([x['base_top1_correct'] for x in records])
    cells, frames = {}, {}
    for s in (1, -1):
        for l in (1, -1):
            sel = keep & np.array([r['source'] == s and r['language'] == l for r in rows])
            cells[f'{s},{l}'] = {'n': int(sel.sum()), 'accuracy': float(correct[sel].mean()) if sel.any() else None}
    for fr in sorted({r['frame'] for r in rows}):
        sel = keep & np.array([r['frame'] == fr for r in rows])
        frames[str(fr)] = {'n': int(sel.sum()), 'accuracy': float(correct[sel].mean()) if sel.any() else None}
    top_rate = float(top[keep].mean()) if keep.any() else None
    passed = bool(by_entity and len(bad)/len(by_entity) <= .20 and top_rate is not None and top_rate >= .80
                  and all(v['n'] and v['accuracy'] >= .90 for v in list(cells.values()) + list(frames.values())))
    return {'passed': passed, 'entities': len(by_entity), 'dropped_entities': sorted(bad),
            'all_n': len(rows), 'all_forced': float(correct.mean()), 'all_top1': float(top.mean()),
            'screened_n': int(keep.sum()), 'screened_top1': top_rate, 'cells': cells, 'frames': frames}, keep & correct


def decision(gate, controls, calibration, rates):
    if not calibration:
        return {k: 'invalid_execution' for k in ('D1', 'D2', 'D3')}
    if not gate:
        return {k: 'behavior_ineligible' for k in ('D1', 'D2', 'D3')}
    if not controls:
        return {k: 'descriptive_only_controls_failed' for k in ('D1', 'D2', 'D3')}
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    return {'D1': tri(rates['pooled'] >= .80, rates['pooled'] < .50),
            'D2': tri(rates['cross_language'] >= .60 and rates['cross_language_wrong_language'] <= .10,
                       rates['cross_language'] < .30),
            'D3': tri(rates['cross_source'] >= .60, rates['cross_source'] < .30)}


def cp_interval(hits, n, alpha=.05):
    from scipy.stats import beta
    return [float(beta.ppf(alpha/2, hits, n-hits+1)) if hits else 0.,
            float(beta.ppf(1-alpha/2, hits+1, n-hits)) if hits < n else 1.]


def rate_ci(values, rows, seed=0):
    values = np.asarray(values, float)
    if not len(values):
        return {'n': 0, 'rate': None, 'entity_ci95': None, 'item_cp95_descriptive': None}
    names = sorted({e for r in rows for e in r['entities']})
    a = np.array([names.index(r['entities'][0]) for r in rows])
    b = np.array([names.index(r['entities'][1]) for r in rows])
    rng = rng_for(str(seed), 'ci')
    vals = []
    for _ in range(2000):
        counts = rng.multinomial(len(names), np.full(len(names), 1/len(names)))
        w = counts[a]*counts[b]
        if w.sum():
            vals.append(float(values@w/w.sum()))
    binary = bool(np.all((values == 0) | (values == 1)))
    return {'n': len(values), 'rate': float(values.mean()), 'entity_ci95': np.quantile(vals, [.025,.975]).tolist(),
            'item_cp95_descriptive': cp_interval(int(values.sum()), len(values)) if binary else None,
            'caveat': 'Entity bootstrap conditions on these templates. Item CP ignores clustering; neither is an independent-node replication.'}


def prereg():
    from scipy.stats import beta
    check_write(OUT, 8*2**30, operation='B1历史候选修订确认')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'prereg.json').exists()
    old = read(OLD / 'plan.json')
    # Explicit prospective sensitivity: absolute-rate lower bound under family
    # independence, at family and entity-sized effective N. Not a claim
    # that 720 dependent items provide 720 independent observations.
    power = {}
    for n, p0, p1 in [(90,.80,.95),(90,.60,.85),(15,.80,.95),(15,.60,.85)]:
        rng = rng_for(str((n,p0,p1)), 'power')
        ks = rng.binomial(n,p1,20000)
        lower = np.where(ks>0,beta.ppf(.05,ks,n-ks+1),0)
        power[str((n,p0,p1))] = {'units': n, 'null_rate': p0, 'expected_rate': p1,
                                    'power': float(np.mean(lower>p0)), 'assumption': 'independent Bernoulli units; sensitivity only'}
    dependencies = [ROOT/'tests/glm5'/s for s in SOURCES] + [ROOT/'tests/glm5'/s for s in
                    ['common_bench_b1.py','common_bench_b1_stage_c.py','mlp_condition_stream_load.py',
                     'mlp_condition_design.py','mlp_condition_run.py']]
    body = {'schema': 'b1_repair_prereg.v2', 'run_id': RUN, 'kind': 'real', 'created_at': datetime.now().astimezone().isoformat(),
            'historical_model_exception': 'Finish and audit the already-frozen Qwen3-4B candidate. New main-model work is Qwen3-14B.',
            'model_snapshot': old['model_snapshot'], 'benchmark_plan_sha256': sha(OLD/'plan.json'),
            'old_d_frozen_sha256': sha(OLD/'staged-frozen.json'), 'old_centres_sha256': sha(OLD/'staged-centres.npz'),
            'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in dependencies},
            'layer': LAYER, 'alpha': ALPHA, 'seed': SEED, 'pilot_families': 32,
            'materials': old['materials'], 'capability': {'path':str(CAP),'sha256':sha(CAP),'n':512,
                'scope':'exposed historical keyed MCQ pool; fixed arbitrary colour edits test bounded side effects, not semantic transfer',
                'preservation':'paired accuracy change lower 95% interval >= -0.02; inconclusive if interval crosses -0.02'},
            'arms': BASE_ARMS, 'path_arms': PATH_ARMS,
            'target': 'balanced target cycle over full-ID-hash-sorted items within (source,language,original colour)',
            'randomisation': 'full SHA256 id+purpose seed; three independent random directions per item',
            'unembed_baseline': 'mean output-weight row per colour across token variants and both languages; target-original; actual BF16 norm matched to pooled',
            'norm_match': {'arms':['unembed','random0','random1','random2','wrong_donor'], 'max_relative_error':.02,
                           'minimum_valid_fraction':.99},
            'calibration': 'same-shape no-op and restored full-vocabulary logits must be bitwise equal; otherwise stop the affected split',
            'behavior_gate': old['prereg']['behavior_gate'],
            'D_thresholds': {'pooled':.80,'cross_language':.60,'cross_source':.60,'cross_language_wrong_language_max':.10},
            'controls': {'random_success_max':.05,'random_unchanged_min':.90,'wrong_donor_target_max':.10},
            'comparison': 'pooled versus equal-effective-norm unembedding baseline, paired full-vocab KL, target margins and success; no accuracy win required to register a negative result',
            'native_path_selection': 'discovery pilot only; blocks 14..31; top 4 positive correct-colour projected head writes per source, same-block count-matched random head groups; no label-based selection on lockbox',
            'path_prediction': 'before each edit predict full logits by subtracting the measured base head-group write from base final pre-norm state; compare with actual original-model head ablation. Local source-state predictor, not text-only.',
            'path_gate': 'observed target margin effect is descriptive; mechanistic prediction candidate needs >=10% paired squared-margin-error reduction versus zero-change in both lockbox panels and matched random-head comparisons; no unique/necessary mechanism claim',
            'generation': 'first 16 full families per panel, max 6 new tokens; base/pooled/unembed/cross_language. Full token histories recorded; no class forcing.',
            'continuous_outputs': 'all-vocabulary KL; original/target logit margins; final normalized states permit recomputing full logits with original weights',
            'new_blindness': 'lockbox responses unopened; some entity behaviour exposed in older pilots; report exposed/unexposed strata. Both panels share entity roster.',
            'state_analysis': 'existing discovery/confirm used retrospectively for natural attn/MLP decomposition, A1/A2 reconstruction and genuine multi-step rollout diagnostic. No new-blind claim.',
            'resource_budget': {'disk_peak_GiB':8,'minimum_free_GiB':20,'gpu_minutes_max_per_stage':60,
                                'whole_campaign_gpu_minutes_max':120,'cuda_batch':8,'models_concurrent':1},
            'power_absolute_rate_sensitivity': power,
            'formal_evidence_promotion': 0}
    save('prereg.json',body,True)
    print(json.dumps({'preregistered':str(OUT/'prereg.json'),'power_sensitivity':power}),flush=True)


def checked():
    p = read(OUT/'prereg.json')
    for name,h in p['source_sha256'].items():
        assert sha(ROOT/name)==h, f'Frozen source changed: {name}'
    assert sha(OLD/'plan.json')==p['benchmark_plan_sha256']
    assert sha(OLD/'staged-centres.npz')==p['old_centres_sha256']
    return p


def native_weight(name):
    import torch
    index=read(MODEL/'model.safetensors.index.json')['weight_map']
    with (MODEL/index[name]).open('rb') as f:
        n=int.from_bytes(f.read(8),'little');hdr=json.loads(f.read(n));entry=hdr[name]
        assert entry['dtype']=='BF16'
        a,b=entry['data_offsets'];f.seek(8+n+a)
        raw=bytearray(f.read(b-a));assert len(raw)==b-a
    return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(entry['shape'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prereg']);args=ap.parse_args()
    globals()[args.stage]()
