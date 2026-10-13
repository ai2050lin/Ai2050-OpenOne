# -*- coding: utf-8 -*-
"""Phase 3224 (G5-G25-B): weights-acquisition readiness inventory — zero-action
report-only arm (metadata fetches only: config.json + safetensors index; NO
safetensors download).

Prereg: MEMO Phase 3223 tail, candidate (b). Candidate (a) (weights download)
requires explicit user authorization with exact byte counts, which THIS arm
produces.

R-checks:
  R1  disk headroom C:/D: vs minimal / full acquisition plans.
  R2  local cache state recheck (4B snapshot empty, 14B config-only,
      glm4 absent) — consistency with 3223 D3/D4 findings.
  R3  metadata fetch (config.json + model.safetensors.index.json) for
      Qwen/Qwen3-4B-Base + Qwen/Qwen3-14B; tie_word_embeddings parsed;
      EXACT minimal shard set for W_U (= lm_head or tied embed_tokens)
      + model.norm (ln_f) located with byte counts.
  R4  HF API file manifests x3 repos (incl glm4-9b-chat optional) with totals.
  R5  inventory doc sealed: weights_readiness_inventory_v1.json
      (download plans A minimal / B full / C +glm4, integrity plan,
      authorization request block).
  R6  ledger untouched (n=375, 5111392c).
"""
import glob
import hashlib
import json
import os
import shutil
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                      'rdc_query_construction_20260913', 'phase3224',
                      'g5g25b_weights_readiness_inventory')
SMOKE = os.environ.get('P3224_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def log(s):
    print(s)


os.makedirs(OUTDIR, exist_ok=True)
bad = []
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------- R1: disk headroom ----------
R1 = {}
for d in ('C:/', 'D:/'):
    t, u, f = shutil.disk_usage(d)
    R1[d] = {'total_gb': round(t / 2**30, 1), 'free_gb': round(f / 2**30, 1)}
    log('R1 %s free=%.1fGB' % (d, R1[d]['free_gb']))

# ---------- R2: local cache state recheck (WEIGHTS-level criterion) ----------
# Note: this arm's own R3 metadata fetch (config/index json, ~KB) may
# materialize snapshot dirs from run1 onward; the consistency criterion is
# therefore WEIGHTS-level: no .safetensors anywhere. Documented side effect.
R2 = {}
for mid, hub in [('qwen3-4b', 'models--Qwen--Qwen3-4B-Base'),
                 ('qwen3-14b', 'models--Qwen--Qwen3-14B'),
                 ('glm4-9b', 'models--THUDM--glm-4-9b-chat')]:
    snaps = glob.glob(os.path.join(HUB, hub, 'snapshots', '*'))
    if not snaps:
        R2[mid] = 'no_snapshot_dir'
    else:
        files = os.listdir(snaps[0])
        has_w = any(f.endswith('.safetensors') for f in files)
        R2[mid] = ('safetensors_present' if has_w
                   else 'no_weights_in_snapshot')
chk2 = all(v == 'no_weights_in_snapshot' or v == 'no_snapshot_dir'
           for v in R2.values())
log('R2 cache state (weights-level): %s -> %s' % (
    R2, 'OK (no safetensors anywhere; consistent with 3223 D3/D4)'
    if chk2 else 'INCONSISTENT'))
if not chk2:
    bad.append(('R2 cache state inconsistent', R2))

# ---------- R3: metadata fetch (json only) ----------
import huggingface_hub
from huggingface_hub import hf_hub_download

R3 = {}
for mid, repo in [('qwen3-4b', 'Qwen/Qwen3-4B-Base'),
                  ('qwen3-14b', 'Qwen/Qwen3-14B')]:
    cfg_p = hf_hub_download(repo, 'config.json')
    idx_p = hf_hub_download(repo, 'model.safetensors.index.json')
    cfg = json.load(open(cfg_p, encoding='utf-8'))
    wm = json.load(open(idx_p, encoding='utf-8'))['weight_map']
    tied = bool(cfg.get('tie_word_embeddings'))
    wu_key = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    assert wu_key in wm, (mid, wu_key)
    norm_key = 'model.norm.weight'
    assert norm_key in wm, (mid, norm_key)
    api = huggingface_hub.HfApi()
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings if s.size}
    minimal = sorted({wm[wu_key], wm[norm_key]} | {'config.json', 'model.safetensors.index.json'})
    mb = sum(sizes.get(f, 0) for f in minimal)
    full = sum(sizes.values())
    R3[mid] = {
        'repo': repo, 'tied': tied, 'wu_key': wu_key,
        'wu_shard': wm[wu_key], 'norm_shard': wm[norm_key],
        'minimal_files': minimal, 'minimal_gb': round(mb / 2**30, 2),
        'full_gb': round(full / 2**30, 2),
        'config_sha8': sha8_file(cfg_p), 'index_sha8': sha8_file(idx_p),
        'hidden_size': cfg.get('hidden_size'),
        'vocab_size': cfg.get('vocab_size'),
    }
    log('R3 %s tied=%s wu_shard=%s norm_shard=%s minimal=%.2fGB full=%.2fGB'
        % (mid, tied, wm[wu_key], wm[norm_key], mb / 2**30, full / 2**30))

# ---------- R4: API manifests x3 ----------
from huggingface_hub import HfApi
api = HfApi()
R4 = {}
for repo in ('Qwen/Qwen3-4B-Base', 'Qwen/Qwen3-14B', 'THUDM/glm-4-9b-chat'):
    info = api.model_info(repo, files_metadata=True)
    sib = [(s.rfilename, s.size) for s in info.siblings if s.size]
    st = [(f, sz) for f, sz in sorted(sib) if f.endswith('.safetensors')]
    R4[repo] = {'files': len(sib),
                'total_gb': round(sum(sz for _, sz in sib) / 2**30, 2),
                'shards': [(f, round(sz / 2**30, 2)) for f, sz in st]}
    log('R4 %s files=%d total=%.2fGB shards=%d'
        % (repo, len(sib), R4[repo]['total_gb'], len(st)))

# ---------- R5: inventory doc ----------
min_gb = round(R3['qwen3-4b']['minimal_gb'] + R3['qwen3-14b']['minimal_gb'], 2)
full_gb = round(R4['Qwen/Qwen3-4B-Base']['total_gb']
                + R4['Qwen/Qwen3-14B']['total_gb'], 2)
glm_gb = R4['THUDM/glm-4-9b-chat']['total_gb']
plan_a_ok = R1['C:/']['free_gb'] > min_gb * 1.5
plan_b_ok = R1['C:/']['free_gb'] > full_gb * 1.3
doc = {
    'schema': 'weights_readiness_inventory_v1',
    'phase': 3224,
    'name': 'g5g25b_weights_readiness_inventory',
    'arm': 'G5-G25-B zero-action report-only (metadata json fetches only; '
           'NO safetensors downloaded this phase)',
    'status': 'SEALED_INVENTORY',
    'created': now,
    'disk': R1,
    'cache_state': R2,
    'w_u_shard_map': R3,
    'repo_manifests': R4,
    'download_plans': {
        'A_minimal_qwen_pair': {
            'bytes_gb': min_gb, 'target': 'W_U + ln_f only (A1 arm minimum)',
            'files': {m: R3[m]['minimal_files'] for m in R3},
            'disk_verdict': 'OK' if plan_a_ok else 'INSUFFICIENT',
            'method': 'hf_hub_download per file (huggingface_hub %s); '
                      'LFS etag sha256 verified by hub at download time; '
                      'post-download file sha8 anchors frozen at A1 '
                      'freeze-before-observation' % huggingface_hub.__version__,
        },
        'B_full_qwen_pair': {
            'bytes_gb': full_gb, 'target': 'full repos (enables future arms)',
            'disk_verdict': 'OK' if plan_b_ok else 'INSUFFICIENT',
        },
        'C_plus_glm4': {
            'bytes_gb': round(full_gb + glm_gb, 2),
            'target': 'full qwen pair + glm-4-9b-chat (restores 3/3 model '
                      'scope; cross-model generality upgrade)',
            'disk_verdict': 'OK' if R1['C:/']['free_gb'] > (full_gb + glm_gb) * 1.3
            else 'INSUFFICIENT',
            'note': 'glm4 cached H (3154/3169) collected from a local glm4 '
                    'install no longer in hub cache — confirm source repo '
                    'THUDM/glm-4-9b-chat matches the earlier H before use',
        },
    },
    'authorization_request': (
        'Candidate (a) execution requires explicit user authorization. '
        'Plan A minimal = %.2f GB (qwen pair W_U+ln_f shards) — RECOMMENDED '
        'for 3225 A1 arm; Plan B full = %.2f GB; Plan C +glm4 = %.2f GB. '
        'Reply with plan letter to authorize.' % (min_gb, full_gb,
                                                  full_gb + glm_gb)),
    'integrity_plan': [
        'hub-internal LFS sha256 (etag) verification at download time',
        'A1 freeze records sha8 of every acquired safetensors shard '
        '(freeze-before-observation discipline)',
        'downloaded weights are used for CPU matrix math only '
        '(ln_f + W_U); no model forward, no GPU',
    ],
    'ledger_untouched': {'n': 375, 'post': 'a1075729_pre_3223_closed_at_5111392c'},
}
content = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
csha = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
doc['sha8'] = csha
dp = os.path.join(OUTDIR, PFX + 'weights_readiness_inventory_v1.json')
with open(dp, 'wb') as f:
    f.write(json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8'))
log('R5 inventory sealed: %s content sha8=%s (file %s)' % (
    os.path.basename(dp), csha, sha8_file(dp)))

# ---------- R6: ledger untouched ----------
led = json.load(open(LEDGER, encoding='utf-8'))
ok6 = (len(led['measurements']) == 375 and led['ledger_sha256_8'] == '5111392c')
log('R6 ledger untouched: n=%d post=%s -> %s' % (
    len(led['measurements']), led['ledger_sha256_8'], 'OK' if ok6 else 'FAIL'))
if not ok6:
    bad.append(('R6 ledger touched', led['ledger_sha256_8']))

clean = not bad
verdict = ('weights_readiness_inventory_registered' if clean
           else 'weights_readiness_inventory_dirty (%d contradictions)'
           % len(bad))
res = {
    'phase': 3224,
    'name': 'g5g25b_weights_readiness_inventory',
    'arm': 'zero-action report-only (metadata json only)',
    'verdict': verdict,
    'created': now,
    'plan_a_minimal_gb': min_gb,
    'plan_b_full_gb': full_gb,
    'plan_c_plus_glm4_gb': round(full_gb + glm_gb, 2),
    'disk_free_gb': {k: v['free_gb'] for k, v in R1.items()},
    'w_u_shard_map': {m: {'tied': R3[m]['tied'],
                          'wu_shard': R3[m]['wu_shard'],
                          'norm_shard': R3[m]['norm_shard'],
                          'minimal_gb': R3[m]['minimal_gb']} for m in R3},
    'inventory_doc': {'file': PFX + 'weights_readiness_inventory_v1.json',
                      'content_sha8': csha,
                      'file_sha8': sha8_file(dp)},
    'contradictions': bad,
}
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'wb') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1).encode('utf-8'))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w', encoding='utf-8') as f:
    f.write('')
log('== Phase 3224 g5g25b_weights_readiness_inventory %s ==' % (
    'CLEAN' if clean else 'DIRTY'))
for b in bad:
    log('  CONTRADICTION: %s' % (b,))
log('verdict: %s' % verdict)
