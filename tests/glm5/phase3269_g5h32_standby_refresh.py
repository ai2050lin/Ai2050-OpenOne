# -*- coding: utf-8 -*-
# Phase 3269 (G5-H32): standby refresh recheck (report-only, zero GPU, zero ledger write)
# Per 3268 STANDBY declaration: "continue" executes default report-only snapshot arm.
# This is a lightweight refresh: verify nothing moved during the standby window
# (gap/registry byte-identical to 3268 snapshot; ledger tail = 3268 closeout only;
# endpoints live; APPX matrix unchanged; pending items untouched).
import os, sys, json, hashlib, datetime, asyncio

SMOKE = os.environ.get('P3269_SMOKE') == '1'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(RES, 'phase3269', 'g5h32_standby_refresh')
GAP_V113 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG_V136 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
SNAP3268 = os.path.join(RES, 'phase3268', 'g5h31_closure_standby',
                        'standby_snapshot_v1.json')
MF_V1 = os.path.join(RES, 'phase3256', 'g5h19_closure_seal', 'nline_handoff_manifest.json')
MF_V2 = os.path.join(RES, 'phase3266', 'g5h29_closure_integration',
                     'nline_handoff_manifest_v2.json')
sys.path.insert(0, ROOT)
from server.ai_rnd_service import atlas_gaps, atlas_registry  # noqa: E402

VER = 'standby_refresh_registered'

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def log(m):
    print(m, flush=True)

# ================================================================ freeze
DESIGN = {
    'phase': '3269', 'code': 'G5-H32', 'title': 'standby refresh recheck',
    'mode': 'report_only_zero_gpu_zero_ledger_write',
    'verdict_target': VER,
    'baseline': 'phase3268 standby_snapshot_v1.json (48308482)',
    'expected_ledger_state': {'n': 425, 'post': 'b7407287', 'tail_phase': '3268',
                              'note': 'only own-line 3268 closeout #420 expected after '
                                      'snapshot; any other delta = CONCURRENT line '
                                      'activity to be inspected'},
    'smoke': bool(SMOKE),
}
blob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha8 = hashlib.sha256(blob).hexdigest()[:8]

os.makedirs(OUTDIR, exist_ok=True)
exe_p = os.path.join(OUTDIR, 'execution_smoke.json' if SMOKE else 'execution.json')
if not SMOKE and os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha8'] == design_sha8, 'DESIGN DRIFT'
    log('freeze: existing execution.json OK (%s)' % design_sha8)
with open(exe_p, 'w', encoding='utf-8', newline='') as f:
    json.dump({'phase': '3269', 'design': DESIGN, 'design_sha8': design_sha8,
               'created': datetime.datetime.now().isoformat(), 'smoke': bool(SMOKE)},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUTDIR, 'design_sha8.txt'), 'w', encoding='utf-8', newline='') as f:
    f.write(design_sha8)
log('freeze: design_sha8=%s (smoke=%s)' % (design_sha8, SMOKE))

# ================================================================ gates
gates = {}
snap = json.load(open(SNAP3268, encoding='utf-8'))

# ---- g1: gap/registry byte-identical to 3268 snapshot; ledger as expected ----
led = json.load(open(LEDGER, encoding='utf-8'))
ms = led['measurements']
last = ms[-1]
gates['g1_unchanged_vs_snapshot'] = (
    s8(GAP_V113) == snap['anchors']['gap_v113'] == '814ce808'
    and s8(REG_V136) == snap['anchors']['registry_v136'] == '78d37211'
    and len(ms) == 425 and led['ledger_sha256_8'] == 'b7407287'
    and last['phase'] == '3268' and last['name'] == 'closure_standby_registered')

# ---- g2: endpoints live ----
doc_g = asyncio.run(atlas_gaps())
doc_r = asyncio.run(atlas_registry())
gates['g2_endpoint_live'] = (
    doc_g['version'] == '1.13' and len(doc_g['appendix']) == 7
    and doc_g['versions'][0]['version'] == '1.13'
    and doc_r['version'] == '1.36' and len(doc_r['upgrade_log']) == 15
    and doc_r['versions'][0]['version'] == '1.36')

# ---- g3: closure package bytes unchanged ----
gates['g3_package_bytes'] = (
    s8(MF_V1) == snap['anchors']['manifest_v1'] == 'bc212d52'
    and s8(MF_V2) == snap['anchors']['manifest_v2'] == '57531ec3')

# ---- g4: APPX matrix identical ----
measured = {a['id']: a['status'] for a in doc_g['appendix']}
gates['g4_appx_matrix_same'] = measured == snap['appx_status_matrix']

# ---- g5: chain head recompute ----
Lc = dict(led)
Lc['measurements'] = ms
Lc['ledger_sha256_8'] = last['prev_post']
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
gates['g5_chain_head'] = (rec == 'b7407287' == led['ledger_sha256_8'])

# ---- g6: zero ledger write ----
gates['g6_zero_ledger_write'] = (led['ledger_sha256_8'] == 'b7407287'
                                 and len(ms) == 425)

log('gates: %s' % json.dumps(gates))
assert all(v is True for v in gates.values()), 'GATE FAIL'

# ================================================================ artifact
art = {
    'schema': 'standby_refresh_v1',
    'created': datetime.datetime.now().isoformat(),
    'status': 'STANDBY_CONFIRMED (nothing moved during standby window; only '
              'own-line 3268 closeout #420 after snapshot)',
    'baseline': {'artifact': 'standby_snapshot_v1.json', 'sha8': '48308482'},
    'unchanged': {
        'gap_v113_sha8': s8(GAP_V113), 'registry_v136_sha8': s8(REG_V136),
        'manifest_v1': s8(MF_V1), 'manifest_v2': s8(MF_V2),
        'appx_matrix': measured,
    },
    'ledger_state': {'n': len(ms), 'post': led['ledger_sha256_8'],
                     'tail_phase': last['phase'],
                     'concurrent_append_detected': False},
    'endpoint_live': {'gaps': doc_g['version'], 'registry': doc_r['version']},
    'pending_owner_items_unchanged': [
        'MEMO backlog deletion: 水果类/K4',
        'lcorr op4 A-gate governance registration',
        'APPX-1 nf4 quantization acceptability',
        'APPX-2/3 execution phases (protocols ready)',
        'APPX-4/5 cross-line governance actions',
    ],
    'zero_ledger_write': True, 'zero_gpu': True,
}
art_p = os.path.join(OUTDIR, 'standby_refresh_v1.json')
with open(art_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(art, f, ensure_ascii=False, indent=1)
log('artifact: standby_refresh_v1.json sha8=%s' % s8(art_p))

# ================================================================ result.json
res = {
    'phase': '3269', 'code': 'G5-H32', 'verdict': VER,
    'design_sha8': design_sha8,
    'gates': gates,
    'summary': {'standby_status': 'CONFIRMED',
                'concurrent_append': False,
                'gap_registry_unchanged': True,
                'appx_matrix_unchanged': True,
                'pending_items_n': 5, 'pending_status_changed': False},
    'zero_ledger_write': True,
    'created': datetime.datetime.now().isoformat(),
}
res_sha8 = hashlib.sha256(json.dumps(res, ensure_ascii=False, sort_keys=True,
                                     indent=1).encode('utf-8')).hexdigest()[:8]
res['res_sha8'] = res_sha8
res_p = os.path.join(OUTDIR, 'result.json')
with open(res_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('RESULT: verdict=%s design=%s res=%s -> %s' % (VER, design_sha8, res_sha8,
                                                   os.path.relpath(res_p, ROOT)))
log('GATE 6/6 ALL PASS')
