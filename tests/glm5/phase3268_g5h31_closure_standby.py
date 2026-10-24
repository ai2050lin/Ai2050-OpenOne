# -*- coding: utf-8 -*-
# Phase 3268 (G5-H31): closure standby snapshot (report-only, zero GPU, zero ledger write)
# Preregistered at 3267 tail as DEFAULT arm (a). Pending user confirmations
# (fruit/K4, lcorr op4, APPX-1 quantization) are NOT approved by "continue";
# APPX-2/3 execution requires explicit owner naming (protocols ready, not started).
import os, sys, json, hashlib, datetime, asyncio

SMOKE = os.environ.get('P3268_SMOKE') == '1'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(RES, 'phase3268', 'g5h31_closure_standby')
GAP_V113 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG_V136 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
CLOSURE_SUM = os.path.join(RES, 'phase3256', 'g5h19_closure_seal',
                           'closure_summary_3239_3255.md')
MF_V1 = os.path.join(RES, 'phase3256', 'g5h19_closure_seal', 'nline_handoff_manifest.json')
MF_V2 = os.path.join(RES, 'phase3266', 'g5h29_closure_integration',
                     'nline_handoff_manifest_v2.json')
sys.path.insert(0, ROOT)
from server.ai_rnd_service import atlas_gaps, atlas_registry  # noqa: E402

VER = 'closure_standby_registered'

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def log(m):
    print(m, flush=True)

# ================================================================ freeze
DESIGN = {
    'phase': '3268', 'code': 'G5-H31', 'title': 'closure standby snapshot',
    'mode': 'report_only_zero_gpu_zero_ledger_write',
    'verdict_target': VER,
    'arm_a': 'incumbent tri-account snapshot + live endpoint visibility '
             '(atlas_gaps/atlas_registry coroutines, 3255/3266 precedent)',
    'arm_b': 'closure package integrity: closure_summary + N-line manifest v2/v1 '
             'sealed bytes + APPX-1..7 status matrix (measured vs declared)',
    'arm_c': 'standby declaration: items requiring explicit owner approval; '
             'no preregistered execution phase follows without owner instruction',
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
    json.dump({'phase': '3268', 'design': DESIGN, 'design_sha8': design_sha8,
               'created': datetime.datetime.now().isoformat(), 'smoke': bool(SMOKE)},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUTDIR, 'design_sha8.txt'), 'w', encoding='utf-8', newline='') as f:
    f.write(design_sha8)
log('freeze: design_sha8=%s (smoke=%s)' % (design_sha8, SMOKE))

# ================================================================ gates
gates = {}

# ---- g1: incumbent tri-account ----
g13 = json.load(open(GAP_V113, encoding='utf-8'))
r36 = json.load(open(REG_V136, encoding='utf-8'))
led = json.load(open(LEDGER, encoding='utf-8'))
ms = led['measurements']
gates['g1_tri_account'] = (
    s8(GAP_V113) == '814ce808' and s8(REG_V136) == '78d37211'
    and led['ledger_sha256_8'] == '6f2a9170' and len(ms) == 424
    and ms[-1]['phase'] == '3267' and ms[-1]['name'] == 'appx23_protocol_v2_registered')

# ---- g2: live endpoint visibility ----
doc_g = asyncio.run(atlas_gaps())
doc_r = asyncio.run(atlas_registry())
appdx = doc_g.get('appendix') or []
gates['g2_endpoint_live'] = (
    doc_g['version'] == '1.13' and len(appdx) == 7
    and any(a.get('id') == 'APPX-7' for a in appdx)
    and doc_g['versions'][0]['version'] == '1.13'
    and '1.12' in [v['version'] for v in doc_g['versions']]
    and doc_r['version'] == '1.36' and len(doc_r.get('upgrade_log', [])) == 15
    and doc_r['versions'][0]['version'] == '1.36'
    and '1.35' in [v['version'] for v in doc_r['versions']])

# ---- g3: closure package on disk (measured hashes) ----
pkg = {'closure_summary': s8(CLOSURE_SUM), 'manifest_v1_sealed': s8(MF_V1),
       'manifest_v2': s8(MF_V2)}
gates['g3_closure_package'] = (pkg['manifest_v1_sealed'] == 'bc212d52'
                               and pkg['manifest_v2'] == '57531ec3'
                               and len(pkg['closure_summary']) == 8)

# ---- g4: APPX status matrix (measured vs declared) ----
declared = {'APPX-1': 'appendix_open', 'APPX-2': 'appendix_open',
            'APPX-3': 'appendix_open', 'APPX-4': 'appendix_open_crossline',
            'APPX-5': 'appendix_open_crossline', 'APPX-6': 'appendix_recorded',
            'APPX-7': 'appendix_recorded'}
measured = {a['id']: a['status'] for a in appdx}
gates['g4_appx_matrix'] = measured == declared and len(measured) == 7

# ---- g5: chain recompute at head (#419) ----
last = ms[-1]
Lc = dict(led)
Lc['measurements'] = ms  # chain rule: rec included, field = prev post
Lc['ledger_sha256_8'] = last['prev_post']
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
gates['g5_chain_head'] = (rec == '6f2a9170' == led['ledger_sha256_8'])

# ---- g6: zero ledger write ----
gates['g6_zero_ledger_write'] = (led['ledger_sha256_8'] == '6f2a9170'
                                 and len(ms) == 424)

log('gates: %s' % json.dumps(gates))
assert all(v is True for v in gates.values()), 'GATE FAIL'

# ================================================================ artifact
snap = {
    'schema': 'closure_standby_snapshot_v1',
    'created': datetime.date.today().isoformat(),
    'status': 'STANDBY (no preregistered execution phase; next phase requires '
              'owner instruction)',
    'incumbent': {
        'gap_ledger': {'version': '1.13', 'sha8': '814ce808', 'appendix_n': 7},
        'registry': {'version': '1.36', 'sha8': '78d37211', 'upgrade_log_n': 15,
                     'anchors_n': 28},
        'atlas_ledger': {'n': 424, 'post': '6f2a9170', 'tail_phase': '3267'},
    },
    'endpoint_live_check': {'gaps_current': doc_g['version'],
                            'registry_current': doc_r['version'],
                            'gaps_chain_head': [v['version'] for v in doc_g['versions'][:3]],
                            'registry_chain_head': [v['version'] for v in doc_r['versions'][:3]]},
    'closure_package': pkg,
    'appx_status_matrix': measured,
    'appx_notes': {
        'APPX-1': 'appendix_open; nf4 quantization acceptability PENDING OWNER '
                  '(context: Qwen3-14B and Qwen3-14B-bnb-nf4 both on disk)',
        'APPX-2': 'appendix_open; protocol v2 PREREGISTERED (phase3267 ce73984d), '
                  'execution requires owner naming',
        'APPX-3': 'appendix_open; protocol v2 PREREGISTERED (phase3267 ce73984d), '
                  'execution requires owner naming',
        'APPX-4': 'cross-line governance, untouched',
        'APPX-5': 'cross-line governance, untouched',
        'APPX-6': 'recorded at gap v1.12 landing (3254); checkpoint-trajectory '
                  'backlog family RESOLVED by 3263/3264/APPX-7',
        'APPX-7': 'recorded at gap v1.13 landing (3265)',
    },
    'standby_conditions': {
        'owner_approval_required': [
            'MEMO backlog deletion: 水果类/K4',
            'lcorr op4 A-gate governance registration (READY_PENDING_CONFIRMATION)',
            'APPX-1 nf4 quantization acceptability',
            'APPX-2/3 execution phases (protocols ready)',
            'APPX-4/5 cross-line governance actions',
        ],
        'auto_resume': 'none (no scheduled automation; "继续" executes default '
                       'report-only snapshot arms only)',
    },
    'zero_ledger_write': True, 'zero_gpu': True,
    'anchors': {'gap_v113': '814ce808', 'registry_v136': '78d37211',
                'ledger_head': '6f2a9170', 'manifest_v2': pkg['manifest_v2'],
                'manifest_v1': pkg['manifest_v1_sealed']},
}
art_p = os.path.join(OUTDIR, 'standby_snapshot_v1.json')
with open(art_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(snap, f, ensure_ascii=False, indent=1)
log('artifact: standby_snapshot_v1.json sha8=%s' % s8(art_p))

# ================================================================ result.json
res = {
    'phase': '3268', 'code': 'G5-H31', 'verdict': VER,
    'design_sha8': design_sha8,
    'gates': gates,
    'arm_a_summary': {'gaps_current': doc_g['version'],
                      'registry_current': doc_r['version'],
                      'appendix_n': len(appdx), 'upgrade_log_n': 15},
    'arm_b_summary': {'closure_package': pkg, 'appx_matrix_ok': gates['g4_appx_matrix']},
    'arm_c_summary': {'status': 'STANDBY',
                      'owner_approval_required_n': 5,
                      'pending_status_changed': False},
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
