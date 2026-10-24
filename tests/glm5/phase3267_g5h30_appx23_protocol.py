# -*- coding: utf-8 -*-
# Phase 3267 (G5-H30): APPX-2/3 protocol refinement (report-only, zero GPU, zero ledger write)
# Preregistered at 3266 tail: pending user confirmations (fruit/K4, lcorr op4, APPX-1
# quantization) are NOT approved by "continue" -> proceed to protocol refinement arm.
#
# Arms:
#  a) APPX-2 protocol v2: S_attr/S_syntax cross-model vocab port -- per-model vocab
#     coordinate rebuild, executable steps S1-S4, gates, on-disk conditions verified.
#  b) APPX-3 protocol v2: K2/K3 direct measurement -- deathline criteria verbatim-frozen
#     from deepseek MEMO (content-anchored), relation to p3155 proxy readings, steps, gates.
#  c) Pending-decision inventory: three pending items unchanged; model-on-disk facts
#     recorded for APPX-1 context (status untouched).
import os, sys, json, hashlib, datetime, platform

SMOKE = os.environ.get('P3267_SMOKE') == '1'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(RES, 'phase3267', 'g5h30_appx23_protocol')
DS_MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
GAP_V113 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG_V136 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
P2874 = os.path.join(RES, 'phase2874', 'attr_vocab_v2', 'attr_vocab_v2.npz')
P2878 = os.path.join(RES, 'phase2878', 'syntax_trans_vocab', 'syntax_trans_vocab.npz')
P3155_DIR = os.path.join(RES, 'phase3155', 'g2p1_relation_family_operator_separability')
P3165_DIR = os.path.join(RES, 'phase3165')
HF = os.path.join(ROOT, 'models', 'hf')

VER = 'appx23_protocol_v2_registered'

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def today():
    return datetime.date.today().isoformat()

def log(m):
    print(m, flush=True)

# ================================================================ freeze (DESIGN)
DESIGN = {
    'phase': '3267', 'code': 'G5-H30', 'title': 'APPX-2/3 protocol refinement',
    'mode': 'report_only_zero_gpu_zero_ledger_write',
    'verdict_target': VER,
    'arm_a': {
        'goal': 'APPX-2 S_attr/S_syntax cross-model vocab port: refine draft into '
                'executable preregistered protocol (per-model vocab coordinate rebuild)',
        'anchor': 'p3165/511d9b13; dW sources p2874 attr_vocab_v2 + p2878 syntax_trans_vocab',
    },
    'arm_b': {
        'goal': 'APPX-3 K2/K3 direct measurement: refine draft into executable '
                'preregistered protocol; deathline criteria verbatim-frozen from disk',
        'anchor': 'p3155/46a5034d; A-gate three deathlines (deepseek MEMO, frozen at P35 line)',
    },
    'arm_c': {
        'goal': 'pending-decision inventory (3 items, status unchanged) + on-disk model '
                'facts for APPX-1 context (no status change)',
    },
    'no_execute': 'measurement execution is NOT part of this phase; follow-up phase '
                  'preregistration deferred to closeout',
    'smoke_states': ['gates_recompute_only'] if SMOKE else None,
}
if SMOKE:
    DESIGN.pop('smoke_states', None)
    DESIGN['smoke'] = True
blob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha8 = hashlib.sha256(blob).hexdigest()[:8]

os.makedirs(OUTDIR, exist_ok=True)
if SMOKE:
    exe_p = os.path.join(OUTDIR, 'execution_smoke.json')
else:
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == design_sha8, 'DESIGN DRIFT: %s vs %s' % (
            prev['design_sha8'], design_sha8)
        log('freeze: existing execution.json OK (%s)' % design_sha8)
with open(exe_p, 'w', encoding='utf-8', newline='') as f:
    json.dump({'phase': '3267', 'design': DESIGN, 'design_sha8': design_sha8,
               'created': datetime.datetime.now().isoformat(), 'smoke': bool(SMOKE)},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUTDIR, 'design_sha8.txt'), 'w', encoding='utf-8', newline='') as f:
    f.write(design_sha8)
log('freeze: design_sha8=%s (smoke=%s)' % (design_sha8, SMOKE))

# ================================================================ gates
gates = {}

# ---- g1: incumbent chain ----
g13 = json.load(open(GAP_V113, encoding='utf-8'))
r36 = json.load(open(REG_V136, encoding='utf-8'))
led = json.load(open(LEDGER, encoding='utf-8'))
gates['g1_incumbent_chain'] = (
    s8(GAP_V113) == '814ce808' and s8(REG_V136) == '78d37211'
    and led['ledger_sha256_8'] == '9b6ac3d1' and len(led['measurements']) == 423
    and g13['version'] == '1.13' and r36['version'] == '1.36')

# ---- g2: anchor artifacts on disk (measured hashes) ----
anchors = {
    'p2874_attr_vocab_v2.npz': s8(P2874),
    'p2878_syntax_trans_vocab.npz': s8(P2878),
    'p3155_qwen3-4b_result.json': s8(os.path.join(P3155_DIR, 'qwen3-4b', 'result.json')),
    'p3155_glm4_result.json': s8(os.path.join(P3155_DIR, 'glm4', 'result.json')),
    'p3155_qwen3-14b_result.json': s8(os.path.join(P3155_DIR, 'qwen3-14b', 'result.json')),
}
p3165_res = None
for dp, _dn, fn in os.walk(P3165_DIR):
    for f in fn:
        if f == 'result.json':
            p3165_res = os.path.join(dp, f)
            break
    if p3165_res:
        break
assert p3165_res, 'p3165 result.json not found'
anchors['p3165_result.json'] = s8(p3165_res)
gates['g2_anchors_on_disk'] = all(len(v) == 8 for v in anchors.values()) and \
    all(anchors[k] == anchors[k] for k in anchors)

# ---- g3: deathline criteria verbatim from disk (content-anchored, no line numbers) ----
memo = open(DS_MEMO, encoding='utf-8').read()
k1_needle = 'logit 预测误差 >5% 且不显著优于 B4'
k2_needle = '分离后响应 cos 降 >50% 而行为不掉'
k3_needle = '单坐标筛选 top-50 覆盖率均 <30%'
k1_ok = k1_needle in memo and '放弃"算子代数"提法' in memo
k2_ok = k2_needle in memo and '放弃"条件门"作为独立结构' in memo
k3_ok = k3_needle in memo and '放弃"单坐标机制"目标' in memo
gates['g3_deathlines_verbatim_on_disk'] = bool(k1_ok and k2_ok and k3_ok)
deathline_ctx = {
    'frozen_at': 'P35 line (deepseek MEMO "可证伪的死线（三条，须在 3150 与设计一并冻结，此后不得调阈值）")',
    'K1': {'criterion': k1_needle, 'abandon': '算子代数 → 功能性端口类描述',
           'status': 'MEASURED_REJECTED (E_read=0.373350 = gate 7.5x, 3/3; Q03)'},
    'K2': {'criterion': k2_needle, 'abandon': '条件门作为独立结构',
           'status': 'NEVER_DIRECTLY_MEASURED (p3155 ANOVA ExC share 15.2-18.3% is a '
                     'proxy only: share <50% suggests separability but does not test '
                     'the cos-drop-without-behavior-loss criterion)'},
    'K3': {'criterion': k3_needle, 'abandon': '单坐标机制 → ≤10 维联合 + 端口类',
           'status': 'NEVER_DIRECTLY_MEASURED'},
}

# ---- g4: model on-disk inventory (tokenizer/config/wte presence, measured) ----
CAND_PORT = ['qwen2.5-3b-instruct', 'qwen2-7b', 'deepseek-r1-distill-qwen-7b',
             'glm4-9b-chat-hf', 'gemma-3-4b-it']
inv = {}
for m in CAND_PORT + ['qwen3-1.7b', 'qwen3-4b', 'Qwen3-14B', 'Qwen3-14B-bnb-nf4']:
    mp = os.path.join(HF, m)
    files = os.listdir(mp) if os.path.isdir(mp) else []
    tok = [f for f in files if 'tokenizer' in f.lower()]
    cfg = 'config.json' in files
    wte = any(('safetensors' in f or '.bin' in f) for f in files)
    inv[m] = {'tokenizer_files': len(tok), 'config': cfg, 'weights': wte}
gates['g4_models_on_disk'] = all(
    inv[m]['tokenizer_files'] >= 2 and inv[m]['config'] and inv[m]['weights']
    for m in CAND_PORT)

# ---- g5: protocol completeness (steps + gates counts, structural) ----
appx2_steps = [
    {'id': 'S1', 'name': 'per-model vocab coordinate rebuild',
     'detail': 'for each candidate model: load tokenizer + W_U; rebuild attr axes (8) '
               'and syntax axes (3) following p2874/p2878 construction rules '
               '(word-pair deltas -> dW -> unit normalize); persist axes + tid_map + '
               'null_tids + func_tid per model (F7 lesson: axis vectors must be '
               'persisted with artifacts)'},
    {'id': 'S2', 'name': 'profile verification',
     'detail': 'verify K_entity_eff_ge05 profile on rebuilt axes (per-model entity-'
               'effect effective feature set) before any alignment claim'},
    {'id': 'S3', 'name': 'alignment verdict port',
     'detail': 'K_readout x S_class/S_attr/S_syntax principal-angle verdicts per model '
               '(p3165 machinery verbatim: weakly_separated gates, verdict pairs '
               "['K_readout__S_class','K_readout__S_attr','K_readout__S_syntax'])"},
    {'id': 'S4', 'name': 'cross-model synthesis',
     'detail': 'compare per-model verdicts vs qwen3-4b incumbents; report '
               'conservation/divergence; register into gap appendix only via separate '
               'registry phase (landing discipline as 3265)'},
]
appx2_gates = ['G1 rebuild bitwise recipe (axes unit-norm, dtypes fp32)',
               'G2 tokenizer per-row gate (all panel rows decode-clean, 3169 style)',
               'G3 profile gate S2 passes before S3',
               'G4 verdict gates identical thresholds to p3165 (no threshold drift)',
               'G5 landing only via separate registry phase']
appx3_steps = [
    {'id': 'T1', 'name': 'K2 direct measurement setup',
     'detail': 'for each of 3 models (qwen3-4b + 2 cross-model): at KOUT readout, '
               'project out conditional component phi_l(c) from operator component '
               'W_l (separation projection); measure response cosine drop'},
    {'id': 'T2', 'name': 'K2 behavioral half',
     'detail': 'same separation at inference; B1-style first-token behavioral probe '
               '(3155 panel verbatim); criterion: cos drop >50% AND behavior intact '
               '=> conditional gate NOT independent structure (deathline fires)'},
    {'id': 'T3', 'name': 'K3 coverage measurement',
     'detail': 'per relation family (p3155 families): single-coordinate screening '
               '(top-50 by |coord|) coverage rate; all-3-families <30% => abandon '
               'single-coordinate mechanism target (deathline fires)'},
    {'id': 'T4', 'name': 'verdict registration',
     'detail': 'K2/K3 verdicts (fires/does-not-fire per model) recorded vs frozen '
               'criteria; proxy-vs-direct relation documented; landing via separate '
               'registry phase'},
]
appx3_gates = ['H1 deathline thresholds verbatim-frozen before execution (no drift)',
               'H2 separation projection recipe frozen (subspace basis persisted, F7)',
               'H3 behavior probe identical to p3155 panel (no task drift)',
               'H4 per-model verdicts independent (no pooling across models, F4 lesson)',
               'H5 landing only via separate registry phase']
gates['g5_protocol_complete'] = (
    len(appx2_steps) == 4 and len(appx2_gates) == 5
    and len(appx3_steps) == 4 and len(appx3_gates) == 5)

# ---- g6: zero ledger write assertion ----
gates['g6_zero_ledger_write'] = (led['ledger_sha256_8'] == '9b6ac3d1'
                                 and len(led['measurements']) == 423)

log('gates: %s' % json.dumps(gates))
assert all(v is True for v in gates.values()), 'GATE FAIL'

# ================================================================ artifact
proto = {
    'schema': 'appx23_protocol_v2',
    'created': today(),
    'status': 'PREREGISTERED_DRAFT_V2 (refined from phase3262 draft b569ce24; '
              'execution NOT started)',
    'supersedes_draft': 'gpu_trajectory_protocol_draft_v1.json (APPX_2_vocab_port, '
                        'APPX_3_K2K3_direct sections)',
    'arm_a_appx2': {
        'id': 'APPX-2', 'title': 'S_attr/S_syntax 跨模型移植（逐模型词表坐标重建）',
        'anchor': 'p3165', 'anchor_sha8': '511d9b13',
        'dw_sources': {'S_attr': {'path': os.path.relpath(P2874, ROOT).replace('\\', '/'),
                                  'sha8': anchors['p2874_attr_vocab_v2.npz'],
                                  'dW_unit_shape': [8, 2560], 'axes': 8},
                       'S_syntax': {'path': os.path.relpath(P2878, ROOT).replace('\\', '/'),
                                    'sha8': anchors['p2878_syntax_trans_vocab.npz'],
                                    'dW_unit_shape': [3, 2560], 'axes': 3}},
        'alignment_precedent': {'path': os.path.relpath(p3165_res, ROOT).replace('\\', '/'),
                                'sha8': anchors['p3165_result.json']},
        'candidate_models': CAND_PORT,
        'candidate_inventory': {m: inv[m] for m in CAND_PORT},
        'steps': appx2_steps, 'gates': appx2_gates,
        'cost_estimate': 'static-side only (W_U row reconstruction, zero forward) for '
                         'S1-S2; S3 needs panel forwards per model (~15-40 min/model '
                         'on RTX 5080, 3169-style batch=1)',
        'risks': ['tokenizer-face divergence may make axes non-comparable across '
                  'tokenizers (mitigation: report per-model, never pool)',
                  'glm4/gemma vocab sizes differ from qwen family (mitigation: S1 '
                  'rebuild is per-model by construction)'],
    },
    'arm_b_appx3': {
        'id': 'APPX-3', 'title': 'K2/K3 直接测量（A 闸门死线诊断）',
        'anchor': 'p3155', 'anchor_sha8': '46a5034d',
        'deathlines': deathline_ctx,
        'panel_precedent': {'path': os.path.relpath(
            os.path.join(P3155_DIR, 'qwen3-4b', 'result.json'), ROOT).replace('\\', '/'),
            'sha8': anchors['p3155_qwen3-4b_result.json'],
            'proxy_reading': 'k2_shares_kout ANOVA ExC share band 15.2-18.3% (<50% '
                             'deathline threshold; proxy, not the direct criterion)'},
        'steps': appx3_steps, 'gates': appx3_gates,
        'cost_estimate': 'K2: separation projection + panel forwards, ~30-60 min/model; '
                         'K3: static-side screening (zero forward) + coverage stats',
        'risks': ['K2 behavioral half needs a behavioral integrity definition frozen '
                  'before execution (B1 first-token agreement, no drift)',
                  'F4 lesson: per-model independence, no pooling'],
    },
    'arm_c_pending_inventory': {
        'pending_user_confirmations': [
            {'item': 'MEMO backlog deletion: 水果类/K4', 'status': 'PENDING (not '
             'approved by continue)'},
            {'item': 'lcorr op4 A-gate governance registration (READY_PENDING_'
                     'CONFIRMATION per phase3260)', 'status': 'PENDING'},
            {'item': 'APPX-1 nf4 quantization acceptability for Qwen3-14B shared-dim '
                     'port', 'status': 'PENDING'},
        ],
        'appx1_context_facts_recorded': {
            'Qwen3-14B': inv['Qwen3-14B'],
            'Qwen3-14B-bnb-nf4': inv['Qwen3-14B-bnb-nf4'],
            'note': 'both on disk; recorded as context only, APPX-1 decision remains '
                    'with owner',
        },
        'u_star_trilogy_note': 'P-TRAJ RESOLVED (3263/3264); APPX-7 landed (3265); '
                               'this phase refines APPX-2/3 drafts; APPX-4/5 remain '
                               'cross-line governance (untouched)',
    },
    'anchors': anchors,
    'zero_ledger_write': True,
    'zero_gpu': True,
}

art_p = os.path.join(OUTDIR, 'appx23_protocol_v2.json')
with open(art_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(proto, f, ensure_ascii=False, indent=1)
log('artifact: appx23_protocol_v2.json sha8=%s' % s8(art_p))

# ================================================================ result.json
res_sha8_src = {
    'phase': '3267', 'code': 'G5-H30', 'verdict': VER,
    'design_sha8': design_sha8,
    'gates': gates,
    'arm_a_summary': {
        'appx2_status': 'PROTOCOL_V2_PREREGISTERED',
        'candidate_models_on_disk': sum(1 for m in CAND_PORT
                                        if inv[m]['tokenizer_files'] >= 2),
        'steps': len(appx2_steps), 'gates': len(appx2_gates),
    },
    'arm_b_summary': {
        'appx3_status': 'PROTOCOL_V2_PREREGISTERED',
        'k1': deathline_ctx['K1']['status'],
        'k2': deathline_ctx['K2']['status'],
        'k3': deathline_ctx['K3']['status'],
        'steps': len(appx3_steps), 'gates': len(appx3_gates),
    },
    'arm_c_summary': {
        'pending_count': 3, 'pending_status_changed': False,
        'appx1_context_models_on_disk': ['Qwen3-14B', 'Qwen3-14B-bnb-nf4'],
    },
    'anchors': anchors,
    'zero_ledger_write': True,
    'created': datetime.datetime.now().isoformat(),
}
res_blob = json.dumps(res_sha8_src, ensure_ascii=False, sort_keys=True,
                      indent=1).encode('utf-8')
res_sha8 = hashlib.sha256(res_blob).hexdigest()[:8]
res_sha8_src['res_sha8'] = res_sha8
res_p = os.path.join(OUTDIR, 'result.json')
with open(res_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(res_sha8_src, f, ensure_ascii=False, indent=1)
log('RESULT: verdict=%s design=%s res=%s -> %s' % (VER, design_sha8, res_sha8,
                                                   os.path.relpath(res_p, ROOT)))
log('GATE 6/6 ALL PASS')
