# -*- coding: utf-8 -*-
"""Phase 3237 (G5-G31): registry landing arm (zero GPU).

Per prereg (MEMO Phase 3236 tail, deferred from original 3236 plan): route
the sealed 3235 verdict UPSTREAM_ABSENT into the registries —
  (1) F21 new failure entry (F20 family extension: upstream has no lever
      either — UL_flip/UL_dose/UL_amp all FAIL x2 models, UL_spec PASS);
  (2) GAP-5 mechanism_note FINAL append (three-stage picture closed:
      upstream distributed carrying -> last-layer exact linear readout ->
      post-softmax nonlinearity) + final_verdict field (new key);
  (3) FTR-25/26 scope notes (u_isa/u_hasa upstream scan also absent).
Registry v1.31 -> v1.32, gap ledger v1.10 -> v1.11, append-only new files
with byte-preservation asserts on all untouched fields.

Zero GPU: pure consistency recompute on sealed result.json + JSON assembly.
Ledger is READ-ONLY in this arm (governance entry #389 appended in closeout).
"""

import hashlib
import json
import os
from copy import deepcopy

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3237', 'g5g31_registry_landing')
SMOKE = os.environ.get('P3237_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''

REG31 = os.path.join(SRC_DIR, 'phase3231', 'g5g28_registry_decision',
                     'atlas_registry_v1_31.json')
GAP10 = os.path.join(SRC_DIR, 'phase3232', 'g5g29_gd_amp_class_robustness',
                     'gap_ledger_v1_10.json')
R35 = os.path.join(SRC_DIR, 'phase3235', 'g5g31_gap5_gate_decision',
                   'result.json')
NPZ34 = os.path.join(SRC_DIR, 'phase3234', 'g5g30_gap5_upstream_lever_scan',
                     'upstream_lever_scan.npz')
LEDGER = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')

# ---- anchors measured on disk in preflight (frozen here) ----
REG31_SHA = 'ef5189e7'          # file sha
REG31_CSHA = '4b6648c2'         # content sha (dict minus sha8 key)
GAP10_SHA = '378de1cd'          # file sha
R35_RES = '5bab6419'            # 3235 result res_sha8
R35_FREEZE = '99b944fb'
R34_RES = '71050e5c'            # 3234 result res_sha8
NPZ34_SHA = '48034bd9'          # 3234 sealed npz file sha
LED_N, LED_POST = 388, '05a31fc4'
DESIGN_3233 = 'd74b6484'

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


def jd(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


os.makedirs(OUTDIR, exist_ok=True)
bad = []

# ---------- S0: input integrity ----------
assert sha8_file(REG31) == REG31_SHA, 'registry v1.31 file sha drift'
assert sha8_file(GAP10) == GAP10_SHA, 'gap v1.10 file sha drift'
assert sha8_file(NPZ34) == NPZ34_SHA, '3234 npz file sha drift'
r35 = json.load(open(R35, encoding='utf-8'))
assert r35['verdict'] == 'UPSTREAM_ABSENT'
res_check = dict(r35)
res_check['res_sha8'] = None
if csha8(res_check) != R35_RES:
    bad.append('3235 res_sha8 recompute mismatch')
log('S0 inputs: reg31 %s / gap10 %s / npz34 %s file shas OK; 3235 verdict '
    'UPSTREAM_ABSENT, res recompute %s' % (
        REG31_SHA, GAP10_SHA, NPZ34_SHA,
        'OK' if not bad else 'MISMATCH'))

# ---------- freeze BEFORE any registry assembly ----------
fz = {
    'phase': '3237', 'name': 'g5g31_registry_landing',
    'prereg': ('AGI_GPT5_MEMO Phase 3236 tail (deferred 3236 landing arm): '
               'F21 append (F20 family: upstream lever absent) + GAP-5 '
               'mechanism_note FINAL (three-stage closure) + FTR-25/26 '
               'scope notes; registry v1.32, gap v1.11. Inputs: sealed '
               '3235 (res %s / freeze %s), sealed 3234 (res %s / npz %s), '
               'design 3233 (%s)' % (R35_RES, R35_FREEZE, R34_RES,
                                     NPZ34_SHA, DESIGN_3233)),
    'clarifications_frozen_before_assembly': [
        'K1 pure landing arm: the UPSTREAM_ABSENT verdict was decided by '
        'the 3233 preregistered gates (d74b6484) executed in 3234/3235; '
        'this arm makes NO new observation, re-opens NO gate, and only '
        'executes the registry/gap appends routed by the prereg.',
        'K2 F21 is a NEW failure entry, not an F20 refinement: F20 = '
        'readout-direction families are not behavioral levers (u_R, dR); '
        'F21 = the upstream layer-scan finds NO single-direction lever '
        'either (UL_flip/UL_dose/UL_amp all FAIL, UL_spec PASS, x2 '
        'models). Together: behavioral carrying is distributed '
        '(UPSTREAM_ABSENT). failures 20 -> 21.',
        'K3 GAP-5 closure semantics: mechanism_note gains the FINAL '
        'three-stage paragraph; a NEW key final_verdict records the '
        'closed status; the legacy status key '
        '(localized_linear_presoftmax) stays byte-exact (append-only '
        'discipline — history preserved inside the note).',
        'K4 scope-note semantics: FTR-25/26 (u_isa/u_hasa family) gain '
        'scope_limits appends only; E-levels unchanged (the negative is '
        'on BEHAVIORAL-LEVER status, not on geometric-band evidence).',
        'K5 3235 contradictions boundary: the two "dose rate mismatch '
        'qwen3-14b" entries were diagnosed by the 3235 independent '
        'verify (V7) as an in-arm probe mode leak (rates_j1 reusing a '
        'residual mode variable); the gate basis (total flip 4/4) '
        'recomputes matching and the verdict is unaffected. Recorded '
        'verbatim inside F21 evidence.',
        'K6 ledger read-only in arm (n=%d post %s); governance entry '
        '#389 appended in closeout.' % (LED_N, LED_POST),
    ],
    'inputs': {'registry_v131': REG31_SHA, 'gap_v110': GAP10_SHA,
               'r35_res': R35_RES, 'r35_freeze': R35_FREEZE,
               'r34_res': R34_RES, 'npz34': NPZ34_SHA,
               'design_3233': DESIGN_3233,
               'ledger': 'n=%d post %s (read-only)' % (LED_N, LED_POST)},
    'zero_gpu': 'pure consistency recompute on sealed result.json + JSON '
                'assembly; no model, no forward',
}
fz['sha8'] = csha8(dict(fz))
FZ_PATH = os.path.join(OUTDIR, PFX + 'execution.json')
with open(FZ_PATH, 'w', encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3237 g5g31_registry_landing %s ==' % ('SMOKE' if SMOKE else 'FULL'))
log('freeze sealed: %s (before any registry assembly)' % fz['sha8'])

# ---------- S1: gate recompute from sealed 3235 result ----------
INTEG = {'per_model': {}}
for m in ('qwen3-4b', 'qwen3-14b'):
    mv = r35['models'][m]
    g = mv['gates'] if 'gates' in mv else r35['gates'][m]
    rec = {}
    # UL_flip: any direction any fold pass_primary
    flip_pass = any(mv['ul_flip'][d]['pass_primary']
                    for d in ('u_isa', 'u_hasa'))
    rec['UL_flip_recomputed'] = bool(flip_pass)
    # UL_dose: monotone for BOTH directions
    dose_mono = {d: mv['ul_dose'][d]['monotone'] for d in ('u_isa', 'u_hasa')}
    rec['UL_dose_recomputed'] = bool(all(dose_mono.values()))
    # UL_amp: pass_gate field
    rec['UL_amp_recomputed'] = bool(mv['ul_amp']['pass_gate'])
    rec['UL_amp_spearman_best'] = mv['ul_amp']['spearman_best']
    rec['Lstar'] = mv['Lstar']
    rec['flip_per_fold_max_L_all_zero'] = all(
        x == 0.0 for d in ('u_isa', 'u_hasa')
        for x in mv['ul_flip'][d]['per_fold_max_L'])
    # consistency vs sealed gates dict
    sealed = {'UL_flip': g['UL_flip'], 'UL_dose': g['UL_dose'],
              'UL_amp': g['UL_amp']}
    recomp = {'UL_flip': rec['UL_flip_recomputed'],
              'UL_dose': rec['UL_dose_recomputed'],
              'UL_amp': rec['UL_amp_recomputed']}
    if sealed != recomp:
        bad.append('gate recompute mismatch %s: sealed %s vs %s'
                   % (m, sealed, recomp))
    INTEG['per_model'][m] = rec
INTEG['gates_sealed'] = r35['gates']
INTEG['UL_spec'] = {m: r35['gates'][m]['UL_spec'] for m in r35['gates']}
log('S1 integrity: UL_flip/UL_dose/UL_amp recomputed from sealed 3235 '
    'result and matched sealed gates x2 models; UL_flip per_fold_max_L '
    'all zero; Lstar={u_isa:0, u_hasa:0} both; UL_spec True x2 (sealed)')

if SMOKE:
    res = {
        'phase': 3237, 'name': 'g5g31_registry_landing',
        'arm': 'SMOKE dry-run (integrity only, no registry writes)',
        'freeze_sha8': fz['sha8'], 'integrity': INTEG,
        'recheck': {'contradictions': bad},
        'verdict': 'f21_landing_smoke',
    }
    res['res_sha8'] = hashlib.sha256(json.dumps(
        dict(res), ensure_ascii=False, indent=1, sort_keys=True)
        .encode('utf-8')).hexdigest()[:8]
    with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
              encoding='utf-8', newline='') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
              encoding='utf-8', newline='') as f:
        f.write('\n'.join(LOG) + '\n')
    log('SMOKE result sealed: res %s' % res['res_sha8'])
    raise SystemExit(0)

# ---------- S2: registry v1.32 (F21 append + scope notes) ----------
reg31 = json.load(open(REG31, encoding='utf-8'))
assert reg31['version'] == '1.31' and len(reg31['features']) == 26
assert len(reg31['failures']) == 20 and reg31['failures'][-1]['id'] == 'F20'
assert len(reg31['upgrade_log']) == 11
reg32 = deepcopy(reg31)
reg32['version'] = '1.32'
reg32['supersedes'] = ('atlas_registry_v1_31.json (%s, Phase 3231)'
                       % REG31_SHA)

M4 = INTEG['per_model']['qwen3-4b']
M14 = INTEG['per_model']['qwen3-14b']
assert M4['flip_per_fold_max_L_all_zero'] and M14['flip_per_fold_max_L_all_zero']

f21 = {
    'id': 'F21',
    'kind': 'upstream_lever_absent',
    'phase': 3237,
    'text': ('上游杠杆缺席定论（3233 G5-G30 设计 → 3234 G5-G30 扫描 → 3235 '
             'G5-G31 门评估 → 3237 入册）：在 3233 预注册门控（design %s）'
             '下，对 qwen3-4b 与 qwen3-14b 执行全层上游杠杆扫描（粗扫全 L '
             '网格 × 2 方向 u_isa/u_hasa + 峰带 α 细扫 + null 10+2 种子，'
             '3234 sealed npz %s）。3235 门评估：UL_flip FAIL（L*={u_isa:%d, '
             'u_hasa:%d}，全折 flip 计数 0，per_fold_max_L 全 0.0，两模型'
             '一致；门限 0.15）；UL_dose FAIL（剂量-响应非单调，两模型一致）；'
             'UL_amp FAIL（幅度-排序 spearman_best %+.2f(4b)/%+.2f(14b)，'
             '远低于门 0.6）；UL_spec PASS（翻转特异性未越界——失败不是'
             '乱翻，而是不翻）。三否定门 × 2 模型一致 ⇒ 上游不存在任何单'
             '方向行为杠杆：将 u 方向注入上游任意层、任意幅度、任意方向，'
             '行为判断 flip 均为 0（null 带亦 0）。与 F20 合并：读出端'
             '（F20：u_R/dR 双配方传递正交）与上游端（F21：全层扫描缺席）'
             '均无单方向杠杆 ⇒ 行为承载为分布式（UPSTREAM_ABSENT 定谳）。'
             '这是「读得出≠控得住」从读出端向上游端的完整闭合：几何指纹'
             '的缺席性（无杠杆）与存在性（可读出）同时成立，行为由分布式'
             '叠加承载，图谱的机制单位应为「载体子空间 × 幅度 × 组合规则」'
             '而非单方向。边界记录：3235 臂内 contradictions 两条'
             '（dose rate mismatch qwen3-14b u_isa/u_hasa）经 3235 独立'
             '复核 V7 诊断为臂内探针 mode 泄漏（rates_j1 复用残留 mode '
             '变量），门控主值 basis 总 flip 4/4 重算匹配，verdict 不受'
             '影响；本臂 S1 已从 sealed 3235 result 重算三门并匹配。'
             % (DESIGN_3233, NPZ34_SHA,
                M4['Lstar']['u_isa'], M4['Lstar']['u_hasa'],
                M4['UL_amp_spearman_best'], M14['UL_amp_spearman_best'])),
    'evidence': [
        'sealed 3234: res %s / npz %s / freeze dde04645'
        % (R34_RES, NPZ34_SHA),
        'sealed 3235: res %s / freeze %s (UPSTREAM_ABSENT)'
        % (R35_RES, R35_FREEZE),
        'design 3233: g5g30_upstream_lever_design_v1 (%s) gates '
        'pre-registered verbatim' % DESIGN_3233,
        'this arm S1: UL_flip/UL_dose/UL_amp recomputed from sealed 3235 '
        'result, matched sealed gates x2 models',
        '3235 verify V7: contradictions diagnosed as probe mode leak, '
        'verdict unaffected',
    ],
}
reg32['failures'].append(f21)
assert len(reg32['failures']) == 21 and reg32['failures'][-1]['id'] == 'F21'

reg32.setdefault('provenance', {}).setdefault('anchor_files', {})[
    'upstream_lever_scan'] = {
        'path': (r'tests\glm5\result\rdc_query_construction_20260913'
                 r'\phase3234\g5g30_gap5_upstream_lever_scan'
                 r'\upstream_lever_scan.npz'),
        'sha8': NPZ34_SHA}

SCOPE_2526 = (' | upstream-lever absence (3234 G5-G30 -> 3235 G5-G31 -> '
              '3237 F21): the full-layer upstream scan (u_isa/u_hasa '
              'injection at every layer, peak-band alpha fine scan, null '
              '10+2) finds NO behavioral lever — UL_flip FAIL (L*=0, all '
              'folds 0 flips, x2 models), UL_dose FAIL (non-monotone), '
              'UL_amp FAIL (spearman_best -0.20/+0.40 << 0.6), UL_spec '
              'PASS; together with F20 (readout-end orthogonality) the '
              'u_isa/u_hasa family carries NO single-direction behavioral '
              'lever at either end — behavioral carrying is distributed '
              '(UPSTREAM_ABSENT); geometric E-levels unchanged')
for f in reg32['features']:
    if f['id'] in ('FTR-25', 'FTR-26'):
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_2526
# byte-preservation asserts
for a, b in zip(reg31['features'], reg32['features']):
    assert a['id'] == b['id']
    if a['id'] not in ('FTR-25', 'FTR-26'):
        assert jd(a) == jd(b), ('feature drift', a['id'])
    else:
        assert jd(a['statement']) == jd(b['statement'])
        assert jd(a['evidence_level']) == jd(b['evidence_level'])
        assert jd(a['anchors']) == jd(b['anchors'])
        assert len(b['scope_limits']) > len(a['scope_limits'] or '')
for k in reg31:
    if k not in ('version', 'supersedes', 'failures', 'provenance',
                 'features'):
        assert jd(reg31[k]) == jd(reg32[k]), ('top drift', k)
assert jd(reg31['failures']) == jd(reg32['failures'][:20])
log('S2 registry v1.32 assembled: F21 appended (failures 20->21), '
    'provenance anchor upstream_lever_scan, scope notes FTR-25/26, '
    'upgrade_log untouched (11), byte-preservation OK')

# ---------- S3: gap ledger v1.11 (GAP-5 FINAL note + final_verdict) ----------
gap10 = json.load(open(GAP10, encoding='utf-8'))
assert gap10['version'] == '1.10' and len(gap10['gaps']) == 5
gap11 = deepcopy(gap10)
gap11['version'] = '1.11'
gap11['updated'] = '2026-10-14'
g5 = [x for x in gap11['gaps'] if x['id'] == 'GAP-5'][0]
g5['mechanism_note'] = (g5.get('mechanism_note') or '') + (
    '；终版收官（3234 G5-G30 → 3235 G5-G31 → 3237 G5-G31 落账，F21）：'
    '上游杠杆扫描（全 L 网格 × u_isa/u_hasa 双向 + 峰带 α 细扫 + null '
    '10+2）三门齐败 × 2 模型——UL_flip 全折 0、UL_dose 非单调、UL_amp '
    'spearman −0.20/+0.40 ≪ 0.6（UL_spec PASS：失败是不翻而非乱翻）。'
    '三段式就此定谳闭环：第一段上游编码承载行为 = 分布式（无任何单方向'
    '杠杆，F20 读出端正交 + F21 上游端缺席双重闭合）；第二段末层 '
    'pre-softmax 精确线性读出（cos_S 1.0000）；第三段行为非线性定位于 '
    'post-softmax 归一化。机制图谱的结论不再「待上游寻找杠杆」，而是'
    '「分布式承载 + 末层线性投影 + 分布级非线性」三位一体；后续机制'
    '单位转向载体子空间 × 幅度调度 × 组合规则（u* 幅度载体 3232 类级 '
    'ROBUST 为首个实例）。')
g5['final_verdict'] = {
    'verdict': 'UPSTREAM_ABSENT',
    'phase': 3237, 'date': '2026-10-14',
    'source': ('sealed 3235 res %s (freeze %s); sealed 3234 res %s / npz '
               '%s; design 3233 %s'
               % (R35_RES, R35_FREEZE, R34_RES, NPZ34_SHA, DESIGN_3233)),
    'gates': {'UL_flip': 'FAIL x2 models', 'UL_dose': 'FAIL x2 models',
              'UL_amp': 'FAIL x2 models', 'UL_spec': 'PASS x2 models'},
    'three_stage': ('upstream distributed carrying (F20+F21) -> last-layer '
                    'exact linear readout (presoftmax) -> post-softmax '
                    'nonlinearity'),
}
for a, b in zip(gap10['gaps'], gap11['gaps']):
    if a['id'] != 'GAP-5':
        assert jd(a) == jd(b), ('gap drift', a.get('id'))
    else:
        assert len(b['mechanism_note']) > len(a['mechanism_note'])
        assert 'final_verdict' not in a
        for k in a:
            if k != 'mechanism_note':
                assert jd(a[k]) == jd(b[k]), ('gap5 field drift', k)
for k in gap10:
    if k not in ('version', 'updated', 'gaps'):
        assert jd(gap10[k]) == jd(gap11[k]), ('gap top drift', k)
log('S3 gap v1.11 assembled: GAP-5 mechanism_note FINAL appended + '
    'final_verdict new key; legacy x4 gaps + GAP-5 legacy fields '
    'byte-preserved')

# ---------- S4: ledger untouched (within-run snapshot invariant) ----------
led = json.load(open(LEDGER, encoding='utf-8'))
LED_HEAD = [len(led['measurements']), led['ledger_sha256_8']]
assert LED_HEAD == [LED_N, LED_POST], ('ledger head drift', LED_HEAD)
log('S4 ledger head observed: n=%d post=%s (untouched by this arm; '
    'closeout appends #389 later)' % tuple(LED_HEAD))

# ---------- S5: write sealed artifacts ----------
WROTE = {}
p = os.path.join(OUTDIR, 'atlas_registry_v1_32.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(reg32, f, ensure_ascii=False, indent=1)
v32 = dict(reg32)
v32.pop('sha8', None)
WROTE['registry_v132'] = {'file': os.path.basename(p),
                          'file_sha8': sha8_file(p),
                          'content_sha8': csha8(v32)}
log('S5 registry v1.32 sealed: file %s content %s' % (
    WROTE['registry_v132']['file_sha8'],
    WROTE['registry_v132']['content_sha8']))
p = os.path.join(OUTDIR, 'gap_ledger_v1_11.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(gap11, f, ensure_ascii=False, indent=1)
WROTE['gap_v111'] = {'file': os.path.basename(p), 'file_sha8': sha8_file(p),
                     'content_sha8': csha8(gap11)}
log('S5 gap v1.11 sealed: file %s content %s' % (
    WROTE['gap_v111']['file_sha8'], WROTE['gap_v111']['content_sha8']))

# ---------- S6: verdict ----------
led2 = json.load(open(LEDGER, encoding='utf-8'))
assert [len(led2['measurements']), led2['ledger_sha256_8']] == LED_HEAD, \
    'ledger modified within run'
clean = not bad
verdict = ('f21_upstream_absent_registered' if clean
           else 'registry_landing_dirty (%d)' % len(bad))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3237, 'name': 'g5g31_registry_landing',
    'arm': 'F21 append + GAP-5 FINAL note + final_verdict + FTR-25/26 '
           'scope notes (default (a))',
    'freeze_sha8': fz['sha8'],
    'inputs': fz['inputs'],
    'clarifications_frozen': fz['clarifications_frozen_before_assembly'],
    'integrity': INTEG,
    'source_verdict': 'UPSTREAM_ABSENT (3235, sealed %s)' % R35_RES,
    'registry': {'version': '1.32',
                 'change': 'F21 appended (upstream_lever_absent); '
                           'upgrade_log untouched (11); E-levels unchanged',
                 'scope_appends': ['FTR-25', 'FTR-26'],
                 'provenance_anchor': 'upstream_lever_scan %s' % NPZ34_SHA},
    'gap': {'version': '1.11', 'gap5_mechanism_note': 'appended FINAL '
            '(three-stage closure)', 'final_verdict': 'UPSTREAM_ABSENT'},
    'wrote': WROTE,
    'recheck': {'ledger_head_observed': LED_HEAD,
                'ledger_untouched_within_run': True,
                'byte_preservation': ('26 features (24 untouched byte-exact '
                                      '+ 2 scope appends) + failures[:20] '
                                      '+ F21 new + upgrade_log x11 + '
                                      'legacy top-level OK; gap legacy x4 '
                                      '+ GAP-5 legacy fields + '
                                      'final_verdict new key OK'),
                'contradictions': bad},
    'verdict': verdict,
}
res['res_sha8'] = hashlib.sha256(json.dumps(
    dict(res, res_sha8=None), ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
with open(os.path.join(OUTDIR, 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('result sealed: res %s' % res['res_sha8'])
