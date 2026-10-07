# -*- coding: utf-8 -*-
"""Phase 2997 closeout: Ledger 136 -> L14 -> MEMO append ->
workspace log -> MEMORY.md."""
import hashlib
import io
import json
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2997'
     r'\omega_f2b_registry_regrade')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
LOGF = R + r'\closeout_log.txt'
o = []


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['final_verdict']
assert verdict == 'regrade_replicated_registered', verdict
assert res['anchor_all_ok'] is True

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
assert all(m.get('phase') != 2997
           for m in led['measurements'])
claim = (
    'Omega-F2b registry REGRADE (card-set v2.1 '
    'registration). Artifact-level degeneracy reproduced '
    'independent of narrative: 2995 npz dirs_attn/dirs_mlp '
    'all 40 rows identical (max|diff|=0.0/0.0). Frozen '
    'forensics verified bit-level: 2995 grades dict '
    '{T1_M2963 replicated, T2_M2947 replicated, '
    'T3_M2989 not_replicated}; 2995 T3 z [-1.75,-2.25,'
    '-1.89]; 2996 verdict registry_robust_across_calibers '
    'with seal d03a47b0 == ledger tail; 2996 T3.k2 z '
    '[+30.55,+10.41,+10.15]; share p=0.0 at glm L10 and '
    'qwen L6; glm_sig True. Frozen regrade mapping applied '
    '-> cards_v21: exactly ONE regrade (T3_M2989 '
    'not_replicated -> replicated), unregraded cards '
    'untouched, every card carries basis-hash provenance '
    '(2996 result sha8). Card-set v2.1 rule registered: no '
    'regrade without a sealed basis. Anchors 6/6.')
meas = {
    'meas_id': 'meas2997_omega_f2b',
    'phase': 2997,
    'claim': claim,
    'verdict': verdict,
    'anchors': '6/6 (a1 degeneracy 0.0/0.0 bit-level; '
               'a2 grades==frozen; a3 z diff 0.0; '
               'a4 verdict+seal+ledger-tail agree '
               'd03a47b0; a5 z diff 0.0 + glm_sig + '
               'share p=0 x2; a6 n_regraded=1, '
               'map_valid)',
    'artifacts': {
        'result': 'phase2997/omega_f2b_registry_regrade/'
                  'result.json',
        'npz': 'phase2997/omega_f2b_registry_regrade/'
               'omega_f2b_registry_regrade.npz',
        'registry': 'phase2997/omega_f2b_registry_regrade/'
                    'cards_v21.json'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8': seal['result_sha256_8'],
        'script_sha256_8': exe['script_sha256_8']},
    'note': 'pure artifact-level audit card (no model '
            'forward, 0.07s); first run authoritative; '
            'card-set v2.1 adds basis-hash provenance '
            'column; tags: lang axis, len-2, snapshot '
            'only, glm4-9b panel',
}
led['measurements'].append(meas)
assert len(led['measurements']) == 136
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
assert all((c.get('meas_id')
            if isinstance(c, dict) else c)
           != 'meas2997_omega_f2b'
           for c in l14['connects'])
l14['connects'].append({
    'meas_id': 'meas2997_omega_f2b',
    'phase': 2997,
    'axis': 'lang',
    'verdict': verdict,
    'grade_change': 'T3_M2989: not_replicated -> '
                    'replicated (basis 2996 d03a47b0)'})
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
led['ledger_sha256_8'] = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)
o.append('ledger n=%d l14=%d sha=%s'
         % (len(led['measurements']),
            len(l14['connects']),
            led['ledger_sha256_8']))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 2997:' not in memo
sec = u'''## Phase 2997: Ω-F2b 注册表重分级——卡片集 v2.1 登记 [%(created)s]

**判决：`regrade_replicated_registered`**（run1 权威，0.07s，锚 6/6；纯产物级审计卡，无模型前向）

### 设计（预注册冻结）
问题：2996 的产物级证据链是否足以支撑冻结重分级映射 ('T3_M2989','not_replicated')→'replicated'，且登记不变式成立。三任务：T1 伪影级退化复现（2995 npz dirs_attn/dirs_mlp 各 40 行 max|row−row0|==0，bit 级，不依赖任何转述）+ 2995 grades 冻结取证 + 2995 T3 z 与冻结值 |Δ|<1e-9；T2 审计链三方一致（2996 verdict `registry_robust_across_calibers` + seal result_sha8 d03a47b0 + ledger 末条同 hash 同 verdict）+ 2996 T3.k2 z [+30.55,+10.41,+10.15] |Δ|<5e-3 + glm_sig + share p=0.0（glm L10、qwen L6）；T3 冻结映射应用于 2995 分级表 → cards_v21，不变式：恰一次改判、旧值 not_replicated、新值 replicated（在分级词表内）、未改判卡 bit 级不动、每卡 basis 引用 2996 hash + 退化复现值 0.0。

### 结果
锚 6/6 全过：a1 退化 0.0/0.0（bit 级）；a2 grades 与冻结取证完全相等；a3 z 差 0.0；a4 三方一致（ledger 末条 phase=2996 hash=d03a47b0）；a5 z 差 0.0 + share p=0×2；a6 恰一次改判、映射合法。**卡片集 v2.1 落地：T3_M2989 not_replicated→replicated（2995 面板三卡中唯一改判；T1_M2963/T2_M2947 保持 replicated 不变），cards_v21.json 增设 basis-hash 溯源列——新登记纪律：无 sealed basis 不得重分级。**

### 意义
2996 审计的结论从此有独立于叙事的产物级锚（任何人重跑 2997 即可复核改判合法性）；Ω-F1 GLM4 面板由 partial 升为全 replicated（三卡一致），跨模型注册表复制正式入册。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase2997/omega_f2b_registry_regrade/`（result/execution/seal/cards_v21/npz/run_log）；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 136 / L14 %(l14)d。

**接续**：候选 2998：A（主选）GLM4 rotation-target 链重建（Ω-F2 s_c 注入卡前置：2945 注入机器 GLM4 版）；B qwen K2 L6 早层反对齐剖面（weight-side z=−21.9 的深度结构）；C 2989 T3 加密复测 + k 剂量；D 2994 Ω-E 错误吸引子操作化。
''' % {'created': created,
       'script8': exe['script_sha256_8'],
       'result8': seal['result_sha256_8'],
       'npz8': seal['npz_sha256_8'],
       'exec8': seal['exec_sha256_8'],
       'l14': len(l14['connects'])}
memo += '\n' + sec
with io.open(MEMO, 'w', encoding='utf-8') as f:
    f.write(memo)
o.append('memo +%d chars' % len(sec))

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
line = ('- Phase 2997 Omega-F2b regrade: verdict '
        'regrade_replicated_registered, anchors 6/6, '
        'T3_M2989 not_replicated->replicated, cards_v21 '
        'with basis-hash provenance; ledger 136/L14 %d.\n'
        % len(l14['connects']))
if 'Phase 2997' not in prev:
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
o.append('wlog appended')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
