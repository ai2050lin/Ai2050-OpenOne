# -*- coding: utf-8 -*-
"""Phase 3178 closeout: five writes + disk read-back self-check.

Writes:
  1. ledger n=329 -> 330 (append measurement, recompute chain sha8)
  2. MEMO: append Phase 3178 section at physical tail (BOM+CRLF preserved)
  3. daily 2026-10-10.md: create-or-append entry
  4. workspace MEMORY.md: fix 3177 typo (恒乙 -> 恒等) + append 3178 entry (LF file)
  5. self-check: disk read-back of all writes (3164 lesson: disk-based, not memory)
"""
import datetime
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = ROOT + r'\research\gpt5\atlas\atlas_ledger.json'
MEMO = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
DAILY = ROOT + r'\.workbuddy\memory\2026-10-10.md'
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'

NOW = datetime.datetime.now()
TS = NOW.strftime('%H:%M')
CREATED = NOW.strftime('%Y-%m-%dT%H:%M:%S')
LOG = []


def log(s):
    LOG.append(str(s))
    print(s, flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ------------------------------------------------- 1. ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(led['measurements']) == 329, ('ledger n', len(led['measurements']))
assert led['measurements'][-1]['phase'] == 3177, ('last phase', )
entry = {
    'phase': 3178,
    'name': 'g5b3_ftr14_a2_attribution',
    'line': 'G',
    'date': '2026-10-10',
    'model': 'qwen3-4b+qwen3-14b+glm4-9b (zero GPU)',
    'verdict': ('g5b3_ftr14_a2_attribution|relation_specific_residual_confirmed|'
                'calib_fail_x3|r_0.0918_0.0951_0.1289|p_null_0.0193_0.0264|'
                'gap_0.0936_0.1500|p_fold_min_-0.0529|registry_v15|html_fields_669_ok'),
    'evidence_level': 'E1_repeatable',
    'model_scope': 'qwen3-4b+qwen3-14b+glm4-9b',
    'prereg_id': 'G5-B3',
    'superseded_by': None,
    'detail': ('G5-B3 FTR-14 A2 attribution decomposition (zero GPU; sealed 3157 '
               'collect npz 95a25965/9552086d/23dd74eb + sealed 3177 result '
               'd451030f/7b746eb2 loaded from pinned file b2ad52ef, zero '
               'transcription). Entity-LOO null within relation: p(m,rel) = '
               '0.0236/0.0231/0.0193 (isa), 0.0264/0.0216/0.0214 (hasa), fold-level '
               'minima negative to -0.0529 (single-entity directional '
               'self-consistency => out-of-sample penalty approx 0). Relation gap '
               'g = cos_train_a2 - cos_held_a2 (3177 same-v_shared) 0.1145/0.1412/'
               '0.1500 (isa), 0.1190/0.0936/0.1486 (hasa), bitwise replication of '
               'sealed 3177 (drift 0.00e+00). Residual r(m) = mean(isa,hasa) of '
               '(g - p) = 0.0918/0.0951/0.1289 all >= 0.05 calibration bar (gate + '
               'aggregation frozen pre-observation) => 3177 A2 failure is genuinely '
               'relation-specific geometry, NOT sampling artifact; closes '
               'consistently with 3177 directional finding tc_held > tc_full 6/6. '
               'FTR-14 stays E1_repeatable; held_out upgrade route closed with '
               'quantitative attribution (root cause = relation geometry itself). '
               'Registry v1.4 -> v1.5 (FTR-14 counter_evidence append only; 21 '
               'features / failures 16 / upgrade_log 5 byte-for-byte; e1942d03, '
               'SMOKE/FULL byte-identical); atlas_v1_6.html rendered 669 fields '
               '0/0/0/0 (b95e67a2). Device fix: html write newline char preserved '
               '(SMOKE-1 caught Windows text-mode CRLF translation in data-k disk '
               'read-back of newline-containing GAP-4.mechanism_note; zero numeric '
               'impact). res 4f521d9b / seal 1f166645.'),
    'created': CREATED,
}
led['measurements'].append(entry)
led.pop('ledger_sha256_8', None)
blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
led['ledger_sha256_8'] = sha8(blob)
with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)
log('1. ledger: n=330 appended, chain %s' % led['ledger_sha256_8'])

# ------------------------------------------------- 2. MEMO append at physical tail
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
assert '## Phase 3178' not in norm, '3178 section already present'
section = (
    '## Phase 3178: G5-B3 FTR-14 A2 归因分解（零 GPU sampling-null 定判） ' + TS + '\n'
    '\n'
    '**verdict: relation_specific_residual_confirmed（calibration FAIL ×3）——3177 A2 失败主体是'
    '真实的关系特异残差、非采样伪影：实体留出 null 测得纯采样惩罚 p 仅 0.0193-0.0264，关系缺口 g '
    '0.0936-0.1500，残差 r(m)=g−p=0.0918/0.0951/0.1289（4b/14b/glm4）全部不低于 0.05 标定 bar'
    '（门与聚合 r(m)=mean(isa,hasa) 均 freeze 前钉定，门冻结不移动）。FTR-14 保持 E1_repeatable；'
    'registry v1.4→v1.5（FTR-14 counter_evidence 量化追加、21 特征/失败 16/升级链 5 字节保全）；'
    'atlas_v1_6.html 669 字段 0/0/0/0。FTR-14 held_out 升级路线判闭：0.1 门对重建型读数不可达的'
    '根源 = 关系特异几何本身（归因定量完成，非门移动）。**\n'
    '\n'
    '**verdict（重复二）：装置全绿——3157 装置门 tc/exch/exch 均值 drift 全部 0.00e+00；3177 复现门'
    '（cos_full_a2/逐 fold cos_train_a2·cos_held_a2·a2_gap_train/tc_full/exchange_obs 对封存 '
    'result d451030f/7b746eb2，数字从 pinned 文件加载 file sha b2ad52ef 双锁、零转录）max drift '
    '0.00e+00（逐位复现）；SMOKE 与 FULL registry v1.5 字节恒等（e1942d03）=> 数值运行模式无关；'
    'FULL res 4f521d9b / seal 1f166645（SMOKE b0627ca3/a4a22451）；execution.json design '
    'a481b544。**\n'
    '\n'
    '**verdict（重复三）：null 装置副产物发现——p 的 fold 级存在负值（min：4b -0.0308/-0.0458、'
    '14b -0.0491/-0.0529、glm4 -0.0351/-0.0414）：同关系内留出实体 2 细胞对 30 细胞质心的 cos 反而'
    '常高于 in-sample 均值（单实体方向自洽），即 out-of-sample 采样惩罚在该几何里接近零甚至为负；'
    '关系 fold 的 in−out gap 因此几乎全部归因于跨关系成分。与 3177 方向性发现（tc_held 0.581-0.623 '
    '> tc_full 0.561-0.564，6/6）闭合一致：T_C 几何按关系分解——关系内自洽、跨关系系统性偏移 '
    '~0.09-0.13（cos 单位）。**\n'
    '\n'
    '### 归因读数（result 现场渲染）\n'
    '\n'
    '| 读数 | 4b | 14b | glm4 | 门/判定 |\n'
    '|---|---|---|---|---|\n'
    '| g isa / hasa | 0.1145 / 0.1190 | 0.1412 / 0.0936 | 0.1500 / 0.1486 | 3177 封存复现 drift 0 |\n'
    '| p_null isa / hasa | 0.0236 / 0.0264 | 0.0231 / 0.0216 | 0.0193 / 0.0214 | 纯采样惩罚基准 |\n'
    '| r isa / hasa | 0.0909 / 0.0926 | 0.1181 / 0.0720 | 0.1307 / 0.1272 | 逐 fold 披露 |\n'
    '| r(m)=mean(isa,hasa) | 0.0918 | 0.0951 | 0.1289 | <0.05 ×3 全 FAIL |\n'
    '| p fold min | -0.0308 / -0.0458 | -0.0491 / -0.0529 | -0.0351 / -0.0414 | sanity abs(p)<0.5 全过 |\n'
    '\n'
    '### 装置与诚实登记\n'
    '\n'
    'freeze 前钉定：数据源=封存 3157 collect npz（prereg 明示零 GPU）；null 协议 verbatim prereg'
    '（同关系 16 fold、v_null 由其余 15 实体×2 极性 30 细胞重建、留出 2 细胞读数、p(m,rel)=16 fold '
    '均值）；聚合 r(m)=mean(isa,hasa) freeze 时钉定、per-fold 全披露；96 p_fold sanity（finite + '
    'abs(p)<0.5）全过。1 次装置修正（SMOKE-1 拦截）：html 写盘 newline 参数固定——Windows 文本模式 '
    '\\n→\\r\\n 翻译致 GAP-4.mechanism_note（含换行字段）data-k 磁盘回读 mismatch；修正仅影响产物字节'
    '表示、数值与门零影响，修正后 registry 与首跑逐位一致（e1942d03）。PASS 分支不写 registry/html'
    '（升级执行属 3179，prereg 字面）；FAIL 分支按 prereg 量化登记（counter_evidence append only）。'
    'smoke 产物 smoke_ 前缀；产物目录 tests/glm5/result/rdc_query_construction_20260913/'
    'phase3178/g5b3_ftr14_a2_attribution/。\n'
    '\n'
    '### 接续：预注册 3179\n'
    '\n'
    '**3179 = G5-B4 FTR-23 立项入册——T_C 关系特异残差升格独立特征（零 GPU，registry v1.5→v1.6 + '
    'atlas_v1_7.html 渲染）**：依据三原则之「失败必入图谱」的正面化——关系特异残差目前只活在 FTR-14 '
    'counter_evidence 字符串里，而它满足立表条件：3177+3178 两 phase 独立测量、3 模型同向（r '
    '0.0918/0.0951/0.1289 同量级）、预注册门历史完整（F16 gate fail → 3178 归因定量）、锚齐备（p3178 '
    'res 4f521d9b/seal 1f166645 + p3177 res d451030f/seal 7b746eb2 + p3157 collect '
    '95a25965/9552086d/23dd74eb）。FTR-23 statement（草案，freeze 前定稿）=「T_C 共享分量携带关系'
    '特异残差：实体留出 null 标定下纯采样惩罚 p 0.019-0.026，关系残差 r 0.072-0.131 ×3 模型（标定 '
    'bar 0.05）；关系内两两 cos tc_held 0.581-0.623 > 全量 tc_full 0.561-0.564（6/6 fold×模型单元）'
    '——T_C 几何按关系分解，关系内自洽、跨关系系统性偏移」。证据级 E1_repeatable（cross_model 即时'
    '成立；held_out 升级留后续设计——对关系特异特征而言「held_out 关系」本身即测量构造）。协议细节 '
    'freeze 前从 3178 渲染器 verbatim 导出。FTR-04 glm4 弱特异性归因与 FTR-15 held_out 顺位其后；'
    '挂账不变：N 线 P3-P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。\n'
)
sep = '' if norm.endswith('\n') else '\n'
norm2 = norm + sep + section
out = norm2.replace('\n', '\r\n') if crlf > 0 else norm2
if had_bom:
    out_bytes = b'\xef\xbb\xbf' + out.encode('utf-8')
else:
    out_bytes = out.encode('utf-8')
with open(MEMO, 'wb') as f:
    f.write(out_bytes)
log('2. MEMO: 3178 section appended at physical tail (BOM=%s, CRLF=%d)' % (had_bom, crlf > 0))

# ------------------------------------------------- 3. daily
entry_daily = (
    '- **3178 G5-B3 FTR-14 A2 归因分解闭环（2026-10-10 ' + TS + '）**：零 GPU sampling-null 定判'
    '——relation_specific_residual_confirmed（calibration FAIL ×3）：纯采样惩罚 p 0.0193-0.0264'
    '（fold 级有负值 min -0.0529），关系缺口 g 0.0936-0.1500（3177 封存逐位复现 drift 0），残差 '
    'r(m)=0.0918/0.0951/0.1289 全不低于 0.05 => 3177 A2 失败主体 = 关系特异几何（非采样伪影），'
    '与 3177 tc_held>tc_full 6/6 闭合一致；FTR-14 保持 E1（held_out 路线判闭：0.1 门不可达根源='
    '关系几何本身）。registry v1.4→v1.5 e1942d03（FTR-14 counter_evidence 量化追加，SMOKE/FULL '
    '字节恒等）+ atlas_v1_6.html 669 字段 0/0/0/0。装置修正 1 次（SMOKE-1 拦截 html 文本模式 CRLF '
    '翻译致 data-k 回读 mismatch → newline 固定修复，零数值影响）。res 4f521d9b/seal 1f166645；'
    'ledger n=329→330。预注册 3179=G5-B4 FTR-23 立项（关系特异残差升格独立特征，registry v1.6 + '
    'atlas v1.7 渲染，零 GPU）。\n'
)
existed = os.path.exists(DAILY)
if existed:
    dtxt = io.open(DAILY, encoding='utf-8').read()
    if '3178 G5-B3' not in dtxt:
        if not dtxt.endswith('\n'):
            dtxt += '\n'
        dtxt += entry_daily
        with io.open(DAILY, 'w', encoding='utf-8', newline='') as f:
            f.write(dtxt)
    log('3. daily: appended to existing 2026-10-10.md')
else:
    with io.open(DAILY, 'w', encoding='utf-8', newline='') as f:
        f.write(entry_daily)
    log('3. daily: created 2026-10-10.md')

# ------------------------------------------------- 4. workspace MEMORY
wraw = open(WMEM, 'rb').read().decode('utf-8')
wm = wraw.replace('\r\n', '\n')
assert wm.count('字节恒乙') == 1, ('typo count', wm.count('字节恒乙'))
wm = wm.replace('字节恒乙', '字节恒等')
assert '✅ 3178' not in wm, '3178 entry already present'
entry_mem = (
    '- **✅ 3178 FTR-14 A2 归因分解闭环（2026-10-10）**：G5-B3 零 GPU sampling-null 定判——'
    '**relation_specific_residual_confirmed（calibration FAIL ×3）**：实体留出 null（同关系 16 '
    'fold、30 细胞重建 v_null、留出 2 细胞）测纯采样惩罚 p=0.0193-0.0264（fold 级有负值，min '
    '-0.0529——单实体方向自洽致 out-of-sample 惩罚≈0），关系缺口 g=cos_train−cos_held '
    '0.0936-0.1500（3177 封存逐位复现 drift 0.00e+00），残差 r(m)=g−p=0.0918/0.0951/0.1289 全 '
    '不低于 0.05 标定 bar（聚合 mean(isa,hasa) freeze 钉定）=> **3177 A2 失败主体=关系特异几何、'
    '非采样伪影**，与 3177 tc_held>tc_full 6/6 闭合一致；FTR-14 保持 E1、held_out 升级路线判闭'
    '（0.1 门不可达根源=关系几何本身）。registry v1.4→v1.5 e1942d03（FTR-14 counter_evidence 量化'
    '追加、21 特征/失败 16/升级链 5 字节保全，SMOKE/FULL 字节恒等）+ atlas_v1_6.html 669 字段 '
    '0/0/0/0 b95e67a2。装置修正 1 次（SMOKE-1 拦截 html 文本模式 CRLF 翻译 → newline 固定修复，'
    '零数值影响）。res 4f521d9b/seal 1f166645；ledger n=329→**330**。下一步 3179=G5-B4 FTR-23 '
    '立项入册（关系特异残差升格独立特征 E1，registry v1.6 + atlas v1.7 渲染，零 GPU）；FTR-04 '
    'glm4 归因、FTR-15 held_out 后续。\n'
)
if not wm.endswith('\n'):
    wm += '\n'
wm += entry_mem
with open(WMEM, 'wb') as f:
    f.write(wm.encode('utf-8'))
log('4. workspace MEMORY: typo fixed + 3178 entry appended (LF preserved)')

# ------------------------------------------------- 5. self-check (disk read-back)
ok = 0
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
chain_ok = True
led2.pop('ledger_sha256_8', None)
b2 = json.dumps(led2, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
recomputed = sha8(b2)
led3 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(led3['measurements']) == 330 and led3['measurements'][-1]['phase'] == 3178
assert led3['ledger_sha256_8'] == recomputed, ('chain mismatch', led3['ledger_sha256_8'], recomputed)
ok += 1
log('check 1/5 ledger: n=330, phase-3178 last, chain %s verified' % recomputed)

m2 = open(MEMO, 'rb').read()
assert m2.startswith(b'\xef\xbb\xbf'), 'MEMO BOM lost'
t2 = m2.decode('utf-8').lstrip('\ufeff')
assert t2.count('\r\n') == crlf + section.count('\n') + (1 if sep else 0) + 1, \
    'MEMO CRLF count shifted unexpectedly'
assert '## Phase 3178: G5-B3 FTR-14 A2 归因分解' in t2
assert t2.rstrip().endswith('水果类、K4。'), 'MEMO tail unexpected'
ok += 1
log('check 2/5 MEMO: BOM+CRLF preserved, 3178 section at tail')

d2 = io.open(DAILY, encoding='utf-8').read()
assert '3178 G5-B3' in d2 and '4f521d9b' in d2
ok += 1
log('check 3/5 daily: 3178 entry on disk (%s)' % ('existed' if existed else 'created'))

w2 = open(WMEM, 'rb').read().decode('utf-8')
assert '字节恒乙' not in w2 and '字节恒等' in w2
assert '✅ 3178 FTR-14 A2 归因分解闭环' in w2 and '329→**330**' in w2
ok += 1
log('check 4/5 MEMORY: typo fixed, 3178 entry on disk')

# smoke/full artifact identity check (registry)
S = ROOT + r'\tests\glm5\result\rdc_query_construction_20260913\phase3178\g5b3_ftr14_a2_attribution'
a = sha8(open(S + r'\smoke_atlas_registry_v1_5.json', 'rb').read())
b = sha8(open(S + r'\atlas_registry_v1_5.json', 'rb').read())
assert a == b == 'e1942d03', ('registry SMOKE/FULL identity', a, b)
r_full = json.load(io.open(S + r'\result.json', encoding='utf-8'))
assert r_full['res_sha8'] == '4f521d9b' and r_full['seal_sha8'] == '1f166645'
assert r_full['arm']['calibration_pass'] is False
ok += 1
log('check 5/5 artifacts: registry SMOKE/FULL byte-identical e1942d03, FULL res/seal verified')

log('SELF-CHECK %d/5 ALL PASS' % ok)
with io.open(ROOT + r'\tests\gpt5_temp\p3178_closeout_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
print('CLOSEOUT DONE')
