"""Phase 2993 closeout: ledger -> MEMO -> wslog -> MEMORY."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913',
                   'phase2993',
                   'logic_signature_registration')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\tmp_closeout2993_log.txt')
out = []


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


r = json.load(io.open(os.path.join(RES, 'result.json'),
                      encoding='utf-8'))
e = json.load(io.open(os.path.join(RES, 'execution.json'),
                      encoding='utf-8'))
s = json.load(io.open(os.path.join(RES, 'seal.json'),
                      encoding='utf-8'))
assert r['final_verdict'] == \
    'logic_signature_length_robust'
created = e['created']
stamp = '%s %s' % (created[:10], created[11:16])
npz8 = s['npz_sha256_8']
res8 = s['result_sha256_8']
exe8 = e.get('script_sha256_8', 'NA')

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('meas_id') ==
           'meas2993_logic_signature_registration'
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': 'meas2993_logic_signature_registration',
        'phase': 2993,
        'claim': ('Omega-D (plan v4 P3, applicability-'
                  'tagged): a LOGIC/CONNECTIVE signature '
                  'exists as a genuine THIRD class at the '
                  'L34 locus -- on the u35 F-C axis logic '
                  'words sit strictly BETWEEN the poles '
                  '(median proj L2: F 42.7 < L 90.9 < C '
                  '145.2; both contrasts p_fam 4e-4, '
                  'Bonferroni x4) and are NOT absorbed '
                  'into the F pole; the signature is '
                  'length-ROBUST, not context-emergent: '
                  'split-half axes (w2 from L_A@L2, w1024 '
                  'from L_A@L1024, held-out L_B vs C_en) '
                  'are significant at ALL 5 bins {2,16,64,'
                  '256,1024} (p_raw at floor 1/10001, '
                  'p_fam 0.001, d_eff 1.99-2.06 stable); '
                  'head topography rearranges with length '
                  '(topo cos 0.195 p 0.70 within null; '
                  'h15@L2 0.80 -> h8/h15/h21 rotate) yet '
                  'per-length maxT survives -- head = '
                  'routing extends to the logic class'),
        'verdict': r['final_verdict'],
        'anchors': '8/8 (six bit-level 0.00: a4/a5 prof '
                   'vs 2986, a6 headC vs 2987, a8 prof34; '
                   'a1 Vt8 3.04e-08; a1w words three-way; '
                   'a7 determinism 0.00)',
        'artifacts': {
            'result': 'phase2993/'
                      'logic_signature_registration/'
                      'result.json',
            'npz': 'phase2993/'
                   'logic_signature_registration/'
                   'logic_signature_registration.npz'},
        'hashes': {'npz_sha256_8': npz8,
                   'result_sha256_8': res8,
                   'script_sha256_8': exe8},
        'note': ('run1 died at T3 key-name bug '
                 '(c2.split("-") on single char); products '
                 'deleted, run2 authoritative and clean. '
                 'Confound registered: L class = high-freq '
                 'single-token connectives, frequency/'
                 'length not matched to C nouns -- tag '
                 'valid within this protocol only '
                 '(plan v4 P0 applicability label)')})
    for l in led['linkage']:
        if l['link_id'] == \
                'L14_readout_spectrum_cross_model':
            cons = l['connects']
            if not any(isinstance(c, dict)
                       and c.get('phase') == 2993
                       for c in cons):
                cons.append({
                    'phase': 2993,
                    'via': 'logic_signature_registration',
                    'adds': 'logic words = third class '
                            '(between F and C poles, '
                            'distinct from both) with '
                            'length-robust split-half '
                            'signature d~2.0 at all '
                            'bins; F/C collapse with '
                            'context spares logic '
                            'detectability'})
led.pop('ledger_sha256_8', None)
h = hashlib.sha256(json.dumps(
    led, sort_keys=True,
    ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = h
with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)
n_meas = len(led['measurements'])
l14 = [l for l in led['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model']
n14d = len([c for c in l14[0]['connects']
            if isinstance(c, dict)])
n14t = len(l14[0]['connects'])
out.append('ledger n=%d L14 dict=%d total=%d hash=%s'
           % (n_meas, n14d, n14t, h))

# ---------- 2. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tag = '## Phase 2993:'
if tag not in memo:
    sec = (
        '## Phase 2993: Omega-D 逻辑签名——第三类在场且长度稳健 '
        '[%(stamp)s]\n\n'
        '**判决：`logic_signature_length_robust`**（run2 权威，'
        '42.8s，锚 8/8，其中六重 bit 级 0.00：prof/headC 对 2986/'
        '2987 全 bit 复现）。\n\n'
        '### 设计\n'
        '方案 v4 P3 顺延（v3 Omega-D），全程带适用域标签：74 基线词'
        '（2977 口径）+ L_EN 连接词 23 个（预注册 24 候选，单 token '
        '过滤，字母序确定分半 L_A=11 轴构造 / L_B=12 保留检验，'
        '抗循环）；长度档 %(lens)s × 2986 make_seq verbatim；读出'
        '三件套：prof_all 沿 M[li]=u35@Wo（2986/2987 verbatim）、'
        'L34 头级谱 C34 切片（2991/2992 verbatim）、res34=L34 层'
        '输入残差（d_lang_u 口径）。T1 主判据：分半泛化（轴仅用 '
        'L_A，在 L2 与 L1024 各建一条，L_B vs C_en 逐档置换 '
        'N=10000 双侧，Bonferroni ×10）；T2 次级头级 maxT 探索；'
        'T3 u35 F-C 轴位置描述确证（L-C、L-F × L2、L1024，×4）。\n\n'
        '### 核心结果（重复三遍）\n'
        '**逻辑词是 F/C 机器之外的真第三类，且其签名不是上下文涌现的：'
        'u35 轴上逻辑词严格居于两极之间（L2 中位投影 F 42.7 < L 90.9 '
        '< C 145.2，L-C 与 L-F 双对比 p_fam 4e-4）而非被 F 极吸收；'
        '分半轴（w2、w1024 双轴）在全部五个长度档对保留集 L_B 显著'
        '（p_raw 全触地板 1/10001，p_fam 0.001，d_eff 1.99-2.06 '
        '稳定）——方案 v3 "逻辑聚合是否随上下文出现"的回答：否定，'
        'L2 已在场、全档位稳健。**\n\n'
        '**第三类在场 + 长度稳健：F<L<C 严格居中（双侧 distinct），'
        '五档 d≈2.0 全显著；非涌现。**\n\n'
        '- T3 长度端：L1024 时 F 73.6 < L 94.7 < C 108.5——两极向'
        '中间收敛（与 2986 F/C 可检测性塌缩一致）但逻辑类仍双侧可分'
        '（p 0.0052 / 0.0004）；\n'
        '- T2 探索性：L2 头级对比 top-1 = h15（0.80，p_maxT 0.004）'
        '——与 F/C 载体头同位；随长度头拓扑重排（L1024 topo cos '
        '0.195，p 0.70 在 null 内；h8/h15/h21 轮替）但逐档 maxT 仍'
        '有存活头——2991 "头=路由"结论延伸到逻辑类；\n'
        '- 适用域标签（plan v4 P0）：T1=长度稳健(L2-1024 全档)；'
        'T3=确证性(×4)；T2=探索性(maxT 参考不作确证)。混淆登记：'
        'L 类为高频单 token 连接词，频率/词长未与 C 名词匹配——标签'
        '仅在本协议内有效。\n\n'
        '### 硬伤（如实登记，删产物重跑）\n'
        '1. run1 崩于 T3 结果键名 bug（c2.split("-") 作用在单字符'
        '"C"/"F" 上 IndexError，崩在收尾构造、扫描与锚已过）；删产物'
        '后 run2 权威重跑，判决确定性成立。\n\n'
        '### 产物\n'
        '- result.json sha256_8=%(res8)s；npz sha256_8=%(npz8)s；'
        'script sha256_8=%(exe8)s；execution created=%(created)s。\n\n'
        % {'stamp': stamp, 'lens': '{2,16,64,256,1024}',
           'res8': res8, 'npz8': npz8, 'exe8': exe8,
           'created': created})
    memo = memo.rstrip('\n') + '\n\n' + sec
    io.open(MEMO, 'w', encoding='utf-8').write(memo)
    out.append('memo appended')
else:
    out.append('memo already has 2993')
tail = memo.split(tag)[1] if tag in memo else ''
residue = '%(' in tail
out.append('memo residue %% = %s' % residue)

# ---------- 3. workspace log ----------
ws = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\2026-09-20.md')
line = ('- Phase 2993 Omega-D 逻辑签名：判决 '
        'logic_signature_length_robust（run2，锚 8/8 六重 bit 级）。'
        '逻辑词=第三类（u35 轴 F<L<C 严格居中，双对比 p_fam 4e-4）且'
        '长度稳健（分半轴五档 d≈2.0 p 触地板）——非上下文涌现；头拓扑'
        '随长度重排（cos 0.20 null 内）。硬伤 1 笔（T3 键名 bug）'
        '登记。Ledger 132 / L14 100 / hash %s。\n' % h)
wtxt = io.open(ws, encoding='utf-8').read() \
    if os.path.exists(ws) else ''
if 'Phase 2993' not in wtxt:
    if not wtxt:
        io.open(ws, 'a', encoding='utf-8').write(
            '# 2026-09-20 工作日志\n')
    io.open(ws, 'a', encoding='utf-8').write(line)
    out.append('wslog appended')

# ---------- 4. MEMORY.md ----------
MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
old2992 = ('2992 稀疏字典：对齐超 null（p_fam 0.014，cos 峰 '
           '0.10）但符号一致率 0.405<0.5——特征对齐=快照属性，'
           'SAE 不立项。')
add993 = ('2993 Ω-D 逻辑签名：第三类在场（u35 轴 F<L<C 双侧 '
          'distinct p 4e-4）且长度稳健（分半 d≈2.0 五档触地板）'
          '——非上下文涌现。')
if add993 not in mm:
    if old2992 in mm:
        mm = mm.replace(old2992, old2992 + add993, 1)
    else:
        out.append('WARN 2992 anchor miss')
mm = mm.replace('## 机制链状态（2936-2992）',
                '## 机制链状态（2936-2993）', 1)
old_next = ('- max=2992，下一个 **2993**（A 主选 P3 顺延：Ω-D 逻辑'
            '签名带适用域标签；B Ω-E 竞争-滞后操作化；C 2989 T3 '
            '加密+k 剂量；D 字典原子×注册表因果耦合）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
new_next = ('- max=2993，下一个 **2994**（A 主选 P3 顺延：Ω-E 竞争-'
            '滞后操作化，"相变"命名门未变；B Ω-F 跨模型 GLM4-9B '
            '卡片复制——顺序不可反；C 2989 T3 加密+k 剂量；D 逻辑'
            '签名头级因果复测）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
if old_next in mm:
    mm = mm.replace(old_next, new_next, 1)
else:
    out.append('WARN next anchor miss')
if len(mm) > 3000:
    out.append('WARN memory %d chars, extra compress'
               % len(mm))
    for a, b in (
            ('（重叠 41-66/128，z 30-55）',
             '（重叠 41-66/128）'),
            ('2988 普查复制+卡片集 v2；',
             '2988 普查+卡片集 v2；'),
            ('（双侧 distinct p 4e-4）',
             '（双对比 p 4e-4）')):
        mm = mm.replace(a, b, 1)
io.open(MP, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2993=%s next2994=%s'
           % (len(mm), 'max=2993' in mm,
              'next **2994**' in mm))

out.append('CLOSEOUT DONE')
io.open(LOG, 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
