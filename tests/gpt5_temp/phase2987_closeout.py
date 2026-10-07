# -*- coding: utf-8 -*-
"""Phase 2987 closeout: seal -> Ledger -> MEMO ->
workspace log -> MEMORY.md. Idempotent guards."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913', 'phase2987',
    'context_minimal_audit')
SCRIPT = os.path.join(
    BASE, 'tests', 'glm5',
    'phase2987_context_minimal_audit.py')
LEDGER = os.path.join(BASE, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
MEMO = os.path.join(BASE, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMO_MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory\MEMORY.md')
LINK_ID = 'L14_readout_spectrum_cross_model'
OUTLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2987_log.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


out = []
res = json.load(io.open(os.path.join(RES, 'result.json'),
                        encoding='utf-8'))
exec_path = os.path.join(RES, 'execution.json')
exec_json = json.load(io.open(exec_path, encoding='utf-8'))
verdict = res['final_verdict']
assert verdict == 'collapse_at_single_token'
stamp = exec_json['created']
s_exec = sha8(exec_path)
s_res = sha8(os.path.join(RES, 'result.json'))
s_npz = sha8(os.path.join(
    RES, 'context_minimal_audit.npz'))
s_scr = exec_json.get('script_sha256_8')
if s_scr is None:
    s_scr = sha8(SCRIPT)
    exec_json['script_sha256_8'] = s_scr
    exec_json['script_sha256_8_note'] = (
        'registration-only append at closeout')
    io.open(exec_path, 'w', encoding='utf-8').write(
        json.dumps(exec_json, indent=2, ensure_ascii=False))
    s_exec = sha8(exec_path)
    out.append('script hash appended to execution.json')
out.append('hashes: exec=' + s_exec + ' res=' + s_res
           + ' npz=' + s_npz + ' script=' + s_scr)

# ---------- 1. seal ----------
seal_path = os.path.join(RES, 'seal.json')
if not os.path.exists(seal_path):
    seal = {'phase': 2987,
            'verdict': verdict,
            'sealed_at': stamp,
            'sha256_8': {'execution': s_exec,
                         'result': s_res,
                         'npz': s_npz,
                         'script': s_scr},
            'anchors_ok': res['anchor_all_ok'],
            'note': 'run1 authoritative; anchors 9/9 with '
                    'six bit-level 0.00 incl a5 direct '
                    'cross-product identity vs 2986 npz '
                    '(L16N prof34+BL) and a9 head-sum '
                    'identity 7.1e-15; one correction '
                    'pre-run (head census index ci3->ci2) '
                    'before freeze, no artifact produced'}
    io.open(seal_path, 'w', encoding='utf-8').write(
        json.dumps(seal, indent=2, ensure_ascii=False))
    out.append('seal written')
else:
    out.append('seal already present')

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
meas = [m for m in led['measurements']
        if m.get('phase') == 2987]
if not meas:
    m = {'phase': 2987,
         'name': 'context_minimal_audit',
         'model': 'qwen3-4b',
         'created': stamp,
         'question': ('where is the collapse boundary of '
                      'the len-2-only L34 F/C signature '
                      '(minimal ladder + filler content '
                      'controls), and which key chain '
                      'cards survive context: word-class '
                      'signature (2962), language effect '
                      '(2963), h15 carrier (2964)'),
         'design': ('5 conditions x 74 words: L2 '
                    '[the,w] verbatim; L3 one neutral '
                    'filler token; L16N neutral sentence '
                    'filler; L16R seeded random-token '
                    'filler (rng 2987, pool 35); L16T '
                    '14x " the"; per-condition independent '
                    'single-sample forwards; readouts: '
                    'prof34 F/C and en/fr contrasts '
                    '(permutation 10000, two-sided), '
                    'per-head L34 contributions via '
                    'C_row=u35@Wo34 slices, D34 drift'),
         'anchors': ('9/9 ok; a2/a3/a6/a7/a8 bit-level '
                     '0.00; a5 L16N prof34+BL vs 2986 '
                     'npz bit-level 0.00 (direct '
                     'cross-product identity); a9 '
                     'head-sum identity 7.11e-15; a1 '
                     '3.04e-08; a4 74/74'),
         'tests': ('T1 sigA p=[.0348,.342,.675,.500,.174] '
                   '-> collapse_at_single_token (L2 only); '
                   'T2 collapse_content_independent '
                   '(N/R/T p .67/.50/.17); T3 '
                   'lang_effect_robust p=[0,0,.0003,0,0] '
                   'sig [-3.87,-3.78,-2.02,-4.17,-1.75]; '
                   'T4 carrier_h15_mixed (p .040 at L2 '
                   'only; L16N migration census: h11 '
                   'p=.0008, h23 p=.0185); T5 '
                   'restructure_gradual (D34(L3)=1.00 = '
                   '15.8% of D34(L16N)=6.34)'),
         'verdict': verdict,
         'sha256_8': {'execution': s_exec,
                      'result': s_res,
                      'npz': s_npz,
                      'script': s_scr}}
    led['measurements'].append(m)
    lk = None
    for l in led['linkage']:
        if l['link_id'] == LINK_ID:
            lk = l
    lk['connects'].append(
        {'phase': 2987,
         'note': ('minimal-ladder audit: F/C signature '
                  'collapse is triggered by a SINGLE '
                  'filler token and is content-'
                  'independent (N/R/T) - len-2 protocol '
                  'is maximally special; DISSOCIATION: '
                  'signature collapse (instant) vs bulk '
                  'readout drift (gradual, L3 only 16% '
                  'of L16N); language effect robust in '
                  'all conditions; h15 carrier len-2 '
                  'only with head migration to h11/h23 '
                  'under context - carrier identity is '
                  'protocol-relative, cards 2962/2964 '
                  'carry len-2 scope tags, card 2963 '
                  'robust'),
         'connects_to': [2986, 2962, 2963, 2964, 2973,
                         2979]})
    if 'ledger_sha256_8' in led:
        led.pop('ledger_sha256_8')
    newh = hashlib.sha256(json.dumps(
        led, sort_keys=True,
        ensure_ascii=False).encode(
        'utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = newh
    io.open(LEDGER, 'w', encoding='utf-8').write(
        json.dumps(led, indent=2, ensure_ascii=False))
    out.append('ledger: n=%d L14=%d newhash=%s'
               % (len(led['measurements']),
                  len(lk['connects']), newh))
else:
    out.append('ledger already has 2987')

# ---------- 3. MEMO ----------
memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2987:' not in memo_txt:
    sec = (
        '## Phase 2987: context 最小阶梯审计——单 token 即坍缩、'
        '与漂移量解离、lang 效应稳健、载体迁移 [' + stamp + ']\n\n'
        '**问题**（2986 插队）：L34 F/C 词类签名的 len-2 专属性'
        '坍缩边界在哪（最小阶梯 + 填充物内容对照）？三张关键卡'
        '（2962 词类签名 / 2963 语言效应 / 2964 h15 载体）在'
        '上下文下的存活？\n\n'
        '**设计（冻结，370 前向）**：5 条件 x 74 词——L2=[the,w] '
        'verbatim；L3=1 个中性填充 token；L16N=中性句填充；'
        'L16R=种子随机 token 池（rng 2987，池 35）；L16T=14x'
        '" the"；逐条件独立单样本前向。读出：L34 读出 F/C 与 '
        'en/fr 对比（双侧置换 10000）、逐头贡献 C_row=u35@Wo34 '
        '切片、D34 漂移。\n\n'
        '**产物**：`phase2987/context_minimal_audit/` execution '
        + s_exec + ' / result ' + s_res
        + ' / context_minimal_audit.npz ' + s_npz
        + ' / script ' + s_scr + '。\n\n'
        '**锚 9/9**：a2/a3/a6/a7/a8 bit 级 0.00；**a5 L16N '
        'prof34+BL vs 2986 npz bit 级 0.00（跨产物直接对账）**；'
        'a9 头和恒等 sum_h headC == prof34 = 7.11e-15；a1 '
        '3.04e-08；a4 74/74。\n\n'
        '**结果**：T1（主判据）sigA p=[0.0348, 0.342, 0.675, '
        '0.500, 0.174]——**一个填充 token（L3）即坍缩**。T2 '
        '内容对照 N/R/T p=.67/.50/.17——坍缩与填充内容无关。'
        'T3 语言效应 p=[0, 0, 0.0003, 0, 0]——**全条件稳健**'
        '（幅度 -3.87→-1.75 波动）。T4 h15 载体仅 L2 边缘显著'
        '（p=.040），L16N 迁移普查：**h11 p=.0008、h23 '
        'p=.0185**——载体身份是协议相对的。T5 D34(L3)=1.00 = '
        'D34(L16N)=6.34 的 15.8%——重构渐进。\n\n'
        '**判决**：`collapse_at_single_token`（冻结映射：'
        'p(L3)>=0.01 分支）。\n\n'
        '**结论（重复三遍）**：F/C 词类签名坍缩由**单个上下文'
        'token 触发且与内容无关**——len-2 协议是最大特例；'
        '**签名坍缩（瞬时）与读出漂移量（渐进，L3 仅 16%）'
        '解离**——坍缩不是漂移量的副产品；语言效应全条件稳健'
        '（2963 卡带 robust 标签）；h15 载体 len-2 专属且在'
        '上下文下向 h11/h23 迁移——**载体身份是协议相对的**，'
        '2962/2964 卡携带 len-2 适用域标签。\n\n'
        '**勘误**：预运行审查修正头普查索引 ci3→ci2（冻结前，'
        '无产物）；run1 权威一次通过。\n\n'
        '**接续**：方案 v4（微观-宏观合流）已立：2988 起按 '
        'plan_v4_micro_macro_merge.md 推进（P0 适用域普查 '
        '→ P1 Ω-G MLP 神经元级注册表 → P2+ Ω-D/E/F）。\n')
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + sec)
    out.append('memo appended')

# ---------- 4. workspace log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2987' not in wl:
    entry = (
        '\n## Phase 2987（2026-09-20）\n'
        '- context 最小阶梯审计：判决 collapse_at_single_token'
        '（run1 权威，锚 9/9 含 a5 跨产物 bit 级对账 + a9 恒等 '
        '7.1e-15）；F/C 签名单 token 即坍缩、content 无关；'
        '与漂移量解离（L3 仅 16% 漂移）。\n'
        '- 三卡分层：2962 len-2 专属 / 2963 全条件稳健 / '
        '2964 h15 len-2 专属且迁移 h11（p 8e-4）——载体身份'
        '协议相对。\n'
        '- Ledger 126 / L14 94。产物 '
        'phase2987/context_minimal_audit/。\n')
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    out.append('wslog appended')

# ---------- 5. MEMORY.md ----------
mem = io.open(MEMO_MEM, encoding='utf-8').read()
old_chain = ('2986 Ω-C 开题：漂移峰 L16 非单调='
             '上下文存在性主导；L34 词类签名 '
             'len-2 专属（全链适用域警示）。'
             '核心：')
new_chain = ('2986 漂移峰 L16 非单调；2987 最小阶梯：'
             '签名单 token 即坍缩（content 无关）且与'
             '漂移量解离；lang 效应稳健、h15 len-2 '
             '专属迁移 h11（载体协议相对）。核心：')
changed = False
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
else:
    out.append('WARN: chain anchor not found')
old_next = ('- max=2986，下一个 **2987**（A 主选 插队'
            '审计：context 最小对照+关键卡 L16 复测；'
            'B 填充物对照；C dose law；D r_lang）。')
new_next = ('- max=2987，下一个 **2988**（方案 v4 P0 '
            '适用域普查：34 卡 len-2 专属/稳健分层'
            '标注，s_c/词盲卡 L16 复测；随后 P1 Ω-G '
            'MLP 神经元级注册表）。')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
else:
    out.append('WARN: MEMORY next-anchor not found')
if changed:
    io.open(MEMO_MEM, 'w', encoding='utf-8').write(mem)
out.append('memory chars=%d ok3000=%s'
           % (len(mem), len(mem) <= 3000))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('closeout done')
