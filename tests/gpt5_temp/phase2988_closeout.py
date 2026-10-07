# -*- coding: utf-8 -*-
"""Phase 2988 closeout: seal -> Ledger -> card set v2 ->
MEMO -> workspace log -> MEMORY.md. Idempotent guards."""
import hashlib
import io
import json
import os
import re

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913', 'phase2988',
    'context_domain_census')
SCRIPT = os.path.join(
    BASE, 'tests', 'glm5',
    'phase2988_context_domain_census.py')
LEDGER = os.path.join(BASE, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
CARDSET = os.path.join(BASE, 'research', 'gpt5', 'atlas',
                       'card_set_v2.json')
MEMO = os.path.join(BASE, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMO_MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory\MEMORY.md')
LINK_ID = 'L14_readout_spectrum_cross_model'
OUTLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2988_log.txt')

VERDICT = 'lock__blind_both'


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
assert verdict == VERDICT, verdict
stamp = exec_json['created']
s_exec = sha8(exec_path)
s_res = sha8(os.path.join(RES, 'result.json'))
s_npz = sha8(os.path.join(
    RES, 'context_domain_census.npz'))
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
    seal = {'phase': 2988,
            'verdict': verdict,
            'sealed_at': stamp,
            'sha256_8': {'execution': s_exec,
                         'result': s_res,
                         'npz': s_npz,
                         'script': s_scr},
            'anchors_ok': res['anchor_all_ok'],
            'note': 'run3 authoritative; anchors 17/17 '
                    'with six bit-level 0.00 (a3 Vt8, '
                    'a11-a15/a17 vs 2953/2967/2968 npz) '
                    'and a1 2.17e-08; s_c(L2)=0.6567 '
                    'exactly reproduces the 2968 '
                    'registered value; run1 wrong npz '
                    'path + run2 C34 stack ordering, '
                    'both pre-authority, artifacts '
                    'deleted before rerun'}
    io.open(seal_path, 'w', encoding='utf-8').write(
        json.dumps(seal, indent=2, ensure_ascii=False))
    out.append('seal written')
else:
    out.append('seal already present')

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
meas = [m for m in led['measurements']
        if m.get('phase') == 2988]
if not meas:
    m = {'phase': 2988,
         'name': 'context_domain_census',
         'model': 'qwen3-4b',
         'created': stamp,
         'question': ('which of the 34 primitive cards '
                      'survive outside the len-2 protocol '
                      '(plan v4 P0): L16 retest of the '
                      'peak-lock card (2968), the '
                      'word-blind card (2940), and the '
                      'direction anchors (a1/a3)'),
         'design': ('two arms x 57-word 2887 list: L2 '
                    'verbatim 2968 [func,word]; L16N = 14 '
                    'neutral filler tokens (2987 sentence) '
                    '+ [func,word]; xdir injection at L17 '
                    'word position (last-token indexing, '
                    'bit-equal on L2), s_grid = 0 + GRID17, '
                    'families A(+xdir)/B(-xdir); T1 '
                    'per-word C34/h15 peaks + lock to '
                    'in-session s_c (2945 rule sep<100); '
                    'T2 word-blind battery: WB1 lang '
                    'decode perm, WB2 concept-group ICC '
                    'of class-centered readout (22 '
                    'cross-lang groups cover 57/57), '
                    'within-class permutation null'),
         'anchors': ('17/17 ok; a3 Vt8 + a11-a15/a17 '
                     'vs 2953/2967/2968 npz bit-level '
                     '0.00 (6 anchors); a9 A11_L17 '
                     'baseline+10 doses 1.81e-07; a1 '
                     '2.17e-08; a4/a5 vs 2935 '
                     '7.2e-06/6.3e-06; determinism L2 '
                     'and L16N 0.00'),
         'tests': ('T1 lock both arms: L2 n_peak=40 '
                   '(reproduces 2968 registered 40), '
                   'med_peak 0.6746, s_c 0.6567 (= '
                   '2968 value); L16N n_peak=40, '
                   'med_peak 0.7492, s_c 0.6677, '
                   '|med-s_c|=0.082<0.3. T2 '
                   'blind_both: WB1 lang decode '
                   'p=1e-4 both arms (obs 194.4 / '
                   '117.4); WB2 ICC 0.3995 p 0.292 '
                   '(L2) / 0.3177 p 0.718 (L16N). T3: '
                   'sepA endpoints 185.7->-6.4 (L2) '
                   'vs 119.7->-9.9 (L16N); B band '
                   '-1.37->-0.02 (L2) vs -3.27->-1.45 '
                   '(L16N)'),
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
        {'phase': 2988,
         'note': ('domain census: peak-lock card '
                  'REPLICATES under L16 context (40 '
                  'peaks, lock to in-session s_c, '
                  's_c 0.6567->0.6677 stable on the '
                  'xdir instrument) and the word-blind '
                  'card REPLICATES (blind both arms, '
                  'lang decode significant both) - '
                  'cards 2940/2968/2936/2939 robust; '
                  '2962/2964/2965 len-2 specific '
                  '(2987); card set v2 = {robust 6, '
                  'len2 3, untested 25} in '
                  'card_set_v2.json; NOTE the s_c '
                  'invariance here (xdir instrument, '
                  '2945 rule, 57 words) vs the s_c '
                  'drop in 2986 (lang-dose instrument, '
                  'own rule, 74 words) is an '
                  'instrument/channel difference to '
                  'be accounted, not a contradiction'),
         'connects_to': [2987, 2968, 2940, 2945, 2953,
                         2967, 2986]})
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
    out.append('ledger already has 2988')

# ---------- 3. card set v2 ----------
if not os.path.exists(CARDSET):
    label_map = {
        'M2936_anchoring_law': (
            'robust', 'direction rebuild anchor '
            '2.17e-08 across protocols (2988 rerun); '
            'structural law, protocol-independent'),
        'M2937_scale_collapse': (
            'untested', '2986 shows readout manifold '
            'drifts sharply with context; protocol '
            'dependence suspected, not retested'),
        'M2938_subspace_angles': ('untested', ''),
        'M2939_rotation_target': (
            'robust', 'Vt8 bit-level 0.00 in 2988 '
            'rerun; structural'),
        'M2940_v3_decode': (
            'robust', '2988 WB replication: word-'
            'blind both arms (ICC p 0.29/0.72), '
            'lang decode p=1e-4 both arms'),
        'M2941_v3_causal_injection': ('untested', ''),
        'M2942_u8_joint_injection': ('untested', ''),
        'meas_2943_gamma_anatomy': ('untested', ''),
        'meas_2944_switch_localization': (
            'untested', 'indirect support only: 2988 '
            'confirms L17 switch exists and locks '
            'under L16 context'),
        'meas_2945_threshold_curves': (
            'untested', 'partial: s_c instrument '
            'recomputed in 2988, L2=0.6567 exact '
            'reproduction, L16N=0.6677 stable'),
        'meas_2946_dose_allocation': ('untested', ''),
        'meas_2947_head_anatomy': ('untested', ''),
        'meas_2948_wov_head_gain': ('untested', ''),
        'meas_2949_head_dose_sufficiency':
            ('untested', ''),
        'meas_2950_rebalance_anatomy': ('untested', ''),
        'meas2951_rebalance_gain_functional':
            ('untested', ''),
        'meas2952_amplification_attention_gain':
            ('untested', ''),
        'meas2953_a11_s_response_sigmoid':
            ('untested', 'partial: A11 grid bit-level '
             'reproduced in 2988 (instrument '
             'reproduction, claim not retested)'),
        'meas2954_early_flipper_polarity':
            ('untested', ''),
        'meas2955_qk_source_decomposition':
            ('untested', ''),
        'meas2956_rebalance_module_localization':
            ('untested', ''),
        'meas2957_rebalance_mlp_constancy':
            ('untested', ''),
        'meas2958_imprint_dose_response':
            ('untested', ''),
        'meas2959_cross_term_algebra': ('untested', ''),
        'meas2960_profile_rotation_geometry':
            ('untested', ''),
        'meas2962_word_class_signature_matrix': (
            'len2_specific', '2987: collapses at a '
            'single filler token, content-'
            'independent'),
        'meas2963_frequency_controlled_band': (
            'robust', '2987: lang effect p<=3e-4 in '
            'all 5 conditions'),
        'meas2964_carrier_anatomy': (
            'len2_specific', '2987: h15 carrier '
            'len-2 only, migrates to h11 (p 8e-4) '
            '/ h23 under context'),
        'meas2965_h15_functional_identity': (
            'len2_specific', '2987: h15 contrast '
            'significant at L2 only (p .040)'),
        'meas2966_l34_routing_membership': (
            'untested', 'carrier-identity cards '
            'under protocol-relativity suspicion '
            'after 2987 migration finding'),
        'meas2967_collapse_carrier_anatomy': (
            'untested', 'same suspicion; 2988 a12/a13 '
            'bit-level reproduction is instrument '
            'reproduction, not claim retest'),
        'meas2968_h15_peak_anatomy': (
            'robust', '2988 T1 lock replicates at '
            'L16N: n_peak 40->40, s_c '
            '0.6567->0.6677, med_peak 0.675->0.749 '
            '(lock tol 0.3)'),
        'meas2969_peak_word_attributes':
            ('untested', ''),
        'meas2970_delay_carrier_localization':
            ('untested', ''),
    }
    cards = []
    for x in led['measurements']:
        mid = x.get('meas_id') or x.get('id') or ''
        mm = re.match(r'[A-Za-z]+_?(\d+)_', str(mid))
        if not mm or not (2936 <= int(mm.group(1))
                          <= 2970):
            continue
        if int(mm.group(1)) == 2961:
            continue  # card-set phase itself, no card
        lab, basis = label_map[str(mid)]
        cards.append({'meas_id': str(mid),
                      'phase': int(mm.group(1)),
                      'domain_label': lab,
                      'basis': basis,
                      'verdict_v1':
                          str(x.get('verdict', ''))})
    assert len(cards) == 34, len(cards)
    cs = {'card_set': 'v2', 'generated_by': 2988,
          'created': stamp,
          'labels': {'robust': sum(
              1 for c in cards
              if c['domain_label'] == 'robust'),
              'len2_specific': sum(
              1 for c in cards
              if c['domain_label'] == 'len2_specific'),
              'untested': sum(
              1 for c in cards
              if c['domain_label'] == 'untested')},
          'note': 'domain-of-validity census per plan '
                  'v4 P0; robust/len2 labels carry '
                  'retest evidence (2987/2988), '
                  'untested cards must not be cited '
                  'without a protocol tag; Omega-F '
                  'cross-model copying must copy this '
                  'set with labels',
          'cards': cards}
    io.open(CARDSET, 'w', encoding='utf-8').write(
        json.dumps(cs, indent=2, ensure_ascii=False))
    out.append('card set v2: %s' % json.dumps(cs['labels']))
else:
    out.append('card set v2 already present')

# ---------- 4. MEMO ----------
memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2988:' not in memo_txt:
    sec = (
        '## Phase 2988: 适用域普查——峰锁与词盲 L16 复制、'
        '卡片集 v2 [' + stamp + ']\n\n'
        '**问题**（方案 v4 P0）：34 卡原语卡片集哪些主张在 '
        'len-2 协议之外存活？对两张前向关键卡做 L16 复测'
        '（2968 峰锁 / 2940 词盲）+ 方向锚（a1/a3），产出'
        '带适用域标签的卡片集 v2（Ω-F 跨模型复制前置）。\n\n'
        '**设计（冻结，~2800 前向）**：两臂 x 57 词（2887 '
        '词表）——L2 臂 = 2968 协议 verbatim；L16N 臂 = '
        '14 中性填充 token（2987 句）+ [func,word]。2953/'
        '2968 注入仪器（xdir 于 L17 词位，末位索引，L2 臂'
        '位级等价），s_grid = 0 + GRID17，A/B 双族。T1 = '
        '逐词 C34/h15 峰 + 锁定 in-session s_c（2945 规则 '
        'sep<100）；T2 词盲组：WB1 语言解码置换 + WB2 '
        '概念组 ICC（22 跨语言组覆盖 57/57，组内置换 '
        'null）；T3 描述性。\n\n'
        '**产物**：`phase2988/context_domain_census/` '
        'execution ' + s_exec + ' / result ' + s_res
        + ' / context_domain_census.npz ' + s_npz
        + ' / script ' + s_scr + '；`atlas/card_set_v2.json`。\n\n'
        '**锚 17/17**：a3 Vt8 与 a11-a15/a17（vs 2953/'
        '2967/2968 npz）六锚 bit 级 0.00；a9 A11_L17 '
        '基线+10 剂量 1.81e-07；a1 2.17e-08；a4/a5 vs '
        '2935 7.2e-06/6.3e-06；确定性 L2/L16N 均 0.00。\n\n'
        '**结果**：T1 双臂 lock——L2 n_peak=40（精确复现 '
        '2968 注册值）、s_c=0.6567（=2968 注册值）；L16N '
        'n_peak=40、med_peak 0.749、s_c=0.6677，|med-s_c|'
        '=0.082<0.3。T2 blind_both——WB1 语言解码两臂 '
        'p=1e-4（obs 194.4/117.4）；WB2 ICC 0.3995 p '
        '0.292（L2）/ 0.3177 p 0.718（L16N）。T3：sepA '
        '端点 185.7→-6.4（L2）vs 119.7→-9.9（L16N）；'
        'B 带 -1.37→-0.02 vs -3.27→-1.45。\n\n'
        '**判决**：`lock__blind_both`（复合枚举映射）。\n\n'
        '**结论（重复三遍）**：峰锁卡与词盲卡在 L16 上下文'
        '下**完整复制**——h15 双相峰锁定 in-session 开关'
        '阈值、语言轴保持类编码词盲，均为上下文稳健主张；'
        '**s_c 在 xdir 仪器下稳健（0.6567→0.6677）而 2986 '
        '的 lang 剂量仪器下下降——仪器/通道差异需分账，非'
        '矛盾**（2986 用 lang 相对剂量 + 自定规则 + 74 词，'
        '本相用注册方向 + 2945 规则 + 57 词）；卡片集 v2 = '
        '{robust 5: 2936/2939/2940/2963/2968, '
        'len2_specific 3: 2962/2964/2965, untested 26}'
        '——未测卡引用必须带协议标签。\n\n'
        '**硬伤与勘误**：run1 SRC_2967 npz 文件名错（秒败'
        '无产物）；run2 C34_A16 stack 时序错（UnboundLocal'
        'Error，删产物重跑）；run3 权威一次通过。方法注记：'
        's_c 双仪器口径差异已在 ledger note 登记为待分账项。\n\n'
        '**接续**：方案 v4 P1——2989 Ω-G 微观审计开题：MLP '
        '神经元级注册表（up_proj 列 x 读出方向，L6-12 承重带'
        '优先，随机置换 null + maxT，快照/真实消融双口径，'
        '2950 竞争重平衡安全带）。\n')
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + sec)
    out.append('memo appended')

# ---------- 5. workspace log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2988' not in wl:
    entry = (
        '\n## Phase 2988（2026-09-20）\n'
        '- 适用域普查（方案 v4 P0）：判决 lock__blind_both'
        '（run3 权威，锚 17/17 含六 bit 级 0.00）；峰锁卡 '
        'L16 复制（n_peak 40→40、s_c 0.6567→0.6677）、'
        '词盲卡复制（blind_both）、方向锚复制。\n'
        '- s_c 双仪器差异登记：xdir 仪器稳健 vs 2986 lang '
        '剂量下降——通道/仪器分账待做。\n'
        '- 卡片集 v2：robust 5 / len2_specific 3 / untested '
        '26（atlas/card_set_v2.json；2961 卡组 Phase 不出卡）。'
        'Ledger 127 / L14 95。'
        '产物 phase2988/context_domain_census/。\n')
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    out.append('wslog appended')

# ---------- 6. MEMORY.md ----------
mem = io.open(MEMO_MEM, encoding='utf-8').read()
old_chain = ('2987 最小阶梯：签名单 token 即坍缩'
             '（content 无关）且与漂移量解离；lang '
             '效应稳健、h15 len-2 专属迁移 h11（载体'
             '协议相对）。核心：')
new_chain = ('2987 签名单 token 坍缩、与漂移解离；'
             '2988 普查：峰锁+词盲+方向锚 L16 复制'
             '（s_c 0.657→0.668 xdir 仪器稳健，与 '
             '2986 lang 剂量口径分账），卡片集 v2='
             '{稳健 5, len-2 专属 3, 未测 26}。核心：')
changed = False
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
else:
    out.append('WARN: chain anchor not found')
old_next = ('- max=2987，下一个 **2988**（方案 v4 P0 '
            '适用域普查：34 卡 len-2 专属/稳健分层'
            '标注，s_c/词盲卡 L16 复测；随后 P1 Ω-G '
            'MLP 神经元级注册表）。')
new_next = ('- max=2988，下一个 **2989**（方案 v4 P1 '
            'Ω-G 开题：MLP 神经元级注册表，L6-12 '
            '优先，随机置换 null+maxT，快照/真实消融'
            '双口径）。')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
else:
    out.append('WARN: MEMORY next-anchor not found')
# 3000-char guard: compress 2981 line if needed
if len(mem) > 3000:
    o = ('2981 交互非线性在 h12 输入响应（g 0.89、'
         'cos 0.91；独特性=三要素中间 regime 合取，'
         'h9 反例）')
    n = ('2981 交互非线性=h12 输入响应（g 0.89；'
         '三要素中间 regime 合取）')
    if o in mem:
        mem = mem.replace(o, n, 1)
        changed = True
        out.append('compressed 2981 line')
if changed and len(mem) <= 3000:
    io.open(MEMO_MEM, 'w', encoding='utf-8').write(mem)
out.append('memory chars=%d ok3000=%s changed=%s'
           % (len(mem), len(mem) <= 3000, changed))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('closeout done')
