# -*- coding: utf-8 -*-
"""Phase 3103 closeout (idempotent):
Ledger -> MEMO Phase 3103 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3103\omega_p101_formula_audit')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

sha8 = io.open(OUTD + r'\ledger_sha8.txt').read().strip()
assert sha8 == 'add57ba7', sha8
led_data = json.load(io.open(
    OUTD + r'\proposition_ledger.json',
    encoding='utf-8'))
assert len(led_data['propositions_review']) == 57
assert len(led_data['propositions_new']) == 5

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3103
           for m in led['measurements']):
    claim = (
        'Omega-P101 (3103, no-forward audit) - '
        'proposition-level evidence-grade ledger for '
        'the RDC research line: 62 propositions '
        '(review R01-R57 + 5 new from 3101-3103); '
        'grade distribution A=5 B=34 C=14 D=21 E=20 '
        '(A=bit-level recompute, B=supported-in-'
        'scope, C=candidate, D=downgraded, '
        'E=withdrawn).  New A-grade: PA-01 3093 '
        'FULL-V-STREAM semantics (d2b=0 bit-'
        'replicated); PA-02 submodular violations '
        '467/1792 (independent recompute); PA-03 '
        'seventh_carrier_absent; PA-05 pending '
        'review-side SwiGLU recomputation (C).  '
        'Consumer-spread check: corrections for '
        '2861 (active_amplification void), 3036 '
        '(protocol-conditioned curvature), 2759 '
        '(bias/discrimination split) have ALREADY '
        'propagated inside the MEMO; 2906 mentions '
        'are mostly filenames/rng seeds (low '
        'proposition risk); remaining spread risks: '
        '3076 depends on 3075 supermodular verdict '
        '(R44 cascade) and 3091-3092 continuum '
        'depended on 3089 strong confirmation '
        '(covered by R48 withdrawal).  Framework '
        'base adopted from review S6: residual '
        'stream equations, log-odds readout, '
        'attention 3-term decomposition, T^D/Atlas '
        'interface = research interface NOT theorem. '
        'NEXT: 3104 T2 true-relation combination '
        'dose design (3093 full-V paradigm, 4B '
        'pilot).')
    meas = {
        'meas_id': 'meas3103_omega_p101_'
                   'formula_audit',
        'phase': 3103,
        'claim': claim,
        'verdict': 'evidence_grade_A5_B34_C14_D21_E20',
        'anchors': 'deterministic audit; ledger '
                   'sha8=%s; 62 props' % sha8,
        'artifacts': {
            'ledger_json': 'phase3103/omega_p101_'
                           'formula_audit/'
                           'proposition_ledger.json',
            'consumer_grep': 'phase3103/omega_p101_'
                             'formula_audit/'
                             'consumer_grep.txt'},
        'hashes': {'ledger_sha256_8': sha8},
        'note': 'no GPU work; source of truth = '
                'review R01-R57 + MEMO consumer '
                'grep + 3101/3102 results',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3103_omega_p101_formula_audit')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3103 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3103:' not in memo:
    sec = u'''## Phase 3103: Ω-P101 RDC 命题级证据分级账本——62 命题 A5/B34/C14/D21/E20，反例传播核查与框架底座定版（formula_audit，免前向）[[NOW]]

**性质**：T1 大任务（清理理论依赖链）落地。对审查报告 R01–R57 全部主张 + 3101–3103 新命题建立**命题级账本**（审查第 7 节"后续否证未传播到总框架"硬伤的直接修复）。产物：`tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/`（proposition_ledger.json sha8=add57ba7、consumer_grep.txt）。

### 1. 等级体系与分布
A=bit 级/独立重算复现；B=限域内成立（方法正确）；C=候选观察待验；D=降级（表述/解释错误但局部材料可用）；E=撤回（反证/作废）。**62 条命题：A=5，B=34，C=14，D=21，E=20**（含复合等级按分量计）。分布解读：约 55% 命题（A+B）可作为有效依赖引用；21 条 D 级须用"可保留基础"栏的修订形式引用；20 条 E 级禁止进入新推理链。

### 2. A 级新命题（本会话新增的最高等级证据）
- **PA-01**（3101）：3093 干预语义 = 全 V 流替换——变体探针 bit 级 1e-12 + 3101 复刻 d2b=0、med_c=0.5104 一致。
- **PA-02**（3103）：3075 次模违反条件二阶差分 **467/1792>0.02**（max D=0.2625）——从 sealed A_S+MASKS 独立重算，与审查逐位一致。
- **PA-03**（3101）：**seventh_carrier_absent**——L37 swap 恢复 = 分布式再平衡而非自然载体归还（三门全败，全锚 bit-0）。
- **PA-04**（3101，B）：L37 跨条件稳定自然写入头组 {21,12,14,…} 与族特异因果焦点头解耦。
- **PA-05**（3102，C）：3100 SwiGLU 修订入口（正确一阶误差 22.5%/23.1%、二阶 5.9%/9.4%）——审查方重算，本方复现后升 B。

### 3. 反例传播核查（E/D 级命题的消费者检查）
对 8 个关键源 Phase 做后续引用 grep（consumer_grep.txt）：
- **纠错已自传播**（无需行动）：2861 active_amplification 已宣告作废（MEMO L4840）；3036 曲率断言已标注"协议条件"（L11175）；2759 已用修订后"偏置/区分可分离"结论（L435/L439）；3045 作为"V 侧无特权"被 3046–3048 正确使用。
- **低风险**：2906 的 37 处提及绝大多数是 rng seed [2906,x] 与文件名，命题级引用少。
- **剩余传播风险（登记）**：① **3076 建立在 3075 之上**——若其引用"超模结构定位"判决须连带降级（R44 级联）；② 3091–3092 连续统依赖 3089 强确认——已被 R48 撤回覆盖，3102 已登记。

### 4. RDC 框架底座定版（采用审查第 6 节，不加新定理）
可保留的计算核对关系：残差流方程 $r_\\ell=H_\\ell+A_\\ell(N_\\ell(H_\\ell)),\\ H_{\\ell+1}=r_\\ell+M_\\ell(N'_\\ell(r_\\ell))$、log-odds 读出式 $\\log(p_a/p_b)=(w_a-w_b)^\\top D_\\gamma h/s$、attention 三项分解 $O=A(X)V(X)W_O$（完整变化含 A₀ΔV、ΔA V₀、ΔAΔV）、RDC 候选接口 $\\mathcal T^D_{\\ell,\\tau}:(\\mathcal L,\\mathbf W_\\ell,\\mathcal X_\\ell)\\rightharpoonup\\mathbf W_{\\ell+1}$ 与 $\\operatorname{Atlas}_D=(G_{ext},G_{int},E^D_{assoc})$——**全部标注为"待填充研究接口/恒等式"，不是新定理**。八行框架底座表（相对关系编码/共享结构/归一化读出/条件齿轮/三图谱/连续生成/新数学）的当前地位与缺口照审查第 6 节采纳。

### 5. 结论与接续
理论依赖链清理完成：**每条强主张现在都有等级、来源与适用域**；E/D 级的传播风险收敛到两个已登记点。接续 **3104**（T2 主线）：真关系组合剂量设计——沿 2754 断边/分叉/同端点异路径范式 + 3093 全 V 流干预，4B 先导实验；验收=冻结提取器在未见组合上超越端点/词袋/位置/加性基线。

产物 sha8：proposition_ledger.json=add57ba7（62 命题全量字段含 grade/retain/consumer_check）。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3103)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3103 Omega-P101 (no-forward): '
          'RDC proposition evidence-grade ledger: 62 '
          'props (R01-R57 + PA-01..05), grades '
          'A5/B34/C14/D21/E20; consumer-spread: '
          '2861/3036/2759 corrections already '
          'propagated, 2906 low-risk, remaining '
          'risks = 3076->3075 supermodular cascade '
          '+ 3089->3091 (covered by R48); framework '
          'base adopted (equations = interface not '
          'theorem); ledger sha8=add57ba7; NEXT '
          '3104 T2 relation-combination dose.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3103 Omega-P101' not in prev:
        try:
            with io.open(wl, 'a', encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3103' not in mem_old:
    mem_new = mem_old.replace(
        '## 下一步',
        '## 机制链状态（3103）\n'
        '- 命题账本 62 条 A5/B34/C14/D21/E20（sha8 '
        'add57ba7）；E 级禁止入新推理链，D 级用修订'
        '形式引用；传播风险：3076→3075 超模级联。\n'
        '- 3093=全 V 流替换（bit 复刻）；3101 '
        'seventh_carrier_absent；L37 自然写入头组 '
        '21/12/14。\n'
        '\n## 下一步')
    mem_new = mem_new.replace(
        'max=3102', 'max=3103').replace(
        '下一 3103：**RDC 公式证据分级审计**（免前向，T1）→',
        '下一 3104：**T2 真关系组合剂量**（3093 全 V 流范式，4B 先导）→')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
