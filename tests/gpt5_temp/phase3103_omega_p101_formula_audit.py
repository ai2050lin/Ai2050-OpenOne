# -*- coding: utf-8 -*-
"""Phase 3103 Omega-P101: RDC proposition evidence-grade audit.

No forwards. Builds a proposition ledger from the review's
57 claims (R01-R57) + new A-grade propositions from 3101/3102,
maps review verdicts to evidence grades, runs consumer-spread
checks for E/D propositions, and writes:
  proposition_ledger.json / audit_summary.md
under phase3103/omega_p101_formula_audit/.
Grade key: A=bit-level recompute; B=supported-in-scope;
C=candidate observation; D=downgraded (wording/interpretation);
E=withdrawn/refuted.
"""
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3103\omega_p101_formula_audit')
os.makedirs(OUTD, exist_ok=True)

# (id, phases, claim, review_verdict, grade, retain)
# grade uses compound codes like 'B+D'.
R = [
    ('R01', '2750,2756-2760',
     '训练世界差异揭示低层全盲、深层才接入历史',
     '限域观察；强解释错误', 'B+D',
     '配对训练世界响应差异、同前缀比较'),
    ('R02', '2751-2755,2762',
     '监督方向参数子集签名及跨模型复现',
     '限域保留', 'B',
     '单块更新稳定差异、子集替换效应'),
    ('R03', '2757-2759',
     '问句事实泄露是错误根源',
     '原假说已被反证', 'E',
     'span删除与世界区分对照'),
    ('R04', '2761,2764,2778',
     '内部知道但说不出，oracle修复证明知识存在',
     '候选待验；强解释不成立', 'C+D',
     '中间层目标可读性、可操控读出'),
    ('R05', '2763-2779,2785',
     '偏置修复、pull符号规则与否定深未知',
     '修复限域保留；深未知降级', 'B+D',
     '特定错误子集修复、方向对照'),
    ('R06', '2780-2787',
     '修复机制自然迁移与共享e0轴',
     '自然迁移失败是有效边界', 'B+D',
     '受控修复及自然材料反例(0/35)'),
    ('R07', '2788-2796',
     '词位驻留、随身档案、前置调制',
     '限域保留；自足性待验', 'B+C',
     '词位/查询位不同响应、前置调制'),
    ('R08', '2797-2801',
     '单坐标角色与一次性换车定律',
     '候选待验', 'C',
     '原生坐标贡献表、层间响应变化'),
    ('R09', '2802-2804',
     '多义对比负相关、平坦谱为普适语义法则',
     '构造部分不得作发现', 'D',
     '特定词义锚对比读出(两义反向=恒等式)'),
    ('R10', '2804-2805',
     '实体感紧凑性定律',
     '普适形式已被否证', 'E+B',
     '词内特定操作化下的几何'),
    ('R11', '2806-2809,2811',
     '层级嵌套、十维骨架、未见词分类',
     '分类限域保留；嵌套证据撤回', 'B+E',
     '99新词分类成绩与低维类别线索'),
    ('R12', '2810-2815',
     '残差是属性流形/微场/GPS；静态嵌入无知识坐标',
     '几种具体假说失败；全称否定错误', 'E+D',
     '各候选被排除的操作化范围'),
    ('R13', '2816-2818',
     'OV无语义或对齐列就是功能神经元',
     '原强命题被修正', 'D',
     '容量/实际写入/删除效应分开的三级证据'),
    ('R14', '2819-2827,2829,2831-2832',
     '属性编辑与实体条件化写头',
     '局部编辑基础可用；完整机制待验', 'B+C',
     '属性方向、实体依赖及跨实体副作用'),
    ('R15', '2828,2830,2833-2835',
     '知识链不在运行时传播',
     '原全称命题不成立', 'E',
     '所测方向/位置/范式的阴性结果'),
    ('R16', '2836-2852',
     '双通道、attention骨干、门与内容分工',
     '限域保留', 'B',
     '源位置贡献与模块干预、末层norm边界纠错'),
    ('R17', '2853-2861',
     '深层MLP主动放大及增益窗口',
     '部分原增益为明确伪影', 'D+E',
     '真实pre-hook工作点clamp比值'),
    ('R18', '2856-2859,2862-2869',
     '机制词坐标、检索、增长率与密度门控',
     '候选待验', 'C',
     '词级响应特征、有限检索收益'),
    ('R19', '2870-2889',
     '类别/属性/语法/语言共享或分离通道',
     '限域响应图谱', 'B',
     '多任务轴同协议响应差异'),
    ('R20', '2890-2903',
     'GLM语言轴缺失、逐层重提取与读出容忍',
     '后续修正有价值', 'B',
     '方向随层变化、探针失配假阴性'),
    ('R21', '2904-2912',
     'margin检测器与幅值定律',
     '测量诊断保留；充分性过强', 'B+D',
     '检测器对矩/符号/类均值的敏感性'),
    ('R22', '2906',
     '伪逆定位0.7%神经元承载20%语言能量',
     '原功能解释错误', 'E+B',
     'down列字典最小范数系数集中度'),
    ('R23', '2913-2934',
     '头x层事件图谱、探针不变骨架',
     '限域保留，需采用纠错版', 'B+D',
     '多重比较、null、功能对照和事件索引'),
    ('R24', '2935-2942',
     '语义上下文压制、旋转和近正交载体',
     '候选待验', 'C',
     '对选定读出轴的方向变化与干预'),
    ('R25', '2943-2960',
     '剂量开关、头竞争重平衡与交叉项',
     '限域计算结构', 'B',
     '干预剂量曲线、直接/间接项与交叉项'),
    ('R26', '2961-2973',
     '卡片链认证、词类签名、频率控制',
     '局部结果保留；强认证降级', 'B+D',
     '可查询卡片及候选差异'),
    ('R27', '2974-2985',
     '子空间握手、h12双轴交互及正交重定向',
     '限域保留', 'B',
     '条件非加性、双轴干预和路径定位'),
    ('R28', '2986-2988',
     '长上下文推广与签名适用域',
     '反例应作为基础', 'D+B',
     '加一个上下文token即可破坏部分类签名'),
    ('R29', '2989-2992',
     '神经元注册表、集中单点与稀疏字典',
     '真实采集保留；专属因果性未定', 'B+C',
     '实际MLP单元、集合删除、单点冗余反证'),
    ('R30', '2993-3006',
     '逻辑签名、跨模型洗消/放大、预训练来源',
     '限域保留', 'B',
     '模型/层差异、base中可见某些现象'),
    ('R31', '3007-3010,3038-3039',
     '自回归签名、再入读出与自然生成',
     '协议需纠正', 'D',
     '生成态记录及token/分布指标差异'),
    ('R32', '3011-3017',
     'L3逻辑位KV门控与下游放大',
     '局部干预候选', 'C',
     '特定位置K/V、消费头、下游路径'),
    ('R33', '3018-3020,3033,3035',
     '主动免疫、阻尼场统一衰减27倍',
     '扰动响应限域保留；免疫解释未证', 'B+D',
     '负叉积、抵消/稀释分解、概率效应比'),
    ('R34', '3021-3032,3034',
     '中继联盟、保护带、招募与深峰',
     '限域保留', 'B',
     '零消融与基线恢复差异、集合干预'),
    ('R35', '3035-3036,3039',
     '指纹竞争、普遍logistic曲率律',
     'log-odds基础可用；曲率普适性撤回', 'B+E',
     '同前缀token差方向及完整softmax竞争'),
    ('R36', '3037,3040-3041',
     'KV写入98.3%词身份、秩1情境轴',
     '原比例解释和普遍秩1不成立', 'D+E',
     'L3 kv7样本内词均值空间高投影能量'),
    ('R37', '3042-3045',
     '风格全局引力场及L20因果反转',
     '相关轴候选；L20旧反转撤回', 'C+E',
     '前缀共享响应及语句差异'),
    ('R38', '3046-3053',
     'K场、全位置KV复放、末层门槽',
     '限域保留', 'B',
     '干预点、qk-norm校验与位置范围对照'),
    ('R39', '3054-3058',
     'gamma白化、读出预对齐、64通道载荷',
     '重加权保留；白化说法已反证', 'B+E',
     'RMSNorm几何、特定通道替换效果'),
    ('R40', '3059-3065',
     '写读异轴、高维body与跨模型拓扑',
     '候选待验', 'C',
     '所测方向族、写读差异、模型符号差别'),
    ('R41', '3066-3071',
     '条件MLP反转、共享神经元池与头压制',
     '局部机制候选可保留', 'C',
     '同池复用、正负贡献竞争、同块模块作用'),
    ('R42', '3072',
     'Attention头是线性读取器',
     '固定路由V路径恒等式可用；整体线性错误',
     'B+E',
     '固定Q/K时对V的线性恒等式'),
    ('R43', '3073-3074,3076',
     'Hill容量定律与头预算守恒',
     '局部拟合保留；普遍守恒未建立', 'B+D',
     '选定8头集合非加性、饱和响应'),
    ('R44', '3075',
     '次模等价于高阶Mobius全非正、大基座补全定律',
     '数学等价式错误；有限违反仍在', 'E+A',
     '条件二阶差分467/1792>0.02(3103独立重算逐位一致)'),
    ('R45', '3076-3077',
     '跨族固定焦点头及可观察路由',
     '固定头普遍性不支持', 'E+B',
     '头x族交互、写入漂移观察'),
    ('R46', '3078',
     '上游状态不含路由信息',
     '原排除推理错误', 'E',
     '所测末位线性特征预测失败(阴性结果)'),
    ('R47', '3079-3081,3094-3095',
     '方向夹角锁定迁移、角度共振',
     '条件相关候选；普适性反证', 'C+D',
     '最终logit差方向f2与迁移部分相关'),
    ('R48', '3082-3089,3091-3092',
     '谱-迁移连续统跨架构显著确认',
     '强统计确认撤回；候选相关保留', 'E+C',
     '模型与层位之间的描述差别'),
    ('R49', '3090,3093',
     '复用底册与低PR必然迁移',
     '底册有用；充分性被反例限制', 'B+D',
     '稳定ID/数据追溯，14B trunk'),
    ('R50', '3094-3098',
     '末块MLP条件重写器',
     '限域保留', 'B',
     '末位、所测前缀族中的MLP子步变化与干预'),
    ('R51', '3099',
     '方向编码而非集合编码、无专职组',
     '排他性结论不成立', 'E+B',
     '激活增量方向与集中度描述'),
    ('R52', '3099',
     'head(norm(delta-m))证明一阶直通桥',
     '机制解释错误/待修', 'D',
     '一个定义明确的描述性相似度'),
    ('R53', '3100',
     'SwiGLU一阶公式与6%高阶残差',
     '原公式和误差解释错误；另有修订', 'E+B',
     '修订入口:正确一阶误差22.5%/23.1%,二阶5.9%/9.4%'),
    ('R54', '3100',
     '96.5%沿残差通道、RMSNorm透明',
     '原精确比例和透明性推论错误', 'E+B',
     '可计算的局部通道分解(含交叉项)'),
    ('R55', '3100',
     'L37自然载体与因果焦点重叠1/0/0证明解耦',
     '原比较作废；修订仅描述性', 'E+A',
     '3101 pre-o_proj重测2/3/2+seventh_carrier_absent'),
    ('R56', '2750-3100',
     'Attention只搬运、MLP独占知识、单坐标=概念',
     '尚未成立的绝对分工', 'D',
     '跨位置聚合/位置内更新、共享参数条件化'),
    ('R57', '2750-3100',
     '三图谱已完备、RDC完整闭合、无限组合已破解',
     '尚未完成', 'D',
     '三图谱接口、证据账本、条件化复用纲领'),
]

# New A-grade propositions from 3101/3102/3103
PA = [
    ('PA-01', '3101',
     '3093干预语义=全V流替换(所有层v_proj钳回自然bank,仅L37[0,4)携带prefix V)',
     'A', '变体探针bit级1e-12;3101复刻d2b=0,med_c=0.5104一致'),
    ('PA-02', '3103',
     '3075次模违反:条件二阶差分467/1792>0.02(max D=0.2625)',
     'A', '3103从sealed A_S+MASKS独立重算,与审查逐位一致'),
    ('PA-03', '3101',
     'seventh_carrier_absent:L37 swap恢复=分布式再平衡,非自然载体归还',
     'A', '三门全败(2/3/2;0.80/0.91/0.80;0.297/0.325),全锚bit-0'),
    ('PA-04', '3101',
     'L37跨条件稳定自然写入头组{21,12,14,...}与族特异因果焦点头解耦',
     'B', 'top8_nat_pre三族前3共享(21/12/14);3093焦点top8族特异'),
    ('PA-05', '3102',
     '3100 SwiGLU正确一阶误差22.5%/23.1%,二阶5.9%/9.4%(修订入口)',
     'C', '审查方GLM2751重算;本方待复现后升B'),
]

# Consumer-spread check: for these source phases, find later
# references in MEMO (heuristic: bare number mentions after the
# phase's own section).
CHECK = ['2861', '2906', '3036', '3045', '3075',
         '3089', '2759', '2861']
memo = io.open(MEMO, encoding='utf-8',
               errors='ignore').read()
lines = memo.splitlines()


def phase_start(pno):
    pat = '## Phase %s:' % pno
    for i, ln in enumerate(lines):
        if ln.startswith(pat):
            return i
    return -1


consumer = {}
for pno in sorted(set(CHECK)):
    st = phase_start(pno)
    hits = []
    if st >= 0:
        pat = re.compile(r'(?<!\d)%s(?!\d)' % pno)
        for i in range(st + 1, len(lines)):
            for m in pat.finditer(lines[i]):
                s = max(0, m.start() - 60)
                e = min(len(lines[i]), m.end() + 60)
                hits.append((i + 1,
                             lines[i][s:e].strip()[:130]))
    consumer[pno] = hits

# grade distribution
from collections import Counter
cnt = Counter()
for rid, ph, cl, vd, gr, rt in R:
    for g in gr.split('+'):
        cnt[g] += 1
for pid, ph, cl, gr, ev in PA:
    cnt[gr] += 1

lines_out = []
lines_out.append('=== grade distribution ===')
for g in 'ABCDE':
    lines_out.append('%s: %d' % (g, cnt.get(g, 0)))
lines_out.append('total props: %d (R=%d, PA=%d)'
                 % (len(R) + len(PA), len(R), len(PA)))

lines_out.append('')
lines_out.append('=== consumer-spread check ===')
for pno, hits in sorted(consumer.items()):
    lines_out.append('Phase %s: %d later mentions'
                     % (pno, len(hits)))
    for ln_no, ctx in hits[:4]:
        lines_out.append('  L%d: %s' % (ln_no, ctx))
    if len(hits) > 4:
        lines_out.append('  ... (%d more)'
                         % (len(hits) - 4))

with io.open(OUTD + r'\consumer_grep.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

ledger = {
    'phase': 3103,
    'name': 'omega_p101_formula_audit',
    'grade_key': ('A=bit-level recompute; '
                  'B=supported-in-scope; C=candidate; '
                  'D=downgraded; E=withdrawn'),
    'grade_distribution': dict(cnt),
    'propositions_review': [
        {'id': rid, 'phases': ph, 'claim': cl,
         'review_verdict': vd, 'grade': gr,
         'retain': rt}
        for rid, ph, cl, vd, gr, rt in R],
    'propositions_new': [
        {'id': pid, 'phases': ph, 'claim': cl,
         'grade': gr, 'evidence': ev}
        for pid, ph, cl, gr, ev in PA],
    'consumer_check': {
        p: [{'line': l, 'ctx': c} for l, c in h]
        for p, h in consumer.items()},
}
with io.open(OUTD + r'\proposition_ledger.json',
             'w', encoding='utf-8') as f:
    json.dump(ledger, f, ensure_ascii=False, indent=1)

import hashlib
sha8 = hashlib.sha256(
    json.dumps(ledger, sort_keys=True,
               ensure_ascii=False).encode('utf-8')
).hexdigest()[:8]
with io.open(OUTD + r'\ledger_sha8.txt', 'w') as f:
    f.write(sha8 + '\n')
print('OK props=%d sha8=%s' % (len(R) + len(PA), sha8))
