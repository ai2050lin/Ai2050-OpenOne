"""Build a bounded next whole campaign only from completed, separately scoped evidence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2676_native_mlp_delivery'


def main():
    checked=read(OUT/'analysis/scientific_checks.json');assert checked['all_checks_passed'];s=checked['summary']
    normal={}
    for site,cells in s['frozen_candidate_scope'].items():
        cells=[v for k,v in cells.items() if k.endswith('/q0/m0')]
        normal[site]={k:sum(v[k] for v in cells) for k in ('n','same_old_direction','target_dominates_form_order','both')}
    interpretation={
        'observed_normal_protocol_candidates':normal,
        'strict_original_survivors':s['survivors'],
        'separate_evidence_levels':[
            '7个旧候选在新实体与独立内容交叉中的方向复验、目标相对句式/顺序的幅值复验，分别计数。严格门为空不抹去方向阳性，也不允许把方向阳性冒称严格门通过。',
            '真实 gate/up/down 权重与激活可逐坐标查询；全token链式偏导、原生乘加/乘积恒等式是已知计算关系，不是新发现的语言数学结构。',
            '真实内存标量改动、逐前缀整矩阵恢复哈希构成实施和可重复性检查；数值误差大小另账。all_checks_passed表示计划与校验完成，不表示语义假说通过。',
            'FP64仅提升读出矩阵乘加和log-softmax；Transformer核心仍FP32。剂量扩大同时改变了非线性程度，不能仅凭误差下降把全部误差归因于一种舍入来源。',
            '4096时序扩展是依据幅值例外后验选族，但实体/动词全交叉方案在新增forward前冻结。它是针对性复核，不是新增八族泛化证明。',
            'Qwen14B冻结候选使用事实方向模式门，不使用4B的旧方向加幅值门；两个比例不可作直接模型优劣排名。GLM4/DS7B保留本模型全部坐标，禁止同下标对应语义。',
            '完整格式化候选序列的概率不是自主生成成功。反问/反码失败、语言差异以及DS7B的16token预算限制均保留。',
            '大量Phase未闭合说明现有证据不足，不能逻辑上证明范式必然致命；下一步更换可检验问题，而不是仅换任务材料重跑必要性门。'],
        'expanded_chronology':s['expanded_chronology']['normal_protocol_candidates'],
        'scalar_numerics':s['scalar_numerics'],
        'scalar_numerics_corrected_ratios':s['scalar_numerics_corrected_ratios'],
        'readout_resolution':s['numeric_resolution']['summary'],
        'source_prefix_audit':s['source_prefix_audit'],'prefix_replays':s['prefix_replays'],'operation_length_audit':s['operation_length_audit'],
        'q14_own_gate':s['q14_frozen'],
        'next_question':'保持同一组原生候选与全坐标背景，逐个source token追问：哪个真实Attention输出项经RMSNorm进入gate/up的哪个输入坐标，如何在外部语言操作改变时重分配、相消、跨层复用，并连接到实际MLP输出。',
        'mechanism_closed':False,
    }
    save(OUT/'analysis/interpretation.json',interpretation)
    plan=[
        {'phase':2677,'task':'复审2670–2676实测与精度范围，特别是相同正文前缀在总长度变化下出现的BF16差异及640次重放对照；未澄清的长度/数值因素不得命名为语义。冻结source-token→原生坐标→MLP输入贡献的大阶段。旧2H/5MLP观察窗全部保留，不按严格幅值门阴性删除；全坐标背景和新增14B实际通过集合分别冻结。对事实、问题、编码规则、示例与格式token作可审核、互斥字符区间标注；跨区token单列mixed，不冒称天然语义分区。'},
        {'phase':2678,'task':'Qwen4B非量化至少8192多族双语条件观察：复用已确认的8语言族，实体、内容、形式、顺序、事实目标、所问实体、问题极性与回答映射分开；加入至少512同正文不同输出功能的配对对照，覆盖直接人名回答、受控填空、普通是非和映射是非（可复用2648的已审计输出功能设计），不只换一套Yes/No提示词。完整自然输出与失败保留。所有H/MLP坐标、正文和任务边界全部背景，所有token原坐标逐坐标流式矩；代表性原轨迹展示，无TopK。新实体/任务变体先冻结后forward。'},
        {'phase':2679,'task':'在至少512平衡heldout前缀上记录四既定中层的实际Q/K/V、实际attention概率、各head全部value维度与全部输出坐标，并均衡覆盖上述四种输出功能。利用真实W_o计算每个source token对两个query边界、每个残差坐标的乘加项；比对真实attention输出，舍入残差单列。V_s本身已聚合此前上下文，不能把source项当成某个孤立原词的独占因果贡献。不得使用donor或将attention权重本身当作语义解释。全heads/source/坐标覆盖，内存分块仅为精确计算。'},
        {'phase':2680,'task':'把source贡献接入真实RMSNorm分母与gamma、五固定单元全部gate/up输入权重和全部down输出权重；对全部输入k的正负相消、条件变化、跨层重用和分化逐坐标记账。冻结当前实际normalizer的分解只作为条件账本，不解释为删除某个source后的反事实。完整支路和低值背景保持；标准加法乘法恒等式不报作新机制。'},
        {'phase':2681,'task':'重要重复模式用至少4096额外新实体/内容条件扩大复核，保留原32或64分组而非事后只选成功族。方向、幅值、source角色重分配和自然行为分别报告；同query token换正文、同正文换query/映射/示例必须成对。不能因某个删除/救援门失败中断观察；不能把新的宽松指标称为旧门通过。'},
        {'phase':2682,'task':'只对已完成数值分辨率检查的局部量做参数算法复验：至少128前缀，实际gate/up/down普通与低值标量，双方向/多剂量与完整token目标。根据2676精度结果先冻结误差判读；若核心FP32仍不能分辨，限制结论并改用可分辨的原生中间量，而非重复宣称终端1e-5概率闭合。逐项恢复并哈希，GPU串行。'},
        {'phase':2683,'task':'顺序Qwen14B、GLM4、DS7B各至少512同族条件，用本模型QKV/MLP架构与词元区间重复source→坐标账本。BF16非量化auto加载；每次释放一个模型后再加载下一个。推理模型预算与停止符先做小规模协议审查，再冻结正式预算；不把16token无答案解释成没有语言机制。'},
        {'phase':2684,'task':'整合八族与扩大复核，发布source token×完整残差坐标、真实gate/up输入项、RMSNorm及全MLP背景热力图；可查询原参数、激活、坐标编号、数值余项与实际文本。真实API/浏览器/构建及逐参数QA后，按精确白名单清理未展示原场。所有Phase连续append，判断下一整批是否同目标并继续。'},
    ]
    result={'frontier':2676,'same_goal':True,'plan':plan,'priority':'语言操作全坐标观察 → 原生source与参数计算结构 → 新材料确认 → 次要数值/因果检验。不是重新把闭合当唯一通行证。',
        'frozen_q4_h':[[24,2355],[27,1217]],'frozen_q4_mlp':[[23,6197],[26,3594],[27,3221],[28,5952],[28,8513]],
        'q4_observation_window_not_strict_survivors':True,'q14_observed_own_gate':s['q14_frozen'],
        'evidence_sha256':sha(OUT/'analysis/scientific_checks.json'),'interpretation_sha256':sha(OUT/'analysis/interpretation.json'),
        'prospective_scope':'This is the NEXT complete campaign, not completed work. Phase2677 must freeze detailed materials, memory/disk bounds and instrumentation no-op checks before launching2678. Do not duplicate a running GPU process or overwrite oldphases.',
        'equation_boundary':r'x_{t,k}=gamma_k/r_t*(h_{t,k}+sum_s c_{t,s,k})+epsilon_{t,k}; g_{t,j}=sum_k Wg_{j,k}x_{t,k}+epsilon_g. Observed r_t is endogenous; the displayed additive allocation is conditional bookkeeping, not counterfactual causal credit.'}
    save(OUT/'analysis/next_campaign.json',result);print('INTERPRETATION AND NEXT WHOLE CAMPAIGN SAVED')


if __name__=='__main__':main()
