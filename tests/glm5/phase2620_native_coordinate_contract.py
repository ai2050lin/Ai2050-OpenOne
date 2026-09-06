"""Audited, finite native-coordinate campaign. No donor-state transplantation."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / 'tests/glm5'
RESULT = TESTS / 'result'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
CAMPAIGN = RESULT / 'phase2620_native_coordinate_contract'

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def finish(phase, title, out, result, principle, formula, cases, analysis, limits, conclusion):
    out=Path(out); result.update(phase=phase, timestamp=datetime.now().astimezone().isoformat(), language_mechanism_closed=False)
    result['checks']={k:bool(v) for k,v in result.get('checks',{}).items()}
    result['all_checks_passed']=all(result.get('checks',{}).values())
    save(out/'analysis/final.json', result)
    previous=MEMO.read_text(encoding='utf-8-sig')
    phases=[int(x) for x in re.findall(r'^## Phase (\d+):',previous,re.M)]
    if phase in phases: raise RuntimeError(f'Phase {phase} already recorded; do not overwrite scientific history')
    if phases[-1] != phase-1: raise RuntimeError(f'Non-contiguous append: {phases[-1]} -> {phase}')
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    summary=result.get('summary',result.get('checks'))
    text=f'\n\n## Phase {phase}: {title} [{stamp}]\n\n**测试原理。** {principle}\n\n**测试用例与范围。** {cases}\n\n$$\n{formula}\n$$\n\n**结果汇总。** {json.dumps(summary,ensure_ascii=False)}。程序检查={json.dumps(result.get("checks",{}),ensure_ascii=False)}。\n\n**相关文件。** `{out}` 的 protocol/material/field/analysis 等实际产物及 `analysis/final.json`；测试代码在 `{TESTS}`，以具体结果中的 provenance 为准。\n\n**分析与理论进展。** {analysis}\n\n**问题硬伤。** {limits}\n\n**结论与下一任务。** {conclusion}\n'
    with MEMO.open('a',encoding='utf-8') as f: f.write(text)
    print(f'PHASE {phase} FINISHED '+json.dumps({'all_checks_passed':result['all_checks_passed'],'final':str(out/'analysis/final.json')},ensure_ascii=True),flush=True)

def main():
    sources=[Path('C:/Users/Admin/.codex/attachments')/x/'pasted-text.txt' for x in ('f5ec6cfc-b09c-428d-b3e8-130115b6e077','991925b6-0c65-43c5-a1a1-834251d303d6','82dff158-f222-4243-ad83-333ee94702e8')]
    f19=read(RESULT/'phase2619_fresh_bidirectional_operator_confirmation/analysis/final.json')
    corrections=[
        '附件二的2615仅时序，不是六族；48.3%是另一侧答案率，不是基线条件翻转率。',
        '2608为12条件×120=1440，不是4条件；2619为7条件×240=1680，不是6条件。',
        '并非六族行为全过门；旧重排只是标签排序，不代表长句内容保持。',
        '2619更强的新词表确认必须作为最新证据；240pair含80基础事件对的3表面复用。',
        '英语93.33%、中文54.17%及双侧正确筛选必须分账；不能声称跨语言通用。',
        '低幅值能量不是必要性；方向有效既不排除单坐标贡献，也不确认独立齿轮。',
        '2617族均值与字典范数不同，不能单由两者率差证明条件粒度效应。',
        '多层oracle累加充分性不证明一次自然跨层传递；两条实验链不能拼成闭环。',
        'activation coordinate、MLP intermediate neuron、learned scalar weight是三个不同对象。',
        '梯度是局部导数；有限扰动需验证；group synergy/LOO不等于自然必要性。',
        '最终logit坐标求和是代数记账，不是语义解释；norm/bias/首token与完整答案必须明确。',
        '需要或不需要新数学均未获证明；任意可逆基底变换也不一般保持带逐坐标非线性的架构形式。',
        '按激活Top-K挑稀疏核心、lasso及高阶数学大擂台不列主线，依用户新重点先基础全坐标与原生计算。',
    ]
    plan={2620:'附件逐项复审与原生坐标—神经元—参数合同',2621:'八族双语交叉材料、真实greedy及完整句子重排核验',2622:'未干预全token全层原坐标与全部MLP边界神经元采集',2623:'最终层全坐标/全神经元/逐参数解析提取和跨条件图谱',2624:'非搬运的单神经元、单坐标、单参数真实BF16前向校验',2625:'Qwen14B非量化原生计算复验',2626:'GLM4非量化原生计算复验',2627:'DS7B非量化原生计算复验',2628:'八族全坐标复用与条件化规律交叉审查、客户端交付',2629:'同目标自动续研：冻结算法的新材料扩大确认与终审'}
    contract={'plan':plan,'models':['qwen3-4b','Qwen3-14B','glm4-9b-chat-hf','deepseek-r1-distill-qwen-7b'],
        'initial_material':'8 families x 2 languages x 12 base items x 2 forms x 2 variants = 768 prompts; base item repeats across forms/variants',
        'split':'base indices 0..5 discovery; 6..11 heldout; not all vocab disjoint; sense anchor may necessarily recur',
        'native_algorithm':'actual last-block SwiGLU units, W_down, RMSNorm and output-head coordinate equations; all units, no donor and no top-k',
        'coverage':'all-layer raw hidden states and MLP boundary units on Qwen4; exact downstream scalar formulas cover final MLP block only, not whole model weights',
        'semantics':'task-first-token contrast is externally specified; native own top1-top2 output is separate; neither amounts to a learned semantic extractor',
        'validation':'actual BF16 forward vs FP32 real-arithmetic accounting, rounding errors and no-op controls; no persistent weight writes',
        'crossmodel':'sequential 32 stratified cases/model, engineering replication not population-level semantic confirmation',
        'expansion':'new index range 12..35, one form, both variants, all eight families/languages = 768 fresh-context cases for frozen final-block algorithm',
        'storage':'FP32 containers; full fields streamed, no coordinate pruning; keep published exemplars and all-coordinate summaries; remove unshown raw fields only after verified manifest',
        'excluded_as_unsupported_core':['sparse/top-k core as prior','statistical or advanced-math tournament','automatic gear naming','closure as sole stop gate'],
        'attachment_sha256':{str(p):sha(p) for p in sources},'corrections':corrections}
    save(CAMPAIGN/'protocol/frozen.json',contract)
    result={'provenance':str(Path(__file__)), 'summary':{'audit_corrections':len(corrections),'planned_phases':list(plan),'prior_best_all240_learned':f19['conditions']['learned']['donor_correct']},'checks':{'all3_attachments_hashed':len(sources)==3,'prior_2619_complete':f19['all_checks_passed']}}
    finish(2620,'从差分搬运转向原生计算的证据复审与完整大阶段合同',CAMPAIGN,result,
        '逐一对照原始final与MEMO，保留英语新事件时序阳性；把可视化响应、真实计算关系、局部输出效应分成三套账。',
        r'h_j\ne a_k\ne\theta_{jk};\quad a_k=\operatorname{SiLU}(g_k)u_k,\quad m_j=\sum_kW_{jk}a_k;\quad c_j=(U_{y,j}-U_{z,j})N(h)_j.',
        '三份最新附件、Phase2600—2619历史；冻结2620—2629十个相关大Phase，具体内容见protocol/frozen.json。',
        '原生权重是已知计算规则，可先回答每个真实坐标如何参与计算，再检验哪些条件响应跨材料复用。固定最后MLP的解析式提供无需donor的单参数可计算基线，不宣称它就是语言机制。',
        '附件宏大理论分解不唯一；未知机制不能由既有实验失败推出。最终层只覆盖输出编译的一段，早中层路由仍需以后按证据扩大。八族模板不是所有语言；重要显著阳性需新材料复核。',
        '执行全部冻结模块及同目标扩大确认；后续调度沿原生坐标、神经元与参数算法继续，而不是重复整体搬运。')

if __name__=='__main__':main()
