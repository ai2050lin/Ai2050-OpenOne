"""Audit actual earlier evidence and freeze the integrated relation-update agenda."""
from collections import Counter
from rdc_update_common import *

ATTACHMENTS=[
 Path('C:/Users/Admin/.codex/attachments/af1fcad7-4fff-4b4a-bfa1-80efe32886e7/pasted-text.txt'),
 Path('C:/Users/Admin/.codex/attachments/fd51ff93-a505-4ca2-96fb-73f4f192111b/pasted-text.txt')]

def main():
    if (BASE/'contract.json').exists():print('UPDATE_CONTRACT_EXISTS');return
    start=time.monotonic();assert read(PRIOR/'verification/final.json')['all_passed']
    immutable(BASE/'resources.json',{'timestamp':stamp(),'result_ceiling_bytes':12*1024**3,
      'disk_floor_bytes':12*1024**3,'compute_ceiling_seconds':21600,'per_process_ceiling_seconds':7200,
      'basis':'Reuse prior finite-run limits; initial pilot gates expansion. Includes one information-bearing automatic follow-up, not an unbounded AGI search.',
      'ledger':'Measured nonnested script durations where possible, failures retained. Implementation and browser time are not GPU compute.'})
    rows=gzread(PRIOR/'natural_discovery.json.gz');counts=Counter(r['split'] for r in rows)
    assert counts=={'train':320,'validation':64,'test':128}
    validation=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    with np.load(PRIOR/'signed_source/banks/b16_original.npz') as z:
        assert len(z['coefficients'])==320
    with np.load(PRIOR/'format_content/all_parameter_gram_decomposition.npz') as z:
        c=z['content'];f=z['format'];cross=z['content_format_cross'];g=z['full']
        decomposition=float(np.max(abs(c+f+cross+cross.T-g))/np.max(abs(g)))
        nonorthogonal=float(np.max(abs(np.diag(cross))))
    assert decomposition<1e-6 and nonorthogonal>0
    bridge=[];panel=gzread(PRIOR/'middle_training/material.json.gz');panel=[r for r in panel if r['split']!='train']
    meta=[r for r in panel for _ in r['positions']]
    with np.load(PRIOR/'middle_training/bridge_baseline.npz') as z:b=z['loss'].copy()
    with np.load(PRIOR/'middle_training/original_native.npz') as z:n=z['loss'].copy()
    for split,cohort in sorted({(r['split'],r['cohort']) for r in meta}):
        ix=[i for i,r in enumerate(meta) if (r['split'],r['cohort'])==(split,cohort)]
        rec={'split':split,'cohort':cohort,'bridge_minus_native':float((b[ix]-n[ix]).mean()),'training_vs_bridge':[]}
        for seed in (2733,2734):
          for condition in ('coherent','order_control'):
            with np.load(PRIOR/f'middle_training/{condition}_{seed}/checkpoint32.npz') as z:delta=z['loss'][ix]-b[ix]
            rec['training_vs_bridge'].append({'seed':seed,'condition':condition,'delta':float(delta.mean()),'cluster':clustered(delta,[meta[i]['source_group'] for i in ix])})
        bridge.append(rec)
    audit=[
      ('A01','retain','原生MLP条件乘积、全三矩阵梯度和真实中层续训存在；仅限声明的模型、层和目标。'),
      ('A02','correct','validation96为说明笔误；实际索引为train320/validation64/test128，旧材料与拟合结果保持不变。'),
      ('A03','resolve','将已有FP32/BF16桥接基线纳入本次独立数值重算，分别报告桥接与训练作用。'),
      ('A04','retain_limit','全坐标不等于保留范数、身份、来源方向和历史充分性；新预测器同时保留原幅值与规范化对照。'),
      ('B01','reject','内容与格式梯度精确相加但并不正交；正交约束是新增优化选择，不是纯语义的证书。'),
      ('B02','reject','乱序也具有非零、可投影的梯度结构；不能称噪声或宣布只有连贯文本能形成低秩语义流形。'),
      ('B03','narrow','条件门乘积是已有SwiGLU架构恒等式；C的反对称部分不参与二次型，gate因子另有作用，不能只用C余弦定义功能。'),
      ('B04','reject','32token的特定来源预测分支偏移，不是所有微调长期崩溃的证明；KV致因和格式截断点均待检验。'),
      ('B05','reject','关系矩阵中心化衰减既未检验坐标等距映射，也不能证明所有跨模型同构不存在。'),
      ('B06','narrow','数字集合内条件CE不等于纯逻辑、普遍知识或自然语言内容；语法也不能一概归入格式。'),
      ('B07','reject','有限阶有符号矩不保证任意历史无碰撞；先做3/4阶反例与KV短程充分性检查，不启动无前提的100K精确压缩。'),
      ('B08','rename_candidate','语言和Python均为文本，非跨模态；有限受控对齐与参数学习测试不能叫AGI底座证明。'),
      ('B09','reject','苹果知识链的流形滑动、格式切断等为未执行教学猜想，不写成已经观测到的内部机制。'),
      ('B10','retain_test','格式正交约束、高阶关系特征、跨表达对齐等未被全部否定，转为可执行候选并设置简单替代解释。')]
    evidence=[PRIOR/'natural_discovery.json.gz',PRIOR/'prediction/frozen.json',PRIOR/'signed_source/frozen.json',
      PRIOR/'format_content/decomposition_result.json',PRIOR/'middle_training/result.json',PRIOR/'scale/relational_analysis.json',
      PRIOR/'analysis/behavior.json',PRIOR/'signed_source/identity_recovery/result.json',PRIOR/'verification/model_checkpoint_fingerprints.json']
    plan=[
      {'phase':2736,'title':'有向来源关系与前缀更新图谱','tasks':['真实计数、桥接、去重与幅值审计','全H12坐标有向head读出与方向/角色打乱','同批无边/二阶/有符号/有向关系预测器及原生门解码','新来源及非标点前缀确认、近摘要异未来检索'],'gate':'16例精确核与对齐试运行通过后扩展；金标head仅训练或离线审计'},
      {'phase':2737,'title':'格式约束的全参数学习与原生形成','tasks':['非正交完整梯度及尺度归一审计','格式约束/无约束/方向反转/随机的实际范数控制','训练前预测局部效应及真实中层多步训练','内容/格式/完整答案分别评分、全标量因子查询'],'gate':'先真实原生suffix反传成本试运行；不得把约束残差命名纯语义'},
      {'phase':2738,'title':'混合操作、自然多关系与自身历史确认','tasks':['alias/mapping/conditional/addition交错且语义家族隔离','自然非标点内容与多族双语受控材料、三模型顺序非量化','同历史原生参照与独立历史、首分叉/答案/停止','原生来源到MLP读入/门/写回的跨层完整坐标账本'],'gate':'采集与能力pilot后冻结规模；不同模型保持自身tokenizer和坐标'},
      {'phase':2739,'title':'同目标自动续研：有限矩边界与预测状态接续','tasks':['三/四阶有符号矩反例及信息保留条件','可达自然近摘要配对的续接实验','有限上下文KV/查询预测状态基线及成本控制','新确认、理论拼图、全数值客户端与最终审计'],'gate':'自动执行有新增信息的有限后续；100K和无损压缩仅在明确充分性及成本门通过后，否则记录被否定前提并交付边界'}]
    immutable(BASE/'plan.json',{'timestamp':stamp(),'phases':plan,'priority':'观察→关系结构→未见预测→按需原生机制；非差分搬运为主',
      'primary_sources':[{'title':'GLU Variants Improve Transformer','url':'https://arxiv.org/abs/2002.05202'},
        {'title':'Neural Tangent Kernel','url':'https://arxiv.org/abs/1806.07572'},
        {'title':'Predictive Representations of State','url':'https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf'},
        {'title':'RoFormer','url':'https://arxiv.org/abs/2104.09864'}],
      'scope':'New finite integrated campaign. Samples/steps frozen in subordinate protocols after cheap pilots; no expectation or fabricated success counts.'})
    contract={'timestamp':stamp(),'source':snapshot(__file__),'attachments':[snapshot(p) for p in ATTACHMENTS],
      'memo_prefix_bytes':MEMO.stat().st_size,'memo_prefix_sha256':sha(MEMO),'previous_phases':[2732,2733,2734,2735],
      'counts':dict(counts),'validation_indices':validation.tolist(),'corrections':[{'id':i,'status':s,'finding':f} for i,s,f in audit],
      'gradient_sum_relative_error':decomposition,'nonorthogonal_cross_diagonal_max_abs':nonorthogonal,
      'bridge_reanalysis':bridge,'prior_evidence_sha256':{str(p.relative_to(PRIOR)):sha(p) for p in evidence},
      'plan_sha256':sha(BASE/'plan.json'),'status':'audit_done_new_experiments_pending','seconds':time.monotonic()-start}
    save(BASE/'contract.json',contract);ledger('contract_audit',contract['seconds'])
    print('UPDATE_CONTRACT',dict(counts),'gradient_sum',decomposition,'bridge',bridge,flush=True)

if __name__=='__main__':main()
