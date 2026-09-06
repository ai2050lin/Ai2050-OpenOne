"""Exact teacher-forced answer-plus-EOS log probability, full-vocabulary normalized."""
import gc
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr
from phase2632_fulltoken_native_adjoints import AdjointCapture,LAYERS
from phase2636_precision_engine import load_precision
from phase2649_output_function_behavior import Capture
from phase2655_truth_answer_contract import OUT as MATERIAL
from phase2656_truth_answer_behavior import OUT as BF

OUT=RESULT/'phase2658_sequence_parameter_engine'


def branch_score(model,row,answer_index):
    prefix=row['prompt_ids'];answer=row['canonical_answer_ids'][answer_index];ids=prefix+answer;targets=answer+[row['eos_token_id']]
    em=model.get_input_embeddings()(torch.tensor([ids],device='cpu')).to('cuda:0');result=model.model(inputs_embeds=em,use_cache=False)
    states=result.last_hidden_state[0,len(prefix)-1:len(prefix)+len(answer)]
    logits=states.to('cpu')@model.lm_head.weight.T
    values=logits.log_softmax(-1)[torch.arange(len(targets)),torch.tensor(targets)]
    return values.sum(),values


@torch.inference_mode()
def sequence_scores(model,row):
    values=[branch_score(model,row,i)[1].tolist() for i in (0,1)]
    return {'branches':values,'contrast':sum(values[0])-sum(values[1]),'first_token_contrast':values[0][0]-values[1][0],
        'eos_contrast':values[0][-1]-values[1][-1]}


@torch.inference_mode()
def raw_checkpoint(model,row):
    cap=Capture(model);cap.enabled=True
    try:
        em=model.get_input_embeddings()(torch.tensor([row['prompt_ids']],device='cpu')).to('cuda:0');result=model.model(inputs_embeds=em,use_cache=False);state=result.last_hidden_state[0,-1]
        h=np.stack(cap.h);pack={'hidden_boundary':h[:,-1],'mlp_boundary':np.stack(cap.a),'normalized_boundary':arr(state)}
        if row['published']:pack['hidden_fulltoken']=h
        logits=model.lm_head.weight@state.cpu();first=int(logits.argmax());logits[first]=-float('inf');second=int(logits.argmax());a,b=row['common_readout_ids']
        return pack,{'fp32_own_top2':[first,second],'canonical_logit_gap':float((state.cpu()*(model.lm_head.weight[a]-model.lm_head.weight[b])).sum())}
    finally:cap.close();cap.reset()


def capture_branch(model,row,i):
    prefix=row['prompt_ids'];answer=row['canonical_answer_ids'][i];ids=prefix+answer;targets=answer+[row['eos_token_id']];cap=AdjointCapture(model)
    try:
        with torch.no_grad():em=model.get_input_embeddings()(torch.tensor([ids],device='cpu')).to('cuda:0')
        em=em.detach().requires_grad_(True);result=model.model(inputs_embeds=em,use_cache=False);states=result.last_hidden_state[0,len(prefix)-1:len(prefix)+len(answer)]
        logits=states.to('cpu')@model.lm_head.weight.T;values=logits.log_softmax(-1)[torch.arange(len(targets)),torch.tensor(targets)];loss=values.sum();loss.backward()
        boundary=len(prefix)-1;pack={'hidden_adjoint_prompt_boundary':np.stack([arr(v.grad[0,boundary]) for v in [em]+cap.hidden]),
            'mlp_adjoint_prompt_boundary':np.stack([arr(v.grad[0,boundary]) for v in cap.a]),'normalized_prompt_boundary':arr(states[0])}
        for l in LAYERS:
            x,v=cap.linears[(l,'v_proj')];pack[f'L{l}_v_x']=arr(x[0]);pack[f'L{l}_v_value']=arr(v[0]);pack[f'L{l}_v_g']=arr(v.grad[0])
        info={'input_ids':ids,'target_ids':targets,'prediction_positions':list(range(boundary,boundary+len(targets))),'logprobs':values.detach().tolist(),'total_logprob':float(loss.detach())}
        assert all(np.isfinite(v).all() for v in pack.values());return pack,info
    finally:cap.close();cap.reset()


def main():
    assert not (OUT/'analysis/final.json').exists()
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['fp_selected']];behavior={r['case_index']:r for r in read(BF/'analysis/records.json')}
    model,info=load_precision('fp32');save(OUT/'protocol/model.json',info)
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(MATERIAL/'material/cases.json'),'case_indices':[r['case_index'] for r in cases],
        'objective':'canonical answer then EOS, logprobYes-logprobNo; two teacher-forced separate branches, fullvocab log_softmax, no answer-class mass claim',
        'native_decision':'for64publishedcases, readraw FP32 H/MLP at BF16generated first recognizable answer prefix; no branch moved across cases',
        'alltoken_parameter':'G=sum(gYes*xYes)-sum(gNo*xNo), g is branch total logprob adjoint, everyinput token included. Frozen old8 scalars fornextphase.'})
    OUT.joinpath('field').mkdir(parents=True,exist_ok=True);records=[];manifest=[]
    for i,r in enumerate(cases):
        torch.cuda.empty_cache();ci=r['case_index'];pack,base=raw_checkpoint(model,r);bb=behavior[ci];branches=[]
        for bi,label in enumerate(('Y','N')):
            values,meta=capture_branch(model,r,bi);pack.update({label+'__'+k:v for k,v in values.items()});branches.append(meta);del values;gc.collect();torch.cuda.empty_cache()
        prefix_exact=np.array_equal(pack['Y__normalized_prompt_boundary'],pack['N__normalized_prompt_boundary'])
        assert prefix_exact
        decision=bb['decision'];dmeta=None
        if r['published'] and decision is not None:
            drow={**r,'prompt_ids':decision['prefix_ids']};dp,dm=raw_checkpoint(model,drow)
            pack.update({'decision_'+k:v for k,v in dp.items()});dmeta={**dm,'step':decision['step'],'prefix_ids':decision['prefix_ids']};del dp
        with np.load(BF/f'field/case_{ci:04d}.npz') as b:
            embedding_exact=np.array_equal(pack['hidden_boundary'][0],b['hidden_boundary'][0]);precision_error=float(np.linalg.norm(pack['hidden_boundary']-b['hidden_boundary'])/max(np.linalg.norm(b['hidden_boundary']),1e-30))
        assert embedding_exact
        y,n=[v['total_logprob'] for v in branches];first=branches[0]['logprobs'][0]-branches[1]['logprobs'][0];end=branches[0]['logprobs'][-1]-branches[1]['logprobs'][-1]
        rec={k:r[k] for k in ('case_index','case_id','field_set','published','family','language','unit','probe_index','polarity','mapping')}
        rec.update(**base,branches=branches,contrast=y-n,first_token_contrast=first,eos_contrast=end,decomposition_error=abs(y-n-first-end),
            native_first_ids=bb['native_ids'],decision=dmeta,branch_causal_prefix_bitwise=prefix_exact,embedding_exact=embedding_exact,bf16_fp32_boundary_relative_l2=precision_error,
            raw_vs_branch_first_gap_error=abs(base['canonical_logit_gap']-first))
        path=OUT/f'field/case_{ci:04d}.npz';np.savez_compressed(path,**pack);manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':ci,'published':r['published']});records.append(rec);del pack
        if (i+1)%16==0:save(OUT/'analysis/progress.json',{'cases':i+1,'total':256});print('fullanswer sequence factors',i+1,'/256',flush=True)
    nograd=all(p.grad is None for p in model.parameters());del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/records.json',records);save(OUT/'analysis/raw_manifest.json',manifest)
    summary={'cases':len(records),'same_shape_branch_prefix_bitwise':sum(r['branch_causal_prefix_bitwise'] for r in records),
        'first_vs_full_sequence_sign_different':int(sum(np.sign(r['contrast'])!=np.sign(r['first_token_contrast']) for r in records)),
        'maximum_decomposition_error':max(r['decomposition_error'] for r in records),'maximum_raw_vs_branch_first_gap_error':max(r['raw_vs_branch_first_gap_error'] for r in records),
        'published_decision_states':sum(r['decision'] is not None for r in records),'published_decision_after_first':sum(r['decision'] is not None and r['decision']['step']>0 for r in records)}
    checks={'256_cases':len(records)==256,'two_branches_allcausal_prefix_exact':all(r['branch_causal_prefix_bitwise'] for r in records),'embeddings_exact':all(r['embedding_exact'] for r in records),
        'no_weight_gradients':nograd,'all28_weights_same':all(info['all28_weight_values_exact'].values()),'two_step_logprob_decomposition':summary['maximum_decomposition_error']<1e-5}
    assert all(checks.values())
    record_result({'provenance':str(Path(__file__)),'summary':summary,'checks':checks})


def record_result(result):
    finish(2658,'完整规范答案加结束符概率差的全token逐参数算法',OUT,result,
        '两条规范答案各自做teacher-forced全词表归一概率，反向传播每条完整序列的分数；同一个真实V参数的导数是两分支所有输入位置乘积和之差，而非搬运激活或只取最后位置。',
        r'L=\sum_{s=1}^{|Y|+1}\log P(Y_s|x,Y_{<s})-\sum_{s=1}^{|N|+1}\log P(N_s|x,N_{<s});\quad \frac{\partial L}{\partial W_{jk}}=\sum_t\bar V^Y_{t,j}X^Y_{t,k}-\sum_t\bar V^N_{t,j}X^N_{t,k}.',
        '256前缀（单位0/4、八族双语、双探问/极性/映射，form0/v0/order0）；规范Yes/No或是/否都为1token再加EOS，故各为2个预测步骤。37H/36MLP全边界伴随、四层V全部token因子。64实例另记录实际BF16答案识别前缀对应的FP32原生状态。',
        '完整规范序列分数明确包含结束选择，既计算首答案词，也计算该词之后停止的概率。分支相同形状下因果前缀状态逐位校验；原长与加答案后的形状数值差异另记。可算出具体参数通过哪些token影响完整规范回答，但不是所有自由答案总概率。',
        '只两条规范短答案，不覆盖解释、大小写、标点等所有等价回答；不是自主生成成功率。FP32是同值数值对照，不覆盖自然BF16行为。只256前缀且共享实体模板，不能宣称全语言概率编译闭合。',
        '用独立单位4的128前缀，直接修改8个冻结真实权重并预测完整答案概率差，检验共享跨位置、跨分支的参数级算法。')


if __name__=='__main__':main()
