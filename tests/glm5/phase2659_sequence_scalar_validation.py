"""Real scalar perturbation predicts full normalized canonical sequence contrast."""
import gc
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2636_precision_engine import load_precision
from phase2632_fulltoken_native_adjoints import module_at,LAYERS
from phase2624_scalar_forward_validation import digest_tensor
from phase2655_truth_answer_contract import OUT as MATERIAL
from phase2658_sequence_parameter_engine import OUT as FP,sequence_scores

OUT=RESULT/'phase2659_sequence_scalar_validation';OLD=RESULT/'phase2632_fulltoken_native_adjoints'


def main():
    assert not (OUT/'analysis/final.json').exists()
    frozen=read(RESULT/'phase2653_output_function_scalar_validation/protocol/frozen.json');save(OUT/'protocol/frozen.json',frozen)
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['fp_selected'] and r['unit']==4];records={r['case_index']:r for r in read(FP/'analysis/records.json')};winfo=read(OLD/'protocol/weights.json')
    model,info=load_precision('fp32');save(OUT/'protocol/model.json',info);before={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};conditions=[]
    for i,r in enumerate(cases):
        ci=r['case_index'];rec=records[ci];base=sequence_scores(model,r);baseline=rec['contrast'];error=abs(base['contrast']-baseline)
        # Reduction order differs: autograd torch.sum vs Python sum of per-token values.
        assert error<1e-5
        conditions.append({'kind':'noop','case_index':ci,'error':error,'observed':base})
        with np.load(FP/f'field/case_{ci:04d}.npz') as z:
            deriv={}
            for s in frozen['sites']:
                l,j,k=s['layer'],s['j'],s['k'];pp=[]
                for label in ('Y','N'):
                    terms=z[f'{label}__L{l}_v_x'][:,k].astype('float64')*z[f'{label}__L{l}_v_g'][:,j].astype('float64');pp.append((float(terms.sum()),float(terms[len(r['prompt_ids'])-1]),float(terms[-1])))
                deriv[l,j,k]=[a-b for a,b in zip(pp[0],pp[1])]
        with torch.no_grad():
            for s in frozen['sites']:
                l,j,k=s['layer'],s['j'],s['k'];W=module_at(model,l,'v_proj').weight;original=W[j,k].clone();value=float(original)
                for sign in (-1,1):
                    target=float(torch.tensor(value+sign*.2*winfo[f'L{l}_v_proj']['rms'],dtype=torch.bfloat16).float())
                    try:
                        W[j,k]=target;delta=float(W[j,k])-value;now=sequence_scores(model,r);g=deriv[l,j,k]
                        conditions.append({'kind':'single_weight','case_index':ci,'family':r['family'],'language':r['language'],'probe':r['probe_index'],'polarity':r['polarity'],'mapping':r['mapping'],**s,
                            'sign':sign,'original_weight':value,'target_weight':target,'actual_delta':delta,'effect':now['contrast']-base['contrast'],
                            'predicted':delta*g[0],'prompt_last_only':delta*g[1],'branch_last_only':delta*g[2],
                            'first_token_effect':now['first_token_contrast']-base['first_token_contrast'],'eos_effect':now['eos_contrast']-base['eos_contrast']})
                    finally:W[j,k].copy_(original)
        if (i+1)%8==0:save(OUT/'analysis/progress.json',{'cases':i+1,'total':128,'conditions':len(conditions)});print('sequence scalar',i+1,'/128',flush=True)
    after={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};del model;gc.collect();torch.cuda.empty_cache();assert before==after
    save(OUT/'analysis/conditions.json',conditions);save(OUT/'analysis/restoration.json',{'before':before,'after':after,'disk_model_changed':False});summary={}
    for l in LAYERS:
        for mapping in ('all',0,1):
            rr=[r for r in conditions if r['kind']=='single_weight' and r['layer']==l and (mapping=='all' or r['mapping']==mapping)];den=sum(abs(r['effect']) for r in rr)
            summary[f'L{l}/mapping{mapping}']={'n':len(rr),'mean_abs_effect':den/len(rr),**{key:sum(abs(r['effect']-r[key]) for r in rr)/den if den else None for key in ('predicted','prompt_last_only','branch_last_only')},
                'active_ge_1e-5':sum(abs(r['effect'])>=1e-5 for r in rr),'active_sign_agreement':float(np.mean([np.sign(r['effect'])==np.sign(r['predicted']) for r in rr if abs(r['effect'])>=1e-5])) if any(abs(r['effect'])>=1e-5 for r in rr) else None}
    checks={'128_cases':len(cases)==128,'2176_conditions':len(conditions)==2176,'2048_actual_scalar_changes':sum(r['kind']=='single_weight' for r in conditions)==2048,
        'noop_error_lt1e_minus5':max(r['error'] for r in conditions if r['kind']=='noop')<1e-5,'all_weights_restored':before==after,'all28_loaded_weights_exact':all(info['all28_weight_values_exact'].values())}
    assert all(checks.values())
    finish(2659,'完整答案序列概率的2048真实单参数改动验证',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '同值FP32核心中改变一个BF16可表示的真实V权重，逐项恢复；两规范答案加EOS的完整归一概率差由全部token、两个分支的原生链式导数预测。',
        r'\widehat{\Delta L}=\Delta\theta[\sum_t\bar V^Y_{t,j}X^Y_{t,k}-\sum_t\bar V^N_{t,j}X^N_{t,k}];\quad E=\frac{\sum|\Delta L-\widehat{\Delta L}|}{\sum|\Delta L|}.',
        '128独立单位4前缀×(4层×2冻结坐标×2方向+1no-op)=2176条件、2048实际参数改动；每次含Yes/No两teacher-forced答案分支。与只算原prompt末token及只算分支末token两种错误近似分账。',
        '从固定首tokenlogit差推进到包含softmax全词表归一及后续结束选择的规范答案概率；若预测通过，支持实际共享参数在跨位置和跨答案上下文中的可计算性，不表明该参数是某语言族的独立必要齿。',
        '八个冻结坐标和小剂量局部扰动，不是全参数干预穷举。只两条短答案；行为分数不因数值预测通过而提升。no-op比较两种浮点求和实现容许1e-5误差并保存实际值。',
        '继续Qwen14非量化模型内坐标及行为复验，之后整批审计发布，不硬对齐模型坐标也不把数值链式闭合称语言机制闭合。')


if __name__=='__main__':main()
