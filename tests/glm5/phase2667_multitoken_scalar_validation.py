"""Frozen real scalar changes: complete multi-token sequence and its three score parts."""
import gc
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2632_fulltoken_native_adjoints import module_at,LAYERS
from phase2624_scalar_forward_validation import digest_tensor
from phase2666_multitoken_parameter_engine import OUT as FP,PARTS,load_fp,score

OUT=RESULT/'phase2667_multitoken_scalar_validation'


def main():
    assert not (OUT/'analysis/final.json').exists();frozen=read(RESULT/'phase2653_output_function_scalar_validation/protocol/frozen.json');save(OUT/'protocol/frozen.json',frozen)
    cases=[r for r in read(FP/'material/cases.json') if r['unit'] in (4,5)];ref={r['case_index']:r for r in read(FP/'analysis/records.json')};winfo=read(RESULT/'phase2632_fulltoken_native_adjoints/protocol/weights.json')
    model,info=load_fp();save(OUT/'protocol/model.json',info);before={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};conditions=[]
    for i,r in enumerate(cases):
        ci=r['case_index'];base=score(model,r);error=abs(base['contrast']-ref[ci]['contrast']);assert error<1e-5;conditions.append({'kind':'noop','case_index':ci,'error':error,'baseline':base});deriv={}
        with np.load(FP/f'field/case_{ci:04d}.npz') as z:
            for site in frozen['sites']:
                l,j,k=site['layer'],site['j'],site['k'];dd={}
                for part in ('all',)+PARTS:
                    pp=[]
                    for label in ('Y','N'):
                        suffix='' if part=='all' else '_'+part;terms=z[f'{label}__L{l}_v_x'][:,k].astype('float64')*z[f'{label}__L{l}_v_g{suffix}'][:,j].astype('float64')
                        pp.append((float(terms.sum()),float(terms[len(r['prompt_ids'])-1]),float(terms[len(r['prompt_ids'])+len(r['canonical_answer_ids'][0 if label=='Y' else 1])-1])))
                    dd[part]=[a-b for a,b in zip(pp[0],pp[1])]
                deriv[l,j,k]=dd
        with torch.no_grad():
            for site in frozen['sites']:
                l,j,k=site['layer'],site['j'],site['k'];W=module_at(model,l,'v_proj').weight;original=W[j,k].clone();value=float(original)
                for sign in (-1,1):
                    target=float(torch.tensor(value+sign*.2*winfo[f'L{l}_v_proj']['rms'],dtype=torch.bfloat16).float())
                    try:
                        W[j,k]=target;delta=float(W[j,k])-value;now=score(model,r);dd=deriv[l,j,k]
                        conditions.append({'kind':'single_weight','case_index':ci,'family':r['family'],'language':r['language'],'unit':r['unit'],'polarity':r['polarity'],'mapping':r['mapping'],**site,
                            'sign':sign,'original_weight':value,'target_weight':target,'actual_delta':delta,'effect':now['contrast']-base['contrast'],'predicted':delta*dd['all'][0],
                            'prompt_last_only':delta*dd['all'][1],'branch_last_only':delta*dd['all'][2],
                            'parts':{p:{'effect':now[p]-base[p],'predicted':delta*dd[p][0]} for p in PARTS}})
                    finally:W[j,k].copy_(original)
        if (i+1)%8==0:save(OUT/'analysis/progress.json',{'cases':i+1,'total':128,'conditions':len(conditions)});save(OUT/'analysis/conditions_checkpoint.json',conditions);print('multitoken scalar',i+1,'/128',flush=True)
    after={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};del model;gc.collect();torch.cuda.empty_cache();assert before==after
    save(OUT/'analysis/conditions.json',conditions);save(OUT/'analysis/restoration.json',{'before':before,'after':after,'disk_model_changed':False});summary={}
    for l in LAYERS:
        rr=[r for r in conditions if r['kind']=='single_weight' and r['layer']==l];den=sum(abs(r['effect']) for r in rr)
        summary[f'L{l}']={'n':len(rr),'mean_abs_effect':den/len(rr),**{k:sum(abs(r['effect']-r[k]) for r in rr)/den if den else None for k in ('predicted','prompt_last_only','branch_last_only')},
            'parts':{p:{'mean_abs_effect':sum(abs(r['parts'][p]['effect']) for r in rr)/len(rr),
                'relative_l1_error':sum(abs(r['parts'][p]['effect']-r['parts'][p]['predicted']) for r in rr)/max(sum(abs(r['parts'][p]['effect']) for r in rr),1e-30)} for p in PARTS}}
    checks={'128_prefixes':len(cases)==128,'two_validation_entitypairs_perlanguage':len({r['unit'] for r in cases})==2,'2176_conditions':len(conditions)==2176,
        '2048_scalar_changes':sum(r['kind']=='single_weight' for r in conditions)==2048,'all_weights_restored':before==after,'noops_below1e-5':max(r['error'] for r in conditions if r['kind']=='noop')<1e-5,
        'all28_loaded_weights_exact':all(info['all28_weight_values_exact'].values())};assert all(checks.values())
    finish(2667,'2048单参数改动的多token内容/格式/停止独立校验',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '仅在内存将冻结真实V参数改为相邻BF16可表示的小剂量值，同值FP32前向评估完整规范序列和三类分数；每次恢复并核对整矩阵哈希。',
        r'\widehat{\Delta L_c}=\Delta\theta G_c;\quad E_c=\frac{\sum|\Delta L_c-\widehat{\Delta L_c}|}{\sum|\Delta L_c|},\quad c\in\{all,content,format,EOS\}.',
        '128验证前缀（每语言2实体对、八族、双极性/映射）×8冻结标量×2方向=2048实际改动，另128no-op；全token与仅prompt末位/仅分支末位错误近似分账。',
        '可以具体检查同一个权重的内容敏感性与格式/停止敏感性是否不同，数值预测只是测量校验；不把一项偏导方向称语义必要性或普遍编码。',
        '八坐标局部扰动，不是全参数有限干预穷举；验证前缀仍共享两个实体对和模板。接近零的分数组可能导致相对误差很大，应同时读绝对效应。',
        '顺序加载14B做1024同任务模型内全坐标复验，再整批审计发布和清理；不以本次局部数值结果替代语言规律证据。')


if __name__=='__main__':main()
