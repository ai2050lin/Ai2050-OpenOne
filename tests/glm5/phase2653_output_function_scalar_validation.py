"""Fixed actual V scalar values tested across four output-function cells."""
import gc,sys
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2650_output_function_adjoints import MATERIAL,CONFIRM,LAYERS
from phase2636_precision_engine import load_precision
from phase2632_fulltoken_native_adjoints import module_at
from phase2624_scalar_forward_validation import digest_tensor
from phase2646_matched_single_parameter import get_scores

OUT=RESULT/'phase2653_output_function_scalar_validation'
OLD=RESULT/'phase2632_fulltoken_native_adjoints'

def freeze():
    path=OUT/'protocol/frozen.json'
    if path.exists():return read(path)
    old=RESULT/'phase2646_matched_single_parameter/protocol/scalar_selection.json';source=read(old)
    r={'timestamp':datetime.now().astimezone().isoformat(),'sites':source['sites'],'source_sha256':sha(old),'scale':0.2,'signs':[-1,1],
       'case_rule':'confirmation unit12, form0,target0,order0;all8families x2languages x4modecells=64',
       'selection':'identical8 old scalar coordinates frozen in2648 contract; no current field or language effect optimization',
       'target_value':'original+signed0.2 matrixRMS rounded to actualBF16-representable value, inserted intoFP32same-valued core; restore every step'}
    save(path,r);return r

def main():
    p=freeze();cases=[r for r in read(MATERIAL/'material/cases.json') if (r['unit'],r['form'],r['target_index'],r['mention_order'])==(12,0,0,0)]
    records={r['case_index']:r for r in read(CONFIRM/'analysis/records.json')};winfo=read(OLD/'protocol/weights.json')
    model,info=load_precision('fp32');save(OUT/'protocol/model.json',info);before={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS}
    conditions=[]
    for i,r in enumerate(cases):
        ci=r['case_index'];rec=records[ci];U=model.lm_head.weight.detach();contrasts=[(U[a]-U[b]).to('cuda:0') for a,b in (rec['native_ids'],rec['common_ids'])];baseline=[rec['native_margin'],rec['common_margin']]
        with np.load(CONFIRM/f'field/case_{ci:04d}.npz') as z:
            gradients={}
            for s in p['sites']:
                l,j,k=s['layer'],s['j'],s['k'];x=z[f'L{l}_v_x'][:,k].astype('float64')
                for obj in ('native','common'):
                    terms=x*z[f'{obj}__L{l}_v_g'][:,j].astype('float64');gradients[(l,j,k,obj)]=(float(terms.sum()),float(terms[-1]))
            oldstate=z['normalized_boundary']
        base,state=get_scores(model,r,contrasts);assert base==baseline and np.array_equal(state,oldstate)
        conditions.append({'kind':'noop','case_index':ci,'mode':r['mode'],'changes':[x-y for x,y in zip(base,baseline)],'state_max_error':float(np.max(np.abs(state-oldstate)))})
        with torch.no_grad():
            for s in p['sites']:
                l,j,k=s['layer'],s['j'],s['k'];W=module_at(model,l,'v_proj').weight;original=W[j,k].clone();value=float(original)
                for sign in p['signs']:
                    target=float(torch.tensor(value+sign*p['scale']*winfo[f'L{l}_v_proj']['rms'],dtype=torch.bfloat16).float())
                    try:
                        W[j,k]=target;delta=float(W[j,k])-value;scores,state=get_scores(model,r,contrasts)
                        values={obj:{'effect':scores[n]-baseline[n],'predicted':delta*gradients[(l,j,k,obj)][0],'last_token_predicted':delta*gradients[(l,j,k,obj)][1]} for n,obj in enumerate(('native','common'))}
                        conditions.append({'kind':'single_weight','case_index':ci,'family':r['family'],'language':r['language'],'mode':r['mode'],**s,
                            'sign':sign,'original_weight':value,'target_weight':target,'actual_delta':delta,'outputs':values,'state_l2_change':float(np.linalg.norm(state-oldstate))})
                    finally:W[j,k].copy_(original)
        save(OUT/'analysis/progress.json',{'cases':i+1,'total':64,'conditions':len(conditions)});print('outputfunction scalar',i+1,'/64',flush=True)
    after={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};assert before==after
    save(OUT/'analysis/conditions.json',conditions);save(OUT/'analysis/restoration.json',{'before':before,'after':after,'disk_model_changed':False})
    from phase2653_causal_shape_control import run_controls
    source_control=run_controls(model)
    del model,U;gc.collect();torch.cuda.empty_cache();summary={}
    for mode in ('all','name','cloze','truth_a','truth_b'):
        for l in LAYERS:
            for obj in ('native','common'):
                rr=[r['outputs'][obj] for r in conditions if r['kind']=='single_weight' and r['layer']==l and (mode=='all' or r['mode']==mode)];den=sum(abs(r['effect']) for r in rr);active=[r for r in rr if abs(r['effect'])>=1e-5]
                summary[f'{mode}/L{l}/{obj}']={'n':len(rr),'mean_abs_effect':den/len(rr),'relative_l1_error':sum(abs(r['effect']-r['predicted']) for r in rr)/den if den else None,
                    'last_only_l1_error':sum(abs(r['effect']-r['last_token_predicted']) for r in rr)/den if den else None,'effect_ge_1e-5_n':len(active),
                    'sign_agreement':float(np.mean([np.sign(r['effect'])==np.sign(r['predicted']) for r in active])) if active else None}
    checks={'64_cases':len(cases)==64,'1088_conditions':len(conditions)==1088,'all_noops_exact':all(r['changes']==[0,0] and r['state_max_error']==0 for r in conditions if r['kind']=='noop'),
        'all_weights_restored':before==after,'all28_disk_values_same':all(info['all28_weight_values_exact'].values())}
    assert all(checks.values())
    checks['all64_source_shape_controls']=source_control['summary']['pair_conditions']==64
    finish(2653,'四输出功能固定单V参数验证与因果源状态数值校准',OUT,{'provenance':str(Path(__file__)),'summary':{'scalar':summary,'source_shape':source_control['summary']},'checks':checks},
        '直接改变一个真实共享V权重，不移植任何donor激活。相同8个历史坐标在四输出功能下各自用全token伴随预测，同时检验错误的末位置近似；原生IDs与固定任务读出分别记录。',
        r'\widehat{\Delta m_r}=\Delta\theta_{jk}\sum_t\bar V^r_{t,j}X_{t,k};\quad E_r=\frac{\sum|\Delta m_r-\widehat{\Delta m_r}|}{\sum|\Delta m_r|};\quad \theta\leftarrow\theta_{original}.',
        '独立单位12的八族双语四模式64前缀×(4层×2固定坐标×2方向+1no-op)=1088条件，1024实际参数修改。另取32个源状态数值差异最坏对和32个固定对（可重叠），做256次原长/等长屏蔽填充前向，检验同因果前缀的形状数值误差。保持同值FP32非量化核心，参数目标取BF16可表示数值，全部恢复后核对完整四矩阵哈希。',
        '验证的是跨不同输出功能的原生标量预测公式能否继续工作，而不是该标量对整个语言族必要。全部微弱和符号不符效应保留，按模式和层分账，不用总平均掩盖差异。',
        '只有8个参数、局部小剂量和两个首token分数；不是自然完整答案控制，也非全模型干预普查。FP32数值模型不同于自然BF16前向，不可替换真实行为记录。',
        '综合独立全坐标规则与源位置条件敏感度，接入客户端并清理未展示原包；仍不把精确链式记账直接叫作语言编码机制。')

if __name__=='__main__':
    if '--freeze' in sys.argv:freeze()
    else:main()
