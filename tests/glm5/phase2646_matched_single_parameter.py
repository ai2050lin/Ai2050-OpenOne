"""Independent-entity maps and prospective fixed scalar V-weight forward tests."""
import gc,sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from phase2644_matched_coordinate_maps import analyze
from phase2643_matched_dual_adjoint_engine import MATERIAL,CONFIRM,LAYERS
from phase2636_precision_engine import load_precision
from phase2632_fulltoken_native_adjoints import module_at
from phase2624_scalar_forward_validation import digest_tensor
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2646_matched_single_parameter'
OLD=RESULT/'phase2632_fulltoken_native_adjoints'

def freeze():
    path=OUT/'protocol/scalar_selection.json'
    if path.exists():return read(path)
    old=next(r for r in read(OLD/'analysis/records.json') if r['frame_id']==20)
    sites=[]
    for l in LAYERS:
        s=next(s for s in old['sites'] if s['layer']==l and s['module']=='v_proj')
        for selector in ('diagnostic_max','matched_index'):
            sites.append({'layer':l,'j':s[selector+'_j'],'k':s[selector+'_k'],'selector':'prior_frame20_'+selector})
    p={'timestamp':datetime.now().astimezone().isoformat(),'source':str(OLD/'analysis/records.json'),'source_sha256':sha(OLD/'analysis/records.json'),
       'sites':sites,'case_rule':'confirmation units12..15, all8families x2languages, form0,target0,order0 =64',
       'dose':0.2,'signs':[-1,1],'actual_target':'original + signed0.2 matrixRMS then rounded to BF16-representable value, inserted into FP32 model',
       'selection_boundary':'Frozen before this finite validation and before confirmation collection. Historical frame20 rule uses NO current language field values. Original campaign wording before this material was too strong: timestamp is after material creation/initial collection, explicitly not a retrospective preregistration.'}
    save(path,p);return p

@torch.inference_mode()
def get_scores(model,row,contrasts):
    em=model.get_input_embeddings()(torch.tensor([row['prompt_ids']],device='cpu')).to('cuda:0')
    h=model.model(inputs_embeds=em,use_cache=False).last_hidden_state[0,-1]
    return [float((h*c).sum()) for c in contrasts],h.detach().cpu().numpy().copy()

def numeric():
    protocol=freeze();cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']=='confirmation' and (r['form'],r['target_index'],r['mention_order'])==(0,0,0)]
    records={r['case_index']:r for r in read(CONFIRM/'analysis/records.json')};winfo=read(OLD/'protocol/weights.json')
    model,info=load_precision('fp32');save(OUT/'protocol/model.json',info)
    before={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};conditions=[]
    for i,row in enumerate(cases):
        ci=row['case_index'];rec=records[ci];U=model.lm_head.weight.detach()
        contrasts=[(U[a]-U[b]).to('cuda:0') for a,b in (rec['native_ids'],rec['common_ids'])]
        baseline=[rec['native_margin'],rec['common_margin']]
        with np.load(CONFIRM/f'field/case_{ci:04d}.npz',allow_pickle=False) as z:
            gradients={}
            for site in protocol['sites']:
                l,j,k=site['layer'],site['j'],site['k'];x=z[f'L{l}_v_x'][:,k].astype('float64')
                for obj in ('native','common'):
                    terms=x*z[f'{obj}__L{l}_v_g'][:,j].astype('float64');gradients[(l,j,k,obj)]=(float(terms.sum()),float(terms[-1]))
            oldstate=z['normalized_boundary']
        noop,h=get_scores(model,row,contrasts)
        conditions.append({'kind':'noop','case_index':ci,'margin_changes':[a-b for a,b in zip(noop,baseline)],'state_max_error':float(np.max(np.abs(h-oldstate)))})
        assert noop==baseline and np.array_equal(h,oldstate)
        with torch.no_grad():
            for site in protocol['sites']:
                l,j,k=site['layer'],site['j'],site['k'];W=module_at(model,l,'v_proj').weight;original=W[j,k].clone();base=float(original)
                for sign in protocol['signs']:
                    target=float(torch.tensor(base+sign*protocol['dose']*winfo[f'L{l}_v_proj']['rms'],dtype=torch.bfloat16).float())
                    try:
                        W[j,k]=target;delta=float(W[j,k])-base;scores,h=get_scores(model,row,contrasts)
                        outputs={obj:{'effect':scores[n]-baseline[n],'predicted':delta*gradients[(l,j,k,obj)][0],
                                'last_token_predicted':delta*gradients[(l,j,k,obj)][1]} for n,obj in enumerate(('native','common'))}
                        conditions.append({'kind':'single_weight','case_index':ci,'family':row['family'],'language':row['language'],'unit':row['unit'],**site,
                            'sign':sign,'actual_delta':delta,'original_weight':base,'target_weight':target,'outputs':outputs,'state_l2_change':float(np.linalg.norm(h-oldstate))})
                    finally:W[j,k].copy_(original)
        save(OUT/'analysis/numeric_progress.json',{'cases':i+1,'total':len(cases),'conditions':len(conditions)})
        print('matched scalar validation',i+1,'/',len(cases),flush=True)
    after={str(l):digest_tensor(module_at(model,l,'v_proj').weight) for l in LAYERS};assert before==after
    save(OUT/'analysis/conditions.json',conditions);save(OUT/'analysis/restoration.json',{'before':before,'after':after,'disk_model_changed':False})
    del U,model;gc.collect();torch.cuda.empty_cache()
    summary={}
    for l in LAYERS:
        for obj in ('native','common'):
            rows=[r['outputs'][obj] for r in conditions if r['kind']=='single_weight' and r['layer']==l];den=sum(abs(r['effect']) for r in rows);active=[r for r in rows if abs(r['effect'])>=1e-5]
            summary[f'L{l}/{obj}']={'n':len(rows),'mean_abs_effect':den/len(rows),'relative_l1_prediction_error':sum(abs(r['effect']-r['predicted']) for r in rows)/den if den else None,
                'last_token_relative_l1_error':sum(abs(r['effect']-r['last_token_predicted']) for r in rows)/den if den else None,
                'effect_ge_1e-5_n':len(active),'sign_agreement':float(np.mean([np.sign(r['effect'])==np.sign(r['predicted']) for r in active])) if active else None}
    checks={'all64_cases':len(cases)==64,'all1088_conditions':len(conditions)==1088,'all_noops_exact':all(r['margin_changes']==[0,0] and r['state_max_error']==0 for r in conditions if r['kind']=='noop'),
        'all_four_matrices_restored':before==after,'all_four_parameter_values_match':all(info['all28_weight_values_exact'].values())}
    assert all(checks.values());save(OUT/'analysis/numeric_completion.json',{'summary':summary,'checks':checks})
    return summary,checks

def main():
    with threadpool_limits(limits=4):maps,checks=analyze('confirmation',OUT)
    numbers,ck=numeric();checks.update(ck)
    finish(2646,'独立实体512全坐标扩大与1088条件固定单V参数实测',OUT,{'provenance':str(Path(__file__)),'summary':{'maps':maps,'scalar_numeric':numbers},'checks':checks},
        '独立人名复核全部目标/顺序/句式坐标图谱；在四层V权重各两个固定标量上做真实正负小步前向。参数点只依据历史frame20选取，不按本轮语言数据优化。每步恢复权重，两个读出同时记录。',
        r'\Delta m_r\approx\Delta\theta_{jk}\sum_t\bar V^r_{t,j}X_{t,k};\quad E_r=\frac{\sum_i|\Delta m_{r,i}-\widehat{\Delta m}_{r,i}|}{\sum_i|\Delta m_{r,i}|};\quad \theta\leftarrow\theta_{original}\ \mathrm{after\ each\ test}.',
        '四个新实体单位的512全部条件；64前缀×(4层×2标量×2方向+1原样复测)=1088条件，1024实际单权重干预。非量化FP32同值权重、BF16可表示目标值，无donor搬运，无持久模型修改。',
        '每个实际参数的局部预测与跨条件地图是不同证据：前者检验导数算法，后者观察条件复用。扩大保留同实体输出条件与跨实体分割边界。选择坐标的真实冻结时间在初始采集之后、扩大采集之前；不把计划中“在材料之前”措辞伪装成既成预注册。',
        '固定8个参数仅覆盖四层V的小步数值，不是全模型参数因果普查，更不能推出某层普遍语义机制。两个目标分数不是完整名字生成控制；无必要性成功也不关闭全坐标路线。跨实体变化同时改变词汇和输出头行，不能把纹理迁移直接归因纯语义。',
        '交付重要坐标图谱和可查询标量接口，完成存储清理；下一研究仍沿条件操作—原生计算—输出编译，但应扩大输出功能及具体操作结构，而非只增加同模板样本。')

if __name__=='__main__':
    if '--freeze' in sys.argv:freeze()
    else:main()
