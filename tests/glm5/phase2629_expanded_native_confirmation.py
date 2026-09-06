"""Same-goal continuation: fresh contexts, larger native scalar audit, and generation."""
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2621_native_language_material import build
from phase2621_native_behavior_run import load_model,run,stats
from phase2622_native_field_capture import collect,summarize
from phase2624_scalar_forward_validation import validate
from phase2623_native_parameter_algorithms import field
from phase2628_native_atlas_delivery import panel,row,ASSET

OUT=RESULT/'phase2629_expanded_native_confirmation'

def main():
    save(OUT/'protocol/frozen.json',{'indices':list(range(12,36)),'forms':[0],'variants':[0,1],'n':768,'base_items':384,
        'algorithm_source_sha256':sha(TESTS/'phase2623_native_parameter_algorithms.py'),'no_refit_no_layer_search':True,
        'pre_run_addition':'save and validate model-own top1/top2 native objective separately; no task answer needed for native objective, all-unit formulas unchanged',
        'scalar_validation':'indices12 and24, all16groups bothvariants =>64cases x34conditions=2176; no-op controls, actual BF16 deltas, weight hash restoration',
        'independence_limits':'new subject/context indices, not every vocabulary token is disjoint; fixed sense anchor Apple and same language templates recur'})
    model,tok=load_model('qwen4');cases=build(tok,start=12,stop=36,forms=(0,));save(OUT/'material/cases.json',cases)
    behavior=run(model,tok,cases,OUT)
    records=collect(model,tok,cases,OUT,save_full=False,include_native=True)
    mapping={r['case_id']:{**r,'source_index':i,'objective_token_ids':r['native_top2_ids'],'semantic_first_token_distinct':False} for i,r in enumerate(records)}
    selected=[r for r in cases if r['index'] in (12,24)]
    interventions,valid,restored=validate(model,tok,selected,mapping,OUT,source_dir=OUT,effect_field='native_neuron_delete_effect')
    # No answer labels enter these sensitivities. Compare with the task-conditioned analysis, keeping all K units.
    G=field('gradient_a',OUT);N=field('native_gradient_a',OUT);cos=[]
    for i in range(len(cases)):
        a=G[i].astype('float64');b=N[i].astype('float64');cos.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30)))
    comparison={g:{'mean_signed_cosine':float(np.mean([cos[i] for i,r in enumerate(cases) if r['family']+'/'+r['language']==g])),
        'mean_abs_cosine':float(np.mean([abs(cos[i]) for i,r in enumerate(cases) if r['family']+'/'+r['language']==g]))} for g in sorted({r['family']+'/'+r['language'] for r in cases})}
    payload=read(ASSET);rr=[];hh=[]
    for i,r in enumerate(cases):
        if r['index']!=12:continue
        rr.extend([row(r['case_id']+'/task-first-token gradient',G[i],phase=2629,kind='task_conditioned_neuron_derivative'),
                   row(r['case_id']+'/native-own-top2 gradient',N[i],phase=2629,kind='label_free_neuron_derivative')])
        for l in (0,1,6,18,36):hh.append(row(r['case_id']+f'/raw checkpoint{l}',field('hidden_anchor_boundary',OUT)[i,l,-1],phase=2629,kind='embedding' if l==0 else 'raw_hidden_coordinate',layer=l))
    panels=[panel('phase2629_native_vs_task','Fresh contexts: label-free vs task-conditioned native unit derivatives',9728,rr,'all physical k; source natural states identical, only output objective changes'),
        panel('phase2629_expanded_raw','768 fresh contexts: FP32 native coordinates',2560,hh,'all physical raw boundary coordinates; embedding and final norm distinguished')]
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2629_')]+panels;payload['phase']=2629
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    result={'provenance':str(Path(__file__)),'summary':{'fresh_context_prompts':len(cases),'fresh_base_items':384,'behavior':stats(behavior),'native_equations':summarize(records),
        'native_objective_actual_forward':valid,'task_vs_native_neuron_gradients':comparison},
        'checks':{'all768_behavior':len(behavior)==768,'all768_native_fields':len(records)==768,'all64_validation_cases':len(selected)==64,'all2176_actual_validation_forwards':len(interventions)==2176,
            'weights_restored':restored,'no_op_identical':all(r['observed_margin_change']==0 for r in interventions if r['kind']=='noop'),
            'new_subject_indices':all(r['index']>=12 for r in cases),'all_native_gradient_coordinates':N.shape==(768,9728),'finite_native_objective_fields':np.isfinite(N).all()},
        'client_asset_sha256':sha(ASSET),'next_goal_same':True,
        'next_large_stage':'token-by-token endogenous output objective; real-arithmetic vs BF16 local sensitivity floors; earlier layer actual QKV/MLP native Jacobian propagation without donor vectors; multiple language families, independent lexical and output-identity controls'}
    finish(2629,'自动续研：768新上下文与2176次无答案标签单参数前向扩大确认',OUT,result,
        '固定原生算法，不再选层或拟合方向；先扩大全部八族自然生成和坐标测绘，再在模型自己选出的top1/top2输出token上验证最终层单坐标/神经元/权重公式。',
        r'(y,z)=\operatorname{Top2}(U N(h));\quad m_{native}=z_y-z_z;\quad\partial_{W_{jk}}m_{native}=a_k\partial_{h_j}m_{native};\quad \text{target-label-free}\ne\text{semantic-feature-identified}.',
        'item12—35，八族中英、固定一表面、双变体共768prompt/384基础item；另选12和24共64例×34实际前向=2176，规模为前轮两倍。所有generated原文、两种输出目标的全9728神经元导数、逐参数查询因子均保留。',
        '相同自然状态切换输出对比就能改变导数纹理，因此输出条件不能隐藏在“语义编码”标签下。无答案目标分析不需要测试donor，也不需要正确答案参与提取，是可复算的原生参数级工具；依然不把最大局部删除者叫语言齿轮。',
        '新上下文不是全部词汇互斥锁箱，Apple、类别与模板复用；native top2可能是空格、大小写或格式token，并不自动对应语义分支。真实前向改变首token不等于整段生成改变。后层单参数可计算不代表早层生成该状态的机制已知。',
        '本轮完整合同和扩大确认已执行；研究目标仍相同，已更新既有每小时续研安排。下一大任务推进生成轨迹中的原生坐标读取及更早层真实参数链，不回到整体差分搬运循环；交付核验与未发布原场清理追加本Phase补记。')

if __name__=='__main__':main()
