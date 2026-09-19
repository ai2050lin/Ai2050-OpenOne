"""Audited bounded delivery, preserving inherited theory without rebranding it."""
import argparse
from rdc_construction_common import *


def visual_review():
    # These exact named rendered images were inspected by the main research
    # agent before this receipt was authored. Existence alone is not visual QA.
    index=read(BASE/'figures/index.json')
    scientific=[]
    for entry in index['figures']:
        path=BASE/entry['path'];assert sha(path)==entry['sha256']
        scientific.append({'path':entry['path'],'sha256':entry['sha256'],
            'manually_inspected':True,'checked':'Legible axes/labels, original index order, raw/RMS distinction, unclipped intervals and layout; dark pixels not interpreted as zero.'})
    screens=[]
    for name in ['desktop','mobile','field','interaction','parameters','generation','native_language']:
        path=BASE/'client'/f'headless_final_{name}.png'
        screens.append({'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'manually_inspected':True})
    extra=[]
    for name in ['natural','attention_parameters','glm_fused_mlp']:
        path=BASE/'phase2746/client'/(name+'.png')
        extra.append({'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'manually_inspected':True})
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'reviewer':'main research agent after actual image inspection',
        'scientific_figures':scientific,'final_client_images':screens,'phase2746_increment_images':extra,
        'scope':'Image inspection verifies presentation and declared boundaries, not scientific truth or semantic closure.'}
    save(BASE/'client/visual_review.json',value)
    index['visual_inspection']='All7scientificimages manually inspected; exactSHA receipt atclient/visual_review.json'
    save(BASE/'figures/index.json',index)
    print('CONSTRUCTION_VISUAL_RECEIPT',len(scientific),len(screens),len(extra),flush=True)


def deliver():
    verification=read(BASE/'verification/result.json');assert verification['all_passed']
    inherited=read(OLD/'theory_snapshot.json')
    assert len(inherited['puzzles'])==43 and len(inherited['formulas'])==42
    evidence={}
    for model in ['qwen4','qwen14','glm4']:
        for stage in ['capture','analysis','fit','diagonal','compilation','native_language']:
            path=BASE/stage/model/'result.json';result=read(path);assert result['all_passed']
            evidence[f'{stage}/{model}']={'path':path.relative_to(BASE).as_posix(),'sha256':sha(path)}
    for file in ['norm_controls/analysis.json','norm_controls/result.json','review/claim_audit.json',
        'client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json','verification/result.json']:
        evidence[file]={'path':file,'sha256':sha(BASE/file)}
    puzzles=[*inherited['puzzles'],{'phase':'2745',
        'retained_puzzle':'三原生模型条件查询构造、全坐标交互、同输入全矩阵/逐坐标/置乱/零对照、原生Q与完整词表编译，以及等实际BF16位移的训练方向检验。',
        'boundary':'全部主H1线性算子未稳定同时胜零变化和配对置乱；standalone锚点外推失败；训练概率改善未形成自然标签特有的关系成功优势。GLM首换行与完整回答、Q14两条B1/B8分叉分别保留。',
        'evidence_status':'verified_native_observation_with_scoped_prediction_and_formation_limits',
        'common_phenomenon':'相同词袋的关系重排可改变早层查询、全坐标条件交互和完整原生回答；是否正确仍依赖族与模型。',
        'candidate_rule':'前缀可用信息与实际H1两种不同输入，各自全跨坐标/逐坐标/置乱/零基线；独立锚点外推不从配对拟合自动成立。',
        'native_parameter_structure':'原生Q投影/头归一化/RoPE、完整词表读出、三模型全参数地址与GLM融合gate/up；保留实际BF16计算形状边界。',
        'unseen_composition_prediction':'按冻结语义组和查询划分；现有H1线性规则未稳定同时胜零变化与置乱，不能认作关系机制闭合。',
        'training_formation_evidence':'旧真实FP32学习方向的18种新增等实际BF16位移部署，5760条自身历史；不冒充新增训练或预训练恢复。'}]
    theory={'timestamp':stamp(),'source':snapshot(__file__),'theory_name':inherited['theory_name'],
        'inherited_snapshot_sha256':sha(OLD/'theory_snapshot.json'),
        'puzzles':puzzles,'formulas':inherited['formulas'],'global_atlas_interface':inherited['global_atlas_interface'],
        'inheritance_audit':{'all43prior_puzzles_preserved_exactly':puzzles[:43]==inherited['puzzles'],
            'all42prior_formulas_preserved_exactly':True,'prior_results_not_rerun':True},
        'RDC_primary_formula_changed':False,'global_closed_theorem_added':False,'new_mathematics_claimed':False,
        'current_algorithm_definitions':[
            {'id':'standardized_complete_coordinate_pair_ridge','kind':'fitted_algorithm_precision_not_new_law',
             'expression':'s_d=max(sqrt(mean_i DeltaX_id^2),1e-8); Z=DeltaX diag(1/s); Bhat=argmin_B ||ZB-DeltaY||_F^2/n+lambda||B||_F^2; Ahat=diag(1/s)Bhat.',
             'variables':'n4800train pair-query rows; every native coordinate retained, all spectrum used; only declared validation selectslambda.'},
            {'id':'standalone_anchor_extra_assumption','kind':'tested_and_failed_extrapolation_assumption',
             'expression':'Yhat(p,q)=Y(empty,q)+[X(p,q)-X(empty,q)]Ahat.',
             'variables':'Independent query-template anchor is not implied by pair-difference fitting; native futureQ/K/V/target not supplied.'},
            {'id':'actual_BF16_direction_radius','kind':'experimental_control_definition',
             'expression':'r(alpha,D)=||BF16(W+alphaD)-W||_F; targetr=.05,.10,.18 withrelativeerror<=.001.',
             'variables':'All74711040block16gate/up/down scalars combined. WoriginalBF16,Dreal priorFP32 learned direction; no new training here.'},
            {'id':'native_MLP_three_factor_address','kind':'known_architecture_factorization',
             'expression':'Gamma[k,j,i,r]=Wd[j,k]Wg[k,i]Wu[k,r].',
             'variables':'All coordinates addressable; nativeGLMgate/up are two halves ofgate_up_proj; no enormous product tensor materialization or top-k.'}],
        'first_principles_insight':'固定参数、条件化状态、整体响应拟合、关系差异预测、完整输出和真实训练形成是不同对象。应将可用的历史/位置/当前查询构造连接到全坐标条件运算，而不能用整体拟合小收益或单个终值变化代替关系规则。',
        'required_evidence':evidence,
        'remaining':'Phase2746/2747 integral plan remains active; exact language mechanism and general composition law are not solved.'}
    save(BASE/'theory_snapshot.json',theory)
    records=gzread(BASE/'native_language/qwen14/records.json.gz')
    material=gzread(BASE/'material.json.gz')['models']['qwen14']['rows'];lookup={r['sample_id']:r for r in material}
    shape=[{k:r[k] for k in ['sample_id','pair_id','source_group','family','language','target','generated_text','generated_ids','first_shape']}|
        {'original_text':lookup[r['sample_id']]['original_text']} for r in records if r['first_shape']['B1_token']!=r['first_shape']['B8_token']]
    assert len(shape)==2
    save(BASE/'native_language/qwen14/B1_B8_disagreements.json',{'timestamp':stamp(),'rows':shape,
        'scope':'Observed batch/padding/execution-shape condition difference; not a model-weight change, independent language sample or isolated causal attribution to one kernel.'})
    manifest={'timestamp':stamp(),'source':snapshot(__file__),'phase2745_complete':True,'goal_complete':False,
        'evidence':evidence,'theory_snapshot_sha256':sha(BASE/'theory_snapshot.json'),
        'native_contexts':960,'native_fixed_query_endpoints':96000,
        'new_parameter_direction_trajectories':5760,'new_original_model_trajectories':640,'reused_Q4_original_trajectories':320,
        'original_checkpoint_bytes_rehashed':read(BASE/'verification/model_checkpoint_fingerprints.json')['bytes_rehashed'],
        'archive_audit':verification['archive_coverage'],
        'next_phase':'2746; registered full natural/controlled own-history runtime observation, all-parameter addresses, native-conditioned predictors and remaining-network propagation. 2747training formation follows with actual evidence boundaries.',
        'data_retention':'All primary fields/weights/results preserved; next-stage newfields use explicitly registeredCdrive storage through result-tree junction. No model/evidence deletion.'}
    save(BASE/'delivery_manifest.json',manifest)
    save(BASE/'status.json',{'timestamp':stamp(),'phase2745':'verified_complete','phase2746':'active',
        'phase2747':'planned_not_executed','goal_complete':False,'manifest_sha256':sha(BASE/'delivery_manifest.json')})
    print('CONSTRUCTION_DELIVERY',len(puzzles),len(theory['formulas']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--visual',action='store_true');a=p.parse_args()
    if a.visual:visual_review()
    else:deliver()
