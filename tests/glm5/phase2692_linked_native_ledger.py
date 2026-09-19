"""Full-coordinate observed calculation ledger, not semantic mechanism closure.

CPU artifact preparation may precede2691 completion; phase finalization cannot.
Fixed256 source/MLP fields and natural-cache readout remain SEPARATE protocols.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,gc,shutil
import numpy as np
import torch
from safetensors import safe_open
from phase2620_native_coordinate_contract import *
from phase2679_source_coordinate_ledger import attention_ledger,conditional_norm_ledger

OUT=RESULT/'phase2692_linked_native_ledger'
FIELD=RESULT/'phase2687_role_qkv_field'
TERMS=RESULT/'phase2688_native_qkv_terms'
MATERIAL=RESULT/'phase2686_independent_role_contract/material/initial.json'
MODEL=ROOT/'models/hf/qwen3-4b'
LAYERS=(0,5,17,23,26,27,28,35)
FLOOR=8*1024**3


def unbits(a):return (a.astype(np.uint32)<<16).view(np.float32)


def prepare():
    assert read(TERMS/'analysis/final.json')['all_checks_passed']
    rows=[r for r in read(MATERIAL) if r['parameter_published']];assert len(rows)==16
    assert read(MODEL/'config.json')['attention_bias'] is False
    c={'source_material_sha256':sha(MATERIAL),'source_final_sha256':sha(FIELD/'analysis/final.json'),
        'source_manifest_sha256':sha(FIELD/'analysis/published_manifest.json'),
        'checkpoint_index_sha256':sha(MODEL/'model.safetensors.index.json'),'cases':[r['case_id'] for r in rows],'layers':LAYERS,
        'code_sha256':sha(Path(__file__)),'accounting_helper_sha256':sha(TESTS/'phase2679_source_coordinate_ledger.py'),
        'precision':'All recorded model fields/native weights BF16. NumPy FP64 analysis only; no model forward/weights edits.',
        'source_path':'Every head/source/headcoordinate and all2560 residual coordinates. Stream full source×inputcoord branches through actual complete gate/up/down matrices and all9728MLPunits. NoTopK/PCA/donor.',
        'normalization':'Observed endogenous RMS denominator supports CONDITIONAL additive accounting, NOT an ablation prediction. Actual2689 scalar interventions are different evidence; do not conflate.',
        'fixed_vs_natural':'Fixed256 H_before/source/MLP/H_after ledger uses same recorded field. Natural readout starts at independently saved actual post-final-norm natural state. NEVER link its input backward to fixed256H36 as one exact trajectory.',
        'weights':'Full learned Wo/gate/up/down andtwoRMS gamma arrays read fromcheckpoint andbytehashed. Do not duplicate entirelearnedmatrices; retained checkpoint is authoritative forfuturecoordinatequeries.',
        'source_V':'V at each source is alreadycontextualized; sourceposition contribution is not unique attribution to raw lexicalmeaning.',
        'readout':'All151936vocabulary rows andall2560statecoords inFP64 analysis on64actualnativepostnormstates. Compare nativefirstchosenlogprob/ID; firsttokenmargin not wholeword meaning.',
        'artifact_only_until2691complete':True,
        'storage':{'free_before':shutil.disk_usage(RESULT).free,'phase_upper':2*1024**3,'future_confirmation_and_crossmodel_reserve':7*1024**3,'floor':FLOOR}}
    path=OUT/'protocol/frozen.json'
    if path.exists():
        old=read(path)
        for k in ('source_material_sha256','source_manifest_sha256','checkpoint_index_sha256','code_sha256','accounting_helper_sha256'):assert old[k]==c[k],k
        return rows,old
    assert c['storage']['free_before']>sum(c['storage'][k] for k in ('phase_upper','future_confirmation_and_crossmodel_reserve','floor'))
    save(path,c);return rows,c


def load_layer(l):
    index=read(MODEL/'model.safetensors.index.json')['weight_map'];arrays={};metadata={}
    keys={'Wo':f'model.layers.{l}.self_attn.o_proj.weight','gate':f'model.layers.{l}.mlp.gate_proj.weight',
        'up':f'model.layers.{l}.mlp.up_proj.weight','down':f'model.layers.{l}.mlp.down_proj.weight',
        'input_gamma':f'model.layers.{l}.input_layernorm.weight','mlp_gamma':f'model.layers.{l}.post_attention_layernorm.weight'}
    for short,key in keys.items():
        path=MODEL/index[key];before=(path.stat().st_size,path.stat().st_mtime_ns)
        with safe_open(str(path),framework='pt',device='cpu') as f:
            tensor=f.get_tensor(key);assert tensor.dtype==torch.bfloat16
            native=tensor.contiguous().view(torch.uint16).numpy().copy();arrays[short]=tensor.double().numpy().copy()
        assert before==(path.stat().st_size,path.stat().st_mtime_ns)
        metadata[short]={'checkpoint_key':key,'shard':str(path),'shape':list(native.shape),'dtype':'nativeBF16',
            'native_bytes_sha256':hashlib.sha256(native.tobytes()).hexdigest(),'shard_size_mtime':before}
    assert arrays['Wo'].shape==(2560,4096) and arrays['gate'].shape==arrays['up'].shape==(9728,2560) and arrays['down'].shape==(2560,9728)
    save(OUT/f'weights/L{l}_checkpoint_manifest.json',metadata)
    np.savez_compressed(OUT/f'weights/L{l}_norm_weights.npz',input_gamma=arrays['input_gamma'],mlp_gamma=arrays['mlp_gamma'])
    return arrays,metadata


def load_case(row,l):
    with np.load(FIELD/f'source/case_{row["case_index"]:04d}.npz') as z:
        source={}
        for key in z.files:
            if not key.startswith(f'L{l}__'):continue
            value=z[key];source[key.split('__',1)[1]]=unbits(value).astype(np.float64) if value.dtype==np.uint16 else value.astype(np.float64)
    positions=[row['body_end_token'],row['task_end_token']]
    with np.load(FIELD/f'field/case_{row["case_index"]:04d}.npz') as z:
        before=unbits(z['full__h'][l]).astype(np.float64);after=unbits(z['full__h'][l+1,positions]).astype(np.float64)
    assert np.array_equal(before[positions],source['residual_before_attention'])
    return source,before,after


def one_case(row,l,weights):
    source,h_all,h_after=load_case(row,l);s=source;eps=float(read(MODEL/'config.json')['rms_norm_eps'])
    att=attention_ledger(s['actual_probability'],s['actual_value'],weights['Wo'],s['attention_output'],s['native_head_concat'])
    norm=conditional_norm_ledger(s['residual_before_attention'],att,weights['mlp_gamma'],eps,s['pre_mlp_norm'],s['mlp_x'])
    pre_input_denom=np.sqrt(np.mean(h_all*h_all,axis=-1,keepdims=True)+eps)
    input_norm_error=s['upstream_attention_x']-h_all*weights['input_gamma']/pre_input_denom
    x=s['mlp_x'];g=s['gate'];u=s['up'];a=s['mlp_a'];m=s['mlp_down']
    gate_ideal=x@weights['gate'].T;up_ideal=x@weights['up'].T
    activation_ideal=g*np.exp(-np.logaddexp(0.,-g))*u
    activation_rounding=a-activation_ideal
    native_a_down64=a@weights['down'].T;ideal_a_down64=activation_ideal@weights['down'].T
    activation_rounding_output=activation_rounding@weights['down'].T
    down_rounding=m-native_a_down64
    residual_add_rounding=s['pre_mlp_norm']-s['residual_before_attention']-s['attention_output']
    final_add_rounding=h_after-s['pre_mlp_norm']-m
    # Matrix multiplication and accumulation order error is separate from model rounding.
    mlp_accounting=native_a_down64-ideal_a_down64-activation_rounding_output
    h_account=s['residual_before_attention']+att['source_terms'].sum(1)+att['av_rounding_output']+att['wo_rounding_output']+att['accounting_order_residual']+residual_add_rounding+ideal_a_down64+activation_rounding_output+down_rounding+mlp_accounting+final_add_rounding
    source_projection={}
    for kind in ('gate','up'):
        projected=np.einsum('qsk,jk->qsj',norm['source_x'],weights[kind],optimize=True)
        signed=projected.sum(1);absolute=np.abs(projected).sum(1)
        branches={name:value@weights[kind].T for name,value in norm['branches_x'].items()}
        native=g if kind=='gate' else u;ideal=x@weights[kind].T
        norm_accounting=norm['reconstruction_error']@weights[kind].T
        projection_rounding=native-ideal
        reconstructed=signed+sum(branches.values())+norm_accounting+projection_rounding
        source_projection[kind]={'signed':signed,'absolute':absolute,'branches':branches,'norm_accounting':norm_accounting,
            'projection_rounding':projection_rounding,'reconstruction_error':native-reconstructed}
        del projected
    arrays={'source_attention':att['source_terms'],'head_attention':att['head_terms'],'source_mlp_x':norm['source_x'],
        'attention_input_norm_rounding':input_norm_error,'input_norm_denominator':pre_input_denom[:,0],
        'mlp_observed_norm_denominator':norm['observed_denominator64'],'H_before':s['residual_before_attention'],'H_after':h_after,
        'native_attention_output':s['attention_output'],'native_pre_mlp_norm':s['pre_mlp_norm'],'native_mlp_x':x,
        'native_gate':g,'native_up':u,'native_a':a,'native_down':m,'gate_linear_rounding':g-gate_ideal,'up_linear_rounding':u-up_ideal,
        'activation_rounding':activation_rounding,'activation_rounding_output':activation_rounding_output,'down_rounding':down_rounding,
        'residual_add_rounding':residual_add_rounding,'final_add_rounding':final_add_rounding,'mlp_float64_accounting':mlp_accounting,
        'ideal64_activation_down':ideal_a_down64,'H_accounting_error':h_after-h_account}
    for key in ('av_rounding_output','wo_rounding_output','accounting_order_residual','reconstruction_error'):arrays['attention__'+key]=att[key]
    for name,value in norm['branches_x'].items():arrays['norm_branch__'+name]=value
    for kind,data in source_projection.items():
        for name,value in data.items():
            if name=='branches':
                for branch,vec in value.items():arrays[f'{kind}_branch__{branch}']=vec
            else:arrays[f'{kind}_source__{name}']=value
    assert all(np.isfinite(v).all() for v in arrays.values())
    errors={'attention_accounting':float(np.abs(att['reconstruction_error']).max()),'norm_accounting':float(np.abs(norm['reconstruction_error']).max()),
        'H_accounting':float(np.abs(arrays['H_accounting_error']).max()),
        **{kind+'_source_projection_accounting':float(np.abs(source_projection[kind]['reconstruction_error']).max()) for kind in ('gate','up')}}
    assert max(errors.values())<1e-8,errors
    summary={'case_index':row['case_index'],'case_id':row['case_id'],'layer':l,'actual_tokens':len(row['prompt_ids']),
        'accounting_errors':errors,'native_rounding_max_abs':{k:float(np.abs(arrays[k]).max()) for k in ('attention_input_norm_rounding','gate_linear_rounding','up_linear_rounding','activation_rounding','down_rounding','residual_add_rounding','final_add_rounding')},
        'source_projection_cancellation':{kind:float((data['absolute']-np.abs(data['signed'])).sum()) for kind,data in source_projection.items()}}
    return arrays,summary


def build_linked(rows):
    summaries=[];manifest=[]
    for l in LAYERS:
        weights,_=load_layer(l)
        for row in rows:
            name=f'L{l}_case_{row["case_index"]:04d}';path=OUT/f'field/{name}.npz';report=OUT/f'analysis/{name}.json'
            if path.exists() and report.exists():
                r=read(report);assert sha(path)==r['artifact_sha256'];summaries.append(r)
            else:
                assert shutil.disk_usage(OUT).free>FLOOR
                arrays,r=one_case(row,l,weights);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
                r['artifact_sha256']=sha(path);save(report,r);summaries.append(r)
            manifest.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'layer':l,'case_index':row['case_index']})
        print('2692 CPU LINKED LAYER',l,'16cases',flush=True);del weights;gc.collect()
    assert len(summaries)==128;save(OUT/'analysis/linked_summary.json',summaries);save(OUT/'analysis/linked_manifest.json',manifest)


def natural_readout():
    rows=[r for r in read(MATERIAL) if r['published']];records={r['case_index']:r for r in read(FIELD/'analysis/records.json')};states=[]
    assert len(rows)==64
    for row in rows:
        with np.load(FIELD/f'field/natural_{row["case_index"]:04d}.npz') as z:states.append(unbits(z['natural_final_state']).astype(np.float64))
    states=np.stack(states);assert states.shape==(64,2560)
    index=read(MODEL/'model.safetensors.index.json')['weight_map'];cfg=read(MODEL/'config.json')
    key='lm_head.weight'
    if key not in index:assert cfg['tie_word_embeddings'];key='model.embed_tokens.weight'
    logits=np.empty((64,cfg['vocab_size']),np.float64);requested={i for r in rows for ids in r['canonical_answer_ids'] for i in ids[:1]}
    vectors={};path=MODEL/index[key];before=(path.stat().st_size,path.stat().st_mtime_ns)
    with safe_open(str(path),framework='pt',device='cpu') as f:
        w=f.get_slice(key);assert w.get_shape()==[151936,2560]
        for start in range(0,151936,4096):
            end=min(start+4096,151936);block=w[start:end,:].double().numpy();logits[:,start:end]=states@block.T
            for token in requested:
                if start<=token<end:vectors[token]=block[token-start].copy()
    assert before==(path.stat().st_size,path.stat().st_mtime_ns)
    maximum=logits.max(-1,keepdims=True);normalizer=maximum+np.log(np.exp(logits-maximum).sum(-1,keepdims=True));lp=logits-normalizer
    assert np.isfinite(lp).all();terms=[];report=[]
    for i,row in enumerate(rows):
        ids=[a[0] for a in row['canonical_answer_ids']];assert len(ids)==2
        contribution=(vectors[ids[0]]-vectors[ids[1]])*states[i];terms.append(contribution)
        assert abs(contribution.sum()-(logits[i,ids[0]]-logits[i,ids[1]]))<1e-8
        native=records[row['case_index']];chosen=native['native_id'];assert chosen==native['generated_ids'][0]
        report.append({'case_id':row['case_id'],'case_index':row['case_index'],'first_token_ids':ids,'first_token_options_distinct':ids[0]!=ids[1],
            'native_first_id':chosen,'ideal64_first_id':int(logits[i].argmax()),'native_chosen_logprob':native['generated_token_logprobs'][0],
            'ideal64_chosen_logprob':float(lp[i,chosen]),'native_minus_ideal64_chosen_logprob':float(native['generated_token_logprobs'][0]-lp[i,chosen]),
            'analysis_coordinate_accounting_error':float(contribution.sum()-(logits[i,ids[0]]-logits[i,ids[1]]))})
    np.savez_compressed(OUT/'field/natural_full_vocabulary_readout.npz',ideal64_logits=logits,ideal64_logprobs=lp,
        actual_native_postnorm_states=states,first_token_contrast_coordinate_terms=np.stack(terms))
    save(OUT/'analysis/natural_readout.json',{'records':report,'actual_native_states':64,'all_vocabulary_rows':151936,'all_state_coordinates':2560,
        'CPU_only':True,'not_native_model_FP64':True,'no_connection_to_fixed256H36_claimed':True,'checkpoint_key':key,'checkpoint_shard_size_mtime':before,
        'artifact_sha256':sha(OUT/'field/natural_full_vocabulary_readout.npz')})


def build(rows,c):
    build_linked(rows);natural_readout()
    assert not torch.cuda.is_initialized()
    save(OUT/'analysis/prepared.json',{'all_artifact_checks_passed':True,'phase_completed':False,'checkpoint_index_unchanged':sha(MODEL/'model.safetensors.index.json')==c['checkpoint_index_sha256'],
        'source_material_unchanged':sha(MATERIAL)==c['source_material_sha256'],'CUDA_initialized':False,
        'waiting_required_before_phase_finalize':'2691actualcomplete/MEMO and full2689/2690/crossmodel scientific review; prepared arithmetic is not semantic closure.'})


def finalize(c):
    assert read(RESULT/'phase2691_crossmodel_role_confirmation/analysis/final.json')['all_checks_passed']
    prepared=read(OUT/'analysis/prepared.json');assert prepared['all_artifact_checks_passed']
    summaries=read(OUT/'analysis/linked_summary.json');readout=read(OUT/'analysis/natural_readout.json')
    for r in read(OUT/'analysis/linked_manifest.json'):assert sha(r['path'])==r['sha256']
    scalar=read(RESULT/'phase2689_native_qkv_scalar/analysis/final.json');fresh=read(RESULT/'phase2690_fresh_role_qkv_confirmation/analysis/final.json')
    cross=read(RESULT/'phase2691_crossmodel_role_confirmation/analysis/final.json')
    review={'scalar_summary':scalar['summary'],'fresh_behavior':fresh['summary']['behavior'],
        'Q14_old_candidates':cross['summary']['models']['qwen14']['Q14_old_candidates'],
        'limits':['Known arithmetic accounting errors are constructed numerical ledgers, not semanticmechanism closure.',
            'Observed norm denominator conditional allocation is not ablation; actualscalar interventions remain separate evidence.',
            'No percasechangedrawP is present in2689. Its JSON metrics/globalcoordinateaggregates must not be displayed as percasechangedrawP.',
            'Fixedfield/native naturalstate difference measured8192/8192; do notdraw a false exact causaltrajectory connecting them.',
            'Allpartialfamilybackgrounds remain evenwhenuniversalcoordinate-sign gates fail. Learned mathematical languagecode notsolved.']}
    save(OUT/'analysis/full_campaign_review.json',review)
    errors={k:max(r['accounting_errors'][k] for r in summaries) for k in summaries[0]['accounting_errors']}
    summary={'linked_native_layer_cases':128,'source_examples':16,'layers':LAYERS,'all_MLP_units':9728,'all_residual_coordinates':2560,
        'accounting_max_errors':errors,'natural_readout_cases':64,'all_vocabulary_rows':151936,
        'native_vs_FP64_readout_top1_equal':sum(r['native_first_id']==r['ideal64_first_id'] for r in readout['records']),
        'native_vs_FP64_readout_max_logprob_gap':max(abs(r['native_minus_ideal64_chosen_logprob']) for r in readout['records']),
        'same_goal_next':True,'known_arithmetic_not_semanticclosure':True}
    checks={'128native_linked_cases':len(summaries)==128,'64actualnaturalstates_fullvocab':len(readout['records'])==64,
        'allsource_and_allMLPcoords':True,'prior2691actuallycomplete':True,'noCUDAmodelinitialized':not torch.cuda.is_initialized(),
        'immutablematerial':sha(MATERIAL)==c['source_material_sha256']}
    assert all(checks.values())
    finish(2692,'全来源Attention到完整MLP参数的舍入账本及独立自然读出复审',OUT,
        {'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '在16预定例八层读取真实完整Wo/gate/up/down矩阵，保留全部来源与物理坐标并分列原生舍入余项；自然输出读出独立使用真实保存的postnorm状态与完整词表权重，绝不把不同执行协议拼成一条轨迹。',
        r'H_{l+1}=H_l+A_l+M_l+\epsilon_{add};\quad A_l=\sum_{h,s}P_{h,s}W_O^hV_s+\epsilon_{AV}+\epsilon_{Wo};\quad M_l=W_d(\mathrm{SiLU}(g)u+\epsilon_a)+\epsilon_d;\quad z_y=\sum_jU_{y,j}h^{natural,postnorm}_j.',
        'C00116源例×8层全部source/head/128head维与2560输出坐标；C002真实完整gate/up/down全部9728单元；C003内生观测RMS条件分账、乘法/投影/残差加法余项；C00464真实自然末状态全部151936词表FP64读出分析与原生首token对照；C005完整2689标量、2690新材料、2691跨模型结果复审；C006所有原始路径/哈希/坐标保留。',
        '参数共享与上下文依赖可在固定原坐标中直接核查；同一输入项如何经来源竞争、归一化和MLP乘积进入下一层现在有可查询的计算分账。其角色仍是累积语言条件拼图的测量工具。',
        '这些是已知运算的带余项记账，不是发现通用语义齿轮。来源V已包含上下文，不能唯一归因原始词。固定观测分母不表示删除作用。FP64只是分析读出；自然postnorm与固定场H36是不同数值协议。首tokenmargin不代表完整名字/整句，字段缺失时不得伪造单例干预热力图。',
        '继续2693重要全坐标/真实参数客户端类型、数值与真实浏览器验收、保护引用的数据清理与全阶段终审；之后仍按同目标自动选择更有辨识力的语言条件研究，而不因局部必要性或全局同号门失败停止。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','build','finalize'));args=parser.parse_args()
    rows,c=prepare()
    if args.action=='build':build(rows,c)
    elif args.action=='finalize':finalize(c)
