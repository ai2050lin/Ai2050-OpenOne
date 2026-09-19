"""Actual evidence review, checkpoint QKV windows and native observation preflight."""
import gc,inspect,os,shutil,time
import numpy as np
import torch
from safetensors import safe_open
from phase2620_native_coordinate_contract import *
from phase2685_native_attention_math import self_test,rms,rope,routing
from phase2685_native_qkv_capture import NativeQKVCapture
from phase2677_padded_native_runtime import padded_inputs
from phase2679_native_source_ledger import exact_bits,real_token_data,pack_case
from phase2662_symmetric_mapping_contract import load_native

OUT=RESULT/'phase2685_native_attention_contract'
PRIOR=RESULT/'phase2684_source_campaign_delivery'
FRESH=RESULT/'phase2681_fresh_source_confirmation'
LAYERS=(0,5,17,23,26,27,28,35)
MODEL=ROOT/'models/hf/qwen3-4b'

def checkpoint_windows():
    index_path=MODEL/'model.safetensors.index.json';index=read(index_path)['weight_map']
    arrays={};controls=[];metadata={};touched={}
    for l in LAYERS:
        for kind,heads in (('q',32),('k',8),('v',8)):
            key=f'model.layers.{l}.self_attn.{kind}_proj.weight';path=MODEL/index[key]
            before=(path.stat().st_size,path.stat().st_mtime_ns);row=(l%heads)*128+(l*7)%128
            with safe_open(str(path),framework='pt',device='cpu') as f:
                sl=f.get_slice(key);assert sl.get_shape()==[heads*128,2560]
                value=sl[row,:].float().numpy().copy()
                # All head windows at two predetermined head coordinates, each
                # with all2560 incoming learned scalars, not activation TopK.
                for h in range(heads):
                    for d in (0,127):arrays[f'L{l}_{kind}_head{h}_d{d}']=exact_bits(sl[h*128+d,:].double().numpy())
            assert before==(path.stat().st_size,path.stat().st_mtime_ns)
            touched[str(path)]={'bytes':before[0],'mtime_ns':before[1]}
            stem=f'L{l}_{kind}_row{row}';arrays[stem]=exact_bits(value.astype(np.float64))
            absvalue=np.abs(value.astype(np.float64));rmsvalue=float(np.sqrt(np.mean(absvalue**2)))
            ordinary=int(np.argmin(np.abs(absvalue-rmsvalue)))
            low_candidates=np.flatnonzero((absvalue>0)&(absvalue<=rmsvalue/4));assert len(low_candidates)
            low=int(low_candidates[0])
            for label,k in (('ordinary',ordinary),('low',low)):
                controls.append({'layer':l,'kind':kind,'output_row':row,'head':row//128,'head_coordinate':row%128,
                    'input_coordinate':k,'control':label,'original_weight':float(value[k]),'row_RMS':rmsvalue,'vector':stem})
            metadata[stem]={'checkpoint_key':key,'row':row,'shape':[2560],'vector_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
        for kind in ('q','k'):
            key=f'model.layers.{l}.self_attn.{kind}_norm.weight'
            with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:arrays[f'L{l}_{kind}_gamma']=exact_bits(f.get_tensor(key).double().numpy())
    path=OUT/'weights/native_qkv_windows.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as z:assert set(z.files)==set(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items())
    else:np.savez(path,**arrays)
    return {'controls':controls,'vectors':metadata,'complete_vector_arrays':len(arrays),'index_sha256':sha(index_path),
        'artifact_sha256':sha(path),'checkpoint_shard_metadata':touched,'selection':'8predeclaredlayers; primary output row=(layer%heads)*128+(7*layer)%128. Ordinary closestabsolute to fullrowRMS, lowfirstnonzero<=rowRMS/4. No activation/behavior selection. Allhead fixed d0/d127 fullincomingrows also retained.',
        'scope':'Native BF16 exact uint16bits. Indexhash/shardsize/mtime checks, not a claim of hashing all model shard contents.'}

def prepare():
    assert read(PRIOR/'analysis/terminal_post_final.json')['all_checks_passed']
    final=read(PRIOR/'analysis/final.json');assert final['phase']==2684 and final['all_checks_passed']
    rows=[r for r in read(FRESH/'material/cases.json') if r['published']];assert len(rows)==64
    path=OUT/'material/preflight_cases.json'
    if path.exists():assert read(path)==rows
    else:save(path,rows)
    math=self_test();save(OUT/'analysis/synthetic_arithmetic_checks.json',math)
    weights=checkpoint_windows();save(OUT/'protocol/native_weights.json',weights)
    free=shutil.disk_usage(OUT).free;assert free>8*1024**3+1024**3
    model_source=ROOT/'.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py'
    plan={'same_goal':True,'next_campaign':read(PRIOR/'analysis/next_campaign.json'),
        'layers':LAYERS,'preflight_cases':64,'preflight_material_sha256':sha(path),'weights_sha256':weights['artifact_sha256'],
        'capturer_sha256':sha(Path(__file__).with_name('phase2685_native_qkv_capture.py')),
        'math_sha256':sha(Path(__file__).with_name('phase2685_native_attention_math.py')),'local_model_implementation_sha256':sha(model_source),
        'local_formula':'Single scalar changes its projection row at EVERY actual token. Recompute full128-dim head RMS denominator, fullRoPE, allsource maskedsoftmax and allhead products. No donor sample or frozen denominator.',
        'preflight':'64existingpublishedfreshprefixes across8fam2lang4functions. Same160shape baseline/observed/restored native BF16 allfinalstates must be bitidentical. New instrumentation only, not independent new language confirmation.',
        'fields':'Native allactualtoken linearQ/K/V, normalizedQ/K, positioncos/sin and postRoPEallQ; existing allK/V/twoqueryallheadsP/MLP fields retained. AllH/MLP background in nextlargephase. This capturer is Qwen3-specific, DS/GLM require explicit adapters.',
        'next_material_change':'At least4096 newconditions; prefer8192 by explicitly separating name-slot assignment from semantic relation target, content, twoforms/twoorders and4outputs. Freshnames/relationshipinstances independent of2681; repeated templates named honestly. Freeze material and scoring beforeformalforwards.',
        'score':'Preserve strict and normalizedwhole-string separately; explicit-last-line parser frozen before newformaloutputs, unparsed/ambiguous/empty/no_boundary retained. Matching output not proof of semanticreasoning. DS native and explicitanswer protocols remain distinct.',
        'important_confirmation':'Qwen14 all5previousglobalgateaddresses and ALL background frozen; >=4096independentconditions, slot/order/rolecounterbalanced, not only5coordinates.',
        'not_closure':'Knownnorm/rotation/softmax arithmetic is a measurement model, not language-specificmechanism. Noop identity is a capture control, not semantic invariance.',
        'storage':{'free_before':free,'floor':8*1024**3,'preflight_raw':'Only2predeclaredchronologyEN truth/name rawpacks; every64case allcoords processed, errors persisted. No otherpreflight rawfieldswritten.'}}
    target=OUT/'protocol/frozen.json'
    if target.exists():
        old=read(target)
        for k in ('preflight_material_sha256','weights_sha256','capturer_sha256','math_sha256','local_model_implementation_sha256'):assert old[k]==plan[k],k
        plan=old
    else:save(target,plan)
    return rows,plan

@torch.inference_mode()
def main():
    assert not (OUT/'analysis/final.json').exists()
    rows,plan=prepare();model,tok=load_native('qwen4');records=[];t0=time.monotonic()
    save(OUT/'protocol/runtime.json',{'dtype':str(model.dtype),'device_map':getattr(model,'hf_device_map',None),
        'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':bool(getattr(model,'is_quantized',False)),'fixed_length':160})
    assert str(model.dtype)=='torch.bfloat16' and not getattr(model,'is_quantized',False)
    for i,r in enumerate(rows):
        inputs=padded_inputs(model,r['prompt_ids'],tok.eos_token_id)
        baseline=model.model(**inputs).last_hidden_state.detach().cpu()
        with NativeQKVCapture(model,LAYERS) as cap:
            cap.reset(r['body_end_token'],r['task_end_token']);cap.enabled=True
            observed=model.model(**inputs).last_hidden_state.detach().cpu();cap.enabled=False
            source=real_token_data(cap.pack(),len(r['prompt_ids']));up=cap.upstream_pack()
        restored=model.model(**inputs).last_hidden_state.detach().cpu()
        assert torch.equal(baseline,observed) and torch.equal(baseline,restored)
        errors={}
        for l in LAYERS:
            a=up[l];s=source[l];att=model.model.layers[l].self_attn;n=len(r['prompt_ids'])
            q=a['linear_q'].reshape(n,32,128);k=a['linear_k'].reshape(n,8,128);v=a['linear_v'].reshape(n,8,128)
            assert np.array_equal(v,s['actual_value']) and np.array_equal(a['query_post_rope_full'][[r['body_end_token'],r['task_end_token']]],s['actual_query_post_rope'])
            qg=att.q_norm.weight.detach().double().cpu().numpy();kg=att.k_norm.weight.detach().double().cpu().numpy();eps=att.q_norm.variance_epsilon
            qp=rms(q,qg,eps);kp=rms(k,kg,eps)
            qr=rope(a['normalized_q'],a['rope_cos'],a['rope_sin']);kr=rope(a['normalized_k'],a['rope_cos'],a['rope_sin'])
            mask=s['actual_mask'];assert mask.shape==(1,1,2,n)
            prob,_=routing(a['query_post_rope_full'],s['actual_key_post_rope'],v,mask[0,0][:,None,:],s['scaling'],(r['body_end_token'],r['task_end_token']))
            errors[str(l)]={'q_norm_max_abs64_vs_native':float(np.abs(qp-a['normalized_q']).max()),'k_norm_max_abs64_vs_native':float(np.abs(kp-a['normalized_k']).max()),
                'q_rope_max_abs64_vs_native':float(np.abs(qr-a['query_post_rope_full']).max()),'k_rope_max_abs64_vs_native':float(np.abs(kr-s['actual_key_post_rope']).max()),
                'softmax_max_abs64_vs_native':float(np.abs(prob-s['actual_probability']).max())}
            assert all(np.isfinite(x) for x in errors[str(l)].values())
        published=r['family']=='chronology' and r['language']=='en' and r['output_function'] in ('truth','name')
        if published:
            arrays=pack_case(source)
            for l,a in up.items():
                for k,v in a.items():arrays[f'L{l}__upstream_{k}']=exact_bits(v)
            field=OUT/f'field/case_{i:04d}.npz';field.parent.mkdir(parents=True,exist_ok=True);np.savez(field,**arrays)
        records.append({'case_id':r['case_id'],'source_case_index':r['case_index'],'tokens':len(r['prompt_ids']),'noop_exact':True,'restored_exact':True,'errors':errors,'published':published})
        save(OUT/'analysis/preflight_records.json',records)
        save(OUT/'analysis/progress.json',{'cases':i+1,'total':64,'elapsed_seconds':time.monotonic()-t0})
        if (i+1)%8==0:print('2685 NATIVE PREFLIGHT',i+1,64,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    maximum={key:max(r['errors'][str(l)][key] for r in records for l in LAYERS) for key in records[0]['errors']['0']}
    checks={'prior2684fullydelivered':True,'64_native_capture_noops':len(records)==64 and all(r['noop_exact'] and r['restored_exact'] for r in records),
        '8layers_allactualtoken_coordinates':all(len(r['errors'])==8 for r in records),'72_synthetic_scalar_cases':self_test()['synthetic_cases']==72,
        '48_actual_scalar_controls':len(read(OUT/'protocol/native_weights.json')['controls'])==48,'2_raw_predeclared_examples':sum(r['published'] for r in records)==2,
        'material_frozen':sha(OUT/'material/preflight_cases.json')==plan['preflight_material_sha256']}
    assert all(checks.values())
    finish(2685,'从MLP条件账本向原生Q/K/V单参数路由推进：完整复审与八层测量资格',OUT,
        {'provenance':str(Path(__file__)),'summary':{'native_preflight_cases':64,'layers':LAYERS,'actual_scalar_controls':48,'synthetic_scalar_cases':72,'native_vs_FP64_analysis_max_errors':maximum,'next_campaign':plan['next_campaign']['phases']},'checks':checks},
        '先复核2677–2684完整终审和评分更正，再读取实际检查点Q/K/V完整输入行与head归一化权重。新只读采集器在八层保留所有真实token与所有head坐标；无操作前向核验其不改模型计算。',
        r'\Delta z_{t,r}=\Delta\theta x_{t,k};\quad N(z)={\gamma\odot z\over\sqrt{d^{-1}\sum_i z_i^2+\epsilon}};\quad Q=R_tN(W_qx_t),\ K=R_sN(W_kx_s);\quad P_{ths}=\operatorname{softmax}_s(Q_{th}^{\mathsf T}K_{s,kv(h)}/\sqrt d+M_{ts}).',
        'C001完整前阶段终审及四协议评分/候选审计；C002八层24完整主输入行、48普通/低值真实标量及所有head的d0/d127完整输入行、16headnorm向量；C00372固定合成算术例双方向Q/K/V标量变化与全来源归一化；C00464旧展示前缀八族双语四输出功能的原生BF16 baseline/采集/restored精确比较；C005全部真实token的Q/K归一化、RoPE和全来源softmax分析误差，保存2预定原场。',
        '单个学习标量同时参与多个token和来源的运算，head分母与softmax竞争内生耦合。由真实坐标提取局部可计算条件路径是后续语言操作图谱的工具，不需要donor搬运。Q14五候选仍是待扩大确认的条件纹理，不把正向Wdown与反向H强行串成闭环。',
        '72例为合成数值测试，64例重用旧材料仅验证新测量器，均不是独立语言确认；此采集器限Qwen3架构。FP64是分析侧，不是模型精度；RMS/旋转/softmax舍入余项如实列出。取定参数窗口不是完整模型参数机制；后续仍计算全部H/MLP背景。旧整串匹配不是语义正确率，事后末行解析不是盲测。最初预检两次在0条完整记录时退出：可选hf_device_map属性读取假设错误，以及分析侧mask维度广播错误；只修正诊断读取与无损轴布局，模型/材料/捕获器/数学算法未改，详见preflight_incident.json。',
        '继续2686大规模新材料与独立槽位/关系/表面操作合同，2687–2693全场、真实QKV标量、扩大确认、顺序跨模型及实际交付。目标相同自动执行，不以合成恒等式或单坐标全局门失败作为完成/终止依据。')

if __name__=='__main__':main()
