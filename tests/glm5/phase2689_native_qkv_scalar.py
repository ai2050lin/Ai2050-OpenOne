"""Actual single native BF16 Q/K/V scalars; endogenous normalization/routing.

No donor, no frozen denominator, no weight checkpoint writes. FP64 predictions
are local analysis only. Full downstream probability/sequence effects are
measured separately, not claimed to be predicted by the local formula.
"""
import gc,itertools,os,shutil,time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2662_symmetric_mapping_contract import load_native
from phase2677_padded_native_runtime import padded_inputs
from phase2685_native_qkv_capture import NativeQKVCapture
from phase2685_native_attention_math import local_scalar_path
from phase2679_native_source_ledger import real_token_data
from phase2687_role_qkv_field import natural

OUT=RESULT/'phase2689_native_qkv_scalar'
FIELD=RESULT/'phase2687_role_qkv_field'
ATT=RESULT/'phase2685_native_attention_contract'
MATERIAL=RESULT/'phase2686_independent_role_contract/material/initial.json'
DOSES=(.025,.1)
SIGNS=(-1,1)
LAYERS=(0,5,17,23,26,27,28,35)
FLOOR=8*1024**3


def prepare():
    assert read(RESULT/'phase2688_native_qkv_terms/analysis/final.json')['all_checks_passed']
    rows=[r for r in read(MATERIAL) if r['unit']==r['form']==r['roster_order']==r['mention_order']==0 and r['output_function'] in ('truth','name')]
    assert len(rows)==128 and sum(r['parameter_published'] for r in rows)==16
    controls=read(ATT/'protocol/native_weights.json')['controls'];assert len(controls)==48
    c={'material_sha256':sha(MATERIAL),'case_ids':[r['case_id'] for r in rows],'controls':controls,'doses':DOSES,'signs':SIGNS,
        'total_real_parameter_conditions':128*48*4,'native_dtype':'BF16 nonquantized','field_shape':256,
        'dose':'Requested delta=sign*dose*RMS(frozen complete native weight row). Set one BF16 scalar and report effective rounded delta. Zero effective changes remain, no minimum-effect filtering.',
        'local':'All actualtokens Q/K/V full128headRMS,RoPE,allsource softmax re-evaluated from observed native linear baseline plus effective_delta*x. Changed weight shared at everytoken, never one source only. Allsource/headcoords retained in response maps.',
        'downstream':'Full151936 next-token probability L1/max and all2560 finalstate measured. Frozen baseline-generated token sequence scored under fixed256 causalteacherforcing before/after; not naturalcache likelihood identity.',
        'natural':'All128 baseline natural records retained from2687;16predeclaredpublishedtruth prefixes additionally generate full32token naturalcache output under everyactualscalarcondition. Report generatedIDs/EOS/logprobs and fixedparser separately. No semanticclosure gate.',
        'resume':'One atomic NPZ cumulative-map checkpoint stores completed prefix count; caseJSON written before checkpoint. If interrupted, replay last incomplete prefix and require exact durable record identity. Only this generated checkpoint is replaced; no raw history cleanup.',
        'noops':'Each128 baseline/captured/restored finalstate exact; selectedlayer input x unchanged under scalar edits. All24 touched matrices full BF16 bytehash equal before/after and each scalar restored in finally.',
        'budget':{'phase_reserve':1024**3,'floor':FLOOR,'free_before':shutil.disk_usage(RESULT).free},
        'code_sha256':{name:sha(TESTS/name) for name in ('phase2689_native_qkv_scalar.py','phase2685_native_attention_math.py','phase2685_native_qkv_capture.py','phase2687_role_qkv_field.py')}}
    p=OUT/'protocol/frozen.json'
    if p.exists():
        old=read(p)
        for k in ('material_sha256','case_ids','controls','code_sha256'):assert old[k]==c[k],k
        c=old
    else:
        assert c['budget']['free_before']>FLOOR+1024**3;save(p,c)
    return rows,controls,c


def matrix_hashes(model):
    return {f'L{l}_{kind}':hashlib.sha256(getattr(model.model.layers[l].self_attn,kind+'_proj').weight.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            for l in LAYERS for kind in ('q','k','v')}


def local_arguments(model,l,source,up,row):
    n=len(row['prompt_ids']);att=model.model.layers[l].self_attn
    mask=source['actual_mask'];assert mask.shape==(1,1,2,n)
    return {'x':up['attention_x'],'q_linear':up['linear_q'].reshape(n,32,128),'k_linear':up['linear_k'].reshape(n,8,128),
        'v_linear':up['linear_v'].reshape(n,8,128),'q_gamma':att.q_norm.weight.detach().double().cpu().numpy(),
        'k_gamma':att.k_norm.weight.detach().double().cpu().numpy(),'epsilon':att.q_norm.variance_epsilon,
        'cos':up['rope_cos'],'sin':up['rope_sin'],'mask':mask[0,0][:,None,:],'scale':source['scaling'],
        'positions':(row['body_end_token'],row['task_end_token'])}


@torch.inference_mode()
def fixed_sequence_scores(model,tok,row,generated_ids):
    ids=list(row['prompt_ids']);tokens=list(generated_ids);assert 0<len(tokens)<=32
    allids=ids+tokens[:-1];assert len(allids)<=256
    out=model.model(**padded_inputs(model,allids,tok.eos_token_id,total=256))
    states=out.last_hidden_state[0,len(ids)-1:len(ids)-1+len(tokens)]
    logits=model.lm_head(states).float();lp=torch.log_softmax(logits,dim=-1)
    selected=lp[torch.arange(len(tokens),device=lp.device),torch.tensor(tokens,device=lp.device)]
    return selected.double().cpu().numpy()


def map_empty():
    arrays={}
    for key,shape in [('probability',(48,4,2,32,256)),('head_output',(48,4,2,32,128)),('final_state',(48,4,2560))]:
        for family in ('native','ideal') if key!='final_state' else ('native',):
            for metric in ('sum','sumabs'):arrays[f'{key}__{family}__{metric}']=np.zeros(shape,np.float64)
    arrays['valid_source_count']=np.zeros((256,),np.uint16)
    arrays['completed_prefixes']=np.array(0,np.int64)
    return arrays


def add_response(maps,ci,di,key,value,family,n):
    v=np.asarray(value,np.float64)
    if key=='probability':
        assert v.shape==(2,32,n);maps[f'{key}__{family}__sum'][ci,di,:,:,:n]+=v;maps[f'{key}__{family}__sumabs'][ci,di,:,:,:n]+=np.abs(v)
    else:
        maps[f'{key}__{family}__sum'][ci,di]+=v;maps[f'{key}__{family}__sumabs'][ci,di]+=np.abs(v)


def delta_report(actual,predicted):
    return {'actual_L1':float(np.abs(actual).sum()),'ideal_L1':float(np.abs(predicted).sum()),
        'prediction_error_L1':float(np.abs(actual-predicted).sum()),'prediction_error_max_abs':float(np.abs(actual-predicted).max()),
        'actual_changed_coordinates':int(np.count_nonzero(actual)),'coordinates':actual.size}


def checkpoint(maps):
    path=OUT/'maps/cumulative.npz';path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('cumulative.pending.npz');np.savez_compressed(temp,**maps)
    os.replace(temp,path)


@torch.inference_mode()
def collect(model,tok,rows,controls,c):
    path=OUT/'maps/cumulative.npz'
    if path.exists():
        with np.load(path) as z:maps={k:z[k].copy() for k in z.files}
    else:maps=map_empty()
    begin=int(maps['completed_prefixes']);assert 0<=begin<=128
    prior_hashes=matrix_hashes(model);saved=OUT/'protocol/matrix_hashes_before.json'
    if saved.exists():assert read(saved)==prior_hashes
    else:save(saved,prior_hashes)
    originals={r['case_index']:r for r in read(FIELD/'analysis/records.json')};t0=time.monotonic()
    for ii in range(begin,128):
        if shutil.disk_usage(OUT).free<FLOOR:raise RuntimeError('8GiB floor; preserve state and do not clean raw midcampaign')
        row=rows[ii];n=len(row['prompt_ids']);inp=padded_inputs(model,row['prompt_ids'],tok.eos_token_id,total=256)
        baseline=model.model(**inp).last_hidden_state.detach().cpu()
        with NativeQKVCapture(model,LAYERS) as cap:
            cap.reset(row['body_end_token'],row['task_end_token']);cap.enabled=True
            output=model.model(**inp);cap.enabled=False
            assert torch.equal(baseline,output.last_hidden_state.detach().cpu())
            ss=real_token_data(cap.pack(),n);up=cap.upstream_pack()
        state=output.last_hidden_state[0,row['task_end_token']].detach().clone();base_logp=torch.log_softmax(model.lm_head(state).float(),dim=-1)
        base_probability=base_logp.exp().double().cpu().numpy();base_state=state.double().cpu().numpy()
        base_sequence=originals[row['case_index']]['generated_ids'];base_scores=fixed_sequence_scores(model,tok,row,base_sequence)
        predictions={l:local_arguments(model,l,ss[l],up[l],row) for l in LAYERS}
        idealbase={l:local_scalar_path(**predictions[l],kind='q',output_row=0,input_coordinate=0,delta=0) for l in LAYERS}
        rec=[];natural_outputs=[]
        for ci,control in enumerate(controls):
            l=control['layer'];kind=control['kind'];rr=control['output_row'];kk=control['input_coordinate']
            weight=getattr(model.model.layers[l].self_attn,kind+'_proj').weight;original=weight[rr,kk].detach().clone()
            assert float(original)==control['original_weight']
            with NativeQKVCapture(model,(l,)) as cap:
                for di,(dose,sign) in enumerate(itertools.product(DOSES,SIGNS)):
                    requested=sign*dose*control['row_RMS']
                    try:
                        weight[rr,kk].copy_(original.float()+requested);effective=float(weight[rr,kk])-float(original)
                        cap.reset(row['body_end_token'],row['task_end_token']);cap.enabled=True
                        output=model.model(**inp);cap.enabled=False
                        changed_state=output.last_hidden_state[0,row['task_end_token']].detach().clone()
                        changed=real_token_data(cap.pack(),n)[l];changedup=cap.upstream_pack()[l]
                        assert np.array_equal(up[l]['attention_x'],changedup['attention_x']), 'Single parameter changed its upstream input'
                        pred=local_scalar_path(**predictions[l],kind=kind,output_row=rr,input_coordinate=kk,delta=effective)
                        dp=changed['actual_probability']-ss[l]['actual_probability'];ip=pred['probability']-idealbase[l]['probability']
                        dh=changed['native_head_concat'].reshape(2,32,128)-ss[l]['native_head_concat'].reshape(2,32,128)
                        ih=pred['head_output']-idealbase[l]['head_output']
                        if kind=='v':assert not dp.any() and not ip.any(),'V projection must not alter same-layer Q/K routing'
                        changed_logp=torch.log_softmax(model.lm_head(changed_state).float(),dim=-1)
                        changed_probability=changed_logp.exp().double().cpu().numpy();dprob=changed_probability-base_probability
                        ds=changed_state.double().cpu().numpy()-base_state
                        scores=fixed_sequence_scores(model,tok,row,base_sequence)
                        item={'control_index':ci,'dose':dose,'sign':sign,'requested_delta':requested,'effective_delta':effective,
                            'projection_kind':kind,'layer':l,'output_row':rr,'input_coordinate':kk,'control':control['control'],
                            'probability':delta_report(dp,ip),'head_output':delta_report(dh,ih),
                            'all_vocabulary_probability_L1':float(np.abs(dprob).sum()),'all_vocabulary_probability_max_abs':float(np.abs(dprob).max()),
                            'baseline_next_id':int(base_logp.argmax()),'changed_next_id':int(changed_logp.argmax()),'final_state_L1':float(np.abs(ds).sum()),
                            'fixed_baseline_sequence_token_logprob_change':(scores-base_scores).tolist(),'fixed_baseline_sequence_logprob_change':float((scores-base_scores).sum()),
                            'upstream_x_exact':True,'candidate_first_token_logprob_changes':[]}
                        for word in row['common_readout_words']:
                            ids=tok.encode(word,add_special_tokens=False);assert ids
                            item['candidate_first_token_logprob_changes'].append({'word':word,'ids':ids,'change':float(changed_logp[ids[0]]-base_logp[ids[0]])})
                        if row['parameter_published']:
                            generated,_=natural(model,tok,row,changed_state)
                            natural_outputs.append({'control_index':ci,'dose':dose,'sign':sign,**generated})
                        for key,act,ideal in (('probability',dp,ip),('head_output',dh,ih)):
                            add_response(maps,ci,di,key,act,'native',n);add_response(maps,ci,di,key,ideal,'ideal',n)
                        add_response(maps,ci,di,'final_state',ds,'native',n);rec.append(item)
                    finally:
                        cap.enabled=False;weight[rr,kk].copy_(original)
                        assert torch.equal(weight[rr,kk],original)
            if (ci+1)%12==0:
                save(OUT/'analysis/progress.json',{'completed_prefixes':ii,'current_prefix':ii+1,'controls_this_prefix':ci+1,'total_prefixes':128,
                    'elapsed_seconds_this_process':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free})
                print('2689 REALSCALAR PREFIX',ii+1,128,'CONTROL',ci+1,48,flush=True)
        restored=model.model(**inp).last_hidden_state.detach().cpu();assert torch.equal(baseline,restored)
        assert len(rec)==192
        result={'case_id':row['case_id'],'case_index':row['case_index'],'native_noop_exact':True,'restored_exact':True,
            'baseline_generated_ids':base_sequence,'baseline_fixed256_sequence_token_logprobs':base_scores.tolist(),
            'baseline_parallel_vs_single_head_first_logprob_difference':float(base_scores[0]-float(base_logp[base_sequence[0]])),
            'conditions':rec,'natural_changed_generations':natural_outputs}
        recordpath=OUT/f'analysis/case_{ii:03d}.json'
        if recordpath.exists():assert read(recordpath)==result,'Durable scalar-prefix replay changed'
        else:save(recordpath,result)
        maps['valid_source_count'][:n]+=1;maps['completed_prefixes'][...]=ii+1;checkpoint(maps)
    after=matrix_hashes(model);assert after==prior_hashes;save(OUT/'analysis/matrix_hashes_after.json',after)
    return maps


def audit(rows,maps):
    reports=[read(OUT/f'analysis/case_{i:03d}.json') for i in range(128)]
    assert all(r['case_id']==m['case_id'] and r['native_noop_exact'] and r['restored_exact'] and len(r['conditions'])==192 for r,m in zip(reports,rows))
    allrows=[x for r in reports for x in r['conditions']];groups={}
    for x in allrows:
        key=f'{x["projection_kind"]}/{x["control"]}/dose{x["dose"]}'
        g=groups.setdefault(key,{'n':0,'effective_weight_zeros':0,'zero_full_probability_effects':0,'native_P_L1':0.,'ideal_P_error_L1':0.,
            'native_head_L1':0.,'ideal_head_error_L1':0.,'whole_vocabulary_L1':0.,'next_token_changes':0})
        g['n']+=1;g['effective_weight_zeros']+=x['effective_delta']==0;g['zero_full_probability_effects']+=x['all_vocabulary_probability_L1']==0
        g['native_P_L1']+=x['probability']['actual_L1'];g['ideal_P_error_L1']+=x['probability']['prediction_error_L1']
        g['native_head_L1']+=x['head_output']['actual_L1'];g['ideal_head_error_L1']+=x['head_output']['prediction_error_L1']
        g['whole_vocabulary_L1']+=x['all_vocabulary_probability_L1'];g['next_token_changes']+=x['baseline_next_id']!=x['changed_next_id']
    for a in maps.values():assert np.isfinite(a).all()
    for k,a in maps.items():
        if k.endswith('__sum'):assert (np.abs(a)<=maps[k+'abs']+1e-8).all()
    summary={'actual_parameter_conditions':len(allrows),'native_prefixes':128,'selected_complete_rows':24,'scalar_controls':48,
        'all_vocabulary_size':151936,'natural_changed_generations':sum(len(r['natural_changed_generations']) for r in reports),
        'aggregate_by_kind_control_dose':groups,'local_prediction_is_not_downstream_prediction':True}
    checks={'24576_actual_conditions':len(allrows)==24576,'128native_noops_and_restoration':True,'all24matrix_hashes_restored':read(OUT/'protocol/matrix_hashes_before.json')==read(OUT/'analysis/matrix_hashes_after.json'),
        '3072complete_changed_natural_paths':summary['natural_changed_generations']==3072,'complete_fullcoordinate_maps':int(maps['completed_prefixes'])==128,
        'material_unchanged':sha(MATERIAL)==read(OUT/'protocol/frozen.json')['material_sha256']}
    assert all(checks.values());save(OUT/'analysis/scientific_checks.json',{'all_checks_passed':True,'summary':summary,'checks':checks})
    finish(2689,'128前缀×48真实QKV标量：完整内生归一化路由与下游概率分账',OUT,
        {'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '仅编辑一个实际BF16参数并报告其舍入后有效变化，所有token共同使用该参数；解析侧重算全部head归一化分母、RoPE、全部来源softmax和head输出。每个条件后立即恢复。',
        r'\delta_{eff}=\operatorname{BF16}(\theta+s\alpha\operatorname{RMS}(W_r))-\theta;\quad z^\prime_{t,r}=z_{t,r}+\delta_{eff}x_{t,k};\quad \Delta P_{local}=P(N(z^\prime))-P(N(z));\quad \epsilon_\Delta=\Delta P_{native}-\Delta P_{local}.',
        'C001128八族双语双关系填充双目标真值/人名分层前缀；C00248冻结普通/低值QKV标量×±两剂量=24576实际变化；C003每例完整QKV/归一化/来源概率/head坐标及全部词表概率；C004全部条件固定基线生成串逐token概率；C00516预定例每条件完整32token自然输出共3072条；C006128无操作恢复与24完整矩阵字节哈希。',
        '从单个真实学习标量到内生竞争路由的局部可计算路径得到直接有限变化检验，零效应、反向和舍入误差完整保留，不以删除/救援成立与否决定图谱价值。',
        '局部FP64预测不是FP64模型，也未预测完整下游网络；BF16小变化可以被舍入吞掉。固定生成串256并行因果评分与自然cache概率是不同数值协议，不拼为自然闭环。3072路径重用16前缀和模型参数，非独立语言事实。选定48标量不是全部参数，答案首token可能不足以区分完整名字。已知归一化/softmax公式不是新语义机制。',
        '继续2690按冻结算法对8192独立新实体/新关系填充扩大确认，然后2691四模型本模型坐标顺序验证；保留所有完整背景和部分规律，再完成2692参数账本及2693真实客户端终审。')


def main():
    assert not (OUT/'analysis/final.json').exists();rows,controls,c=prepare()
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(OUT/'protocol/runtime.json',{'dtype':str(model.dtype),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':False})
    maps=collect(model,tok,rows,controls,c);del model;gc.collect();torch.cuda.empty_cache();audit(rows,maps)


if __name__=='__main__':main()
