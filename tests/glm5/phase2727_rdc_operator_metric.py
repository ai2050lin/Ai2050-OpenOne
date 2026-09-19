"""Automatic same-goal followup: output-sensitive composition and full-vocabulary metric.

Uses existing confirmation diagnostically and a separate native question-query task
transfer. No post-hoc source collection is called a new untouched semantic holdout.
"""
import argparse
import gc
from collections import defaultdict
from rdc_operator_common import *
from phase2726_rdc_operator_compile import Compiler,causal_meta
from rdc_operator_qa import cases,repeated_ngrams
from rdc_native_conditional_operator import apply_operator


def protocol():
    out=BASE/'metric_followup';compiled=read(BASE/'compiled/result.json')
    val=compiled['local']['validation'];original=read(BASE/'operators/frozen.json')['choices'];choice={}
    for b in (6,16,34):
        best=min([r for r in val if r['name'] in (f'L{b}_global',f'L{b}_selected')],key=lambda r:(r['raw_KL'],r['name']))['name']
        choice[b]=original[str(b)] if best.endswith('selected') else 'frozen_gate_global'
    natural=[r for r in rows() if r['split']=='confirmation']
    qa=[]
    for lang in ('en','zh'):qa.extend([r for r in cases('qwen4','confirmation') if r['language']==lang][:32])
    multi=gzread(BASE/'qa_multihop_material.json.gz')
    for typ in ('bridge','comparison'):qa.extend([r for r in multi if r['question_type']==typ][:32])
    p={'timestamp':stamp(),'source':snapshot(Path(__file__)),'parent_compiled_sha':sha(BASE/'compiled/result.json'),'hybrid':choice,
        'selection':'Per block choose original global or original selected by existing128validation-anchor full-vocabulary KL. No selection on confirmation or QA. Hybrid joint configuration itself has not been tested.',
        'natural_sources':256,'natural_anchors':512,'QA_transfer_queries':len(qa),'QA_question_ids':[r['question_id'] for r in qa],
        'QA_transfer_scope':'New task-boundary/local-operator transfer on original native human-question chat queries; baseline QA behavior was already observed. Not fresh articles, unseen semantic concepts or a new untouched hypothesis-test set.',
        'metric':'Exact full-vocabulary logit-path KL identity integral (1-s)Var_softmax(z+s*delta)(delta) ds, Gauss-Legendre32 then64node discrepancy; quadratic endpoint Fisher variance separately. No Top-K vocabulary.',
        'autonomous':'64same natural prefixes,48step hybrid with independent approximate KV; native diagnostics maintain a separate KV for exactly the same chosen-token history and never refresh the approximate branch. Four complete native layer/postnorm query vectors per step retained. Native remaining modules and first-position exception remain.',
        'objective':'Explain why lower local coordinate MSE can fail to improve probability, test output-sensitive composition without claiming a reset/repetition cure.',
        'resource':'Same existing6GiB/6hour envelope; no additional pretrained model training or unbounded corpus expansion.'}
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',p)
    stored=read(out/'protocol.json');assert {int(k):v for k,v in stored['hybrid'].items()}==choice
    return out,choice,natural,qa


def logit_metric(z,zz):
    import torch
    # All151936 categories are retained. Float64 log partitions isolate quadrature error.
    z=z.double();delta=(zz.double()-z);lp=z.log_softmax(-1);p=lp.exp();pp=zz.double().log_softmax(-1)
    exact=float((p*(lp-pp)).sum());mean=(p*delta).sum();var=float((p*(delta-mean).square()).sum())
    answers={}
    for n in (32,64,128,256,512):
        nodes,weights=np.polynomial.legendre.leggauss(n);total=0.
        for at in range(0,n,8):
            s=torch.as_tensor((nodes[at:at+8]+1)/2,device=z.device)
            w=torch.as_tensor(weights[at:at+8]/2,device=z.device)
            path=(z[None]+s[:,None]*delta[None]).softmax(-1)
            mu=(path*delta[None]).sum(-1)
            variance=(path*(delta[None]-mu[:,None]).square()).sum(-1)
            total+=float((w*(1-s)*variance).sum())
        answers[n]=total
        if n>=64 and abs(total-exact)<1e-7 and abs(total-answers[n//2])<1e-7:
            break
    final_nodes=max(answers)
    return {'exact_KL':exact,'endpoint_Fisher_half_variance':.5*var,'integral32':answers[32],'integral64':answers[64],
        'integral64_absolute_error':abs(answers[64]-exact),'quadrature32_vs64':abs(answers[32]-answers[64]),
        'adaptive_integral':answers[final_nodes],'adaptive_nodes':final_nodes,'adaptive_absolute_error':abs(answers[final_nodes]-exact),
        'logit_delta_full_variance_unweighted':float(delta.var(unbiased=False)),
        'native_entropy':float(-(p*lp).sum()),'mean_logit_shift':float(delta.mean()),'argmax_agreement':bool(z.argmax()==zz.argmax())}


def complete_readout_geometry(model,z,zz,packet,out,job):
    import torch
    path=out/'readout_geometry'/f'{job["id"]}.json'
    if path.exists():return read(path)
    start=time.monotonic();width=model.config.hidden_size;p=z.double().softmax(-1)
    gram=torch.zeros((width,width),device=z.device,dtype=torch.float64);mean=torch.zeros(width,device=z.device,dtype=torch.float64)
    dh=torch.as_tensor(unbits(packet['joint_global_postnorm']).astype(float)-unbits(packet['native_postnorm']).astype(float),device=z.device,dtype=torch.float64)
    logit_parts=[]
    # Streaming vocabulary blocks is exact coverage, not vocabulary or coordinate selection.
    for at in range(0,len(p),4096):
        w=model.lm_head.weight[at:at+4096].double();prob=p[at:at+4096]
        weighted=prob[:,None]*w;gram.addmm_(w.T,weighted);mean+=(weighted).sum(0)
        logit_parts.append(w@dh);del w,weighted,prob
    gram-=mean[:,None]*mean[None,:]
    real_delta=torch.cat(logit_parts);mu=(p*real_delta).sum()
    direct=float((p*(real_delta-mu).square()).sum());matrix=float(dh@gram@dh)
    assert abs(matrix-direct)/max(direct,1e-12)<1e-8
    diagonal=float((dh.square()*gram.diagonal()).sum());actual_delta=zz.double()-z.double();am=(p*actual_delta).sum()
    actual=float((p*(actual_delta-am).square()).sum())
    symmetric=float((gram-gram.T).abs().max());assert symmetric<1e-9
    npz(out/'readout_geometry'/f'{job["id"]}.npz',G_full_native_coordinates=gram.cpu().numpy(),
        full_readout_row_mean=mean.cpu().numpy(),postnorm_delta=dh.cpu().numpy(),real_readout_logit_delta=real_delta.cpu().numpy(),
        observed_BF16_head_logit_delta=actual_delta.cpu().numpy())
    result={'timestamp':stamp(),'query_id':job['id'],'scope':job['scope'],'full_native_width':width,'full_vocabulary':len(p),
        'formula':'G_h=W_U^T[diag(p_native)-p_native p_native^T]W_U; all categories/coordinates retained.',
        'direct_full_readout_variance':direct,'full_matrix_quadratic_form':matrix,'diagonal_only_quadratic_form':diagonal,
        'offdiagonal_signed_contribution':matrix-diagonal,'observed_BF16_head_delta_variance':actual,
        'matrix_symmetry_max_error':symmetric,'seconds':time.monotonic()-start,
        'limits':'Known real-valued linear-readout sensitivity around native probability. Actual native BF16 head has rounding, so observed finite logit differences need not equal W_U times postnorm state difference. Postnorm space, not raw pre-RMS residual space. Two predetermined representatives, not a universal coordinate geometry.'}
    save(path,result);return result


def main():
    import torch
    from rdc_operator_model import load
    out,hybrid,natural,qa=protocol()
    if (out/'result.json').exists():return
    start=time.monotonic();guard(340*1024**2)
    if not (out/'quadrature_refinement.json').exists():
        immutable(out/'quadrature_refinement.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
            'refinement':'Before full-model execution: keep original32/64 estimates, adapt to128/256/512 if absolute identity error or nested discrepancy exceeds1e-7. Retain failed record and halt if final error exceeds1e-5; no difficult query discarded.',
            'selection_unchanged':True})
    original=read(BASE/'operators/frozen.json')['choices']
    configs={'joint_global':{b:'frozen_gate_global' for b in (6,16,34)},
        'joint_selected':{b:original[str(b)] for b in (6,16,34)},'output_selected_hybrid':hybrid}
    model,tok=load('qwen4',out);device=model.get_input_embeddings().weight.device
    compiler=Compiler(model,tok);records=[];local_records=[];local_fields={};capture_local=False;local_handles=[]
    for b in (6,16,34):
        def capture(m,a,o,b=b):
            if capture_local:local_fields[b]=(a[0][0,-1:].clone(),o[0,-1:].clone())
        local_handles.append(model.model.layers[b].mlp.register_forward_hook(capture))
    if not (out/'same_prefix_local_refinement.json').exists():
        immutable(out/'same_prefix_local_refinement.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
            'motivation':'Primary local fit was evaluated on full-window collection, while real-network compilation uses prefix-only execution. Same exact prefix/native input local MSE is needed before attributing rank inversion to output geometry.',
            'procedure':'At original native query, capture full x and native MLP output at all3blocks. Compare existing frozen global/selected operators on that identical x; cast prediction to nativeBF16 exactly as actual substitution. No fitting, no new selection, no extra reference state in autonomous branch.',
            'scope':'Re-analysis / predeclared task transfer; execution shape is explicitly controlled, not assumed irrelevant.'})
    if not (out/'readout_geometry_protocol.json').exists():
        immutable(out/'readout_geometry_protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
            'representatives':'Predetermined jobs0(first natural confirmation anchor) and512(first QA transfer query), not effect-selected.',
            'procedure':'Complete2560x2560 real-valued readout Fisher pullback from every151936native vocabulary row in4096row streaming blocks; FP64 accumulation, no PCA/Top-K. Compare full/diagonal quadratic form to direct full-vocabulary variance, plus actualBF16logit-delta variance.',
            'resource':'Two full FP64 matrices within existing6GiB result envelope; same loaded4B, no concurrent model. Known identity, not a new mathematical structure.'})
    try:
      with torch.inference_mode():
        jobs=[]
        for r in natural:
            for a,p in enumerate(r['anchors']):jobs.append({'id':r['sample_id']+f'_a{a}','sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'scope':'natural_reanalysis','ids':r['prompt_ids'][:p+1]})
        for r in qa:
            scope='main' if r['question_type'] in ('bridge','comparison') else 'confirmation'
            cp=read(BASE/'qa/qwen4'/scope/'commits'/f'{r["question_id"]}.json')
            jobs.append({'id':r['question_id'],'sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'scope':'QA_query_transfer','ids':cp['prompt_ids'],
                'native_answer_string_match':cp['normalized_full_EM'],'question':r['question'],'reference_generated_first_ID':cp['generated_ids'][0]})
        for i,r in enumerate(jobs):
            cp=out/'commits'/f'{r["id"]}.json'
            if cp.exists():
                saved=read(cp);records.extend(saved['rows']);local_records.extend(saved['same_prefix_local']);continue
            ids=torch.tensor([r['ids']],device=device);compiler.meta=causal_meta(r['ids'],r['language'],tok);compiler.local=True;compiler.active={}
            local_fields={};capture_local=True
            native=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1]
            capture_local=False
            if r['scope']=='QA_query_transfer':
                qa_scope='main' if any(s['question_id']==r['id'] and s['question_type'] in ('bridge','comparison') for s in qa) else 'confirmation'
                with np.load(BASE/'qa/qwen4'/qa_scope/'fields'/f'{r["id"]}.npz') as old:
                    assert np.array_equal(bits(native),old['postnorm']),r['id']
            z=model.lm_head(native).float();packet={'native_postnorm':bits(native)};rr=[];lr=[]
            for b,(xx,target) in local_fields.items():
                packet[f'L{b}_native_x']=bits(xx[0]);packet[f'L{b}_native_mlp']=bits(target[0])
                for name in ('global','selected'):
                    operator='frozen_gate_global' if name=='global' else original[str(b)]
                    pred=apply_operator(operator,xx.float(),compiler.meta[-1:],compiler.banks[b],compiler.w[b]).to(target.dtype).float()
                    lr.append({k:r[k] for k in ('id','sample_id','source_group','language','scope')}|{'block':b,'name':name,'actual_operator':operator,
                        'relative_MSE':float((pred-target.float()).square().mean()/target.float().square().mean()),
                        'raw_MSE':float((pred-target.float()).square().mean())})
                del pred
            for name,config in configs.items():
                compiler.active=config;prediction=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1]
                zz=model.lm_head(prediction).float();metric=logit_metric(z,zz)
                if metric['adaptive_absolute_error']>=1e-5:
                    save(out/'quadrature_failure.json',{'job':r,'name':name,'metric':metric})
                    raise AssertionError(metric)
                metric.update({k:r[k] for k in ('id','sample_id','source_group','language','scope')});metric['name']=name
                metric['postnorm_coordinate_relative_MSE']=float((native.float()-prediction.float()).square().mean()/native.float().square().mean())
                if 'native_answer_string_match' in r:metric['native_answer_string_match']=r['native_answer_string_match']
                rr.append(metric);packet[name+'_postnorm']=bits(prediction)
                if name=='joint_global':global_logits=zz.clone()
                if i in (0,1,512,513):npz(out/'full_vocab'/f'{r["id"]}_{name}.npz',native_logits=z.cpu().numpy(),approximate_logits=zz.cpu().numpy())
            npz(out/'fields'/f'{r["id"]}.npz',**packet)
            if i in (0,512):complete_readout_geometry(model,z,global_logits,packet,out,r)
            save(cp,{'row':r,'rows':rr,'same_prefix_local':lr});records.extend(rr);local_records.extend(lr);local_fields={}
            del ids,native,z,zz,prediction,packet,global_logits
            if i<2 or (i+1)%32==0:print('OUTPUT_METRIC',i+1,len(jobs),'seconds',round(time.monotonic()-start,1),flush=True)
            guard();assert time.monotonic()-start<7200
        # Four complete query vectors are retained: three observed block outputs and final normalized state.
        traces={};handles=[]
        for b in (6,16,34):
            def trace(m,a,o,b=b):
                o=o[0] if isinstance(o,tuple) else o;traces[b]=bits(o[0,-1])
            handles.append(model.model.layers[b].register_forward_hook(trace))
        selected=[]
        for lang in ('en','zh'):selected.extend([r for r in natural if r['language']==lang][:32])
        eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);autonomous=[]
        for i,r in enumerate(selected):
            cp=out/'autonomous'/f'{r["sample_id"]}.json'
            if cp.exists():autonomous.append(read(cp));continue
            initial=r['prompt_ids'][:r['anchors'][0]+1];known=list(initial);generated=[];cache=None;native_cache=None
            stats=[];approxfields=[];nativefields=[]
            for step in range(48):
                compiler.local=False;compiler.meta=causal_meta(known,r['language'],tok);compiler.active=hybrid
                ids=torch.tensor([known if cache is None else [known[-1]]],device=device)
                output=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=output.past_key_values
                zz=model.lm_head(output.last_hidden_state[0,-1]).float();lp=zz.log_softmax(-1);t=int(zz.argmax())
                approxfields.append(np.stack([traces[b] for b in (6,16,34)]+[bits(output.last_hidden_state[0,-1])]))
                compiler.active={}
                reference=model.model(input_ids=ids,past_key_values=native_cache,use_cache=True);native_cache=reference.past_key_values
                assert cache is not native_cache
                z=model.lm_head(reference.last_hidden_state[0,-1]).float();ref_lp=z.log_softmax(-1)
                nativefields.append(np.stack([traces[b] for b in (6,16,34)]+[bits(reference.last_hidden_state[0,-1])]))
                stats.append({'step':step,'chosen_ID':t,'native_same_history_KL':float((ref_lp.exp()*(ref_lp-lp)).sum()),
                    'native_same_history_argmax_agrees':bool(z.argmax()==t),'native_same_history_chosen_NLL':float(-ref_lp[t]),
                    'approx_chosen_NLL':float(-lp[t]),'native_same_history_entropy':float(-(ref_lp.exp()*ref_lp).sum()),
                    'approx_entropy':float(-(lp.exp()*lp).sum())})
                known.append(t);generated.append(t)
                if t in stop:break
            npz(out/'autonomous_fields'/f'{r["sample_id"]}.npz',approximate_query_fields=np.stack(approxfields),native_same_chosen_history_query_fields=np.stack(nativefields))
            rec={'sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'initial_ids':initial,'generated_ids':generated,
                'generated_text':tok.decode(generated,skip_special_tokens=True),'stopped_by_native_EOS':any(t in stop for t in generated),
                'repeated_4gram_fraction':repeated_ngrams(generated),'steps':stats,
                'scope':'Native reference uses identical chosen-token history but separate cached computation; no state or token from that diagnostic is fed back into the hybrid.'}
            save(cp,rec);autonomous.append(rec)
            del cache,native_cache,output,reference,z,zz,lp,ref_lp,ids,approxfields,nativefields
            if i<2 or (i+1)%8==0:print('OUTPUT_HYBRID_HISTORY',i+1,64,'seconds',round(time.monotonic()-start,1),flush=True)
            guard();assert time.monotonic()-start<7200
        for handle in handles:handle.remove()
    finally:
        for handle in local_handles:handle.remove()
        compiler.close();del compiler,model;gc.collect();torch.cuda.empty_cache()
    summaries=[]
    for scope in ('natural_reanalysis','QA_query_transfer'):
        for name in configs:
            rr=[r for r in records if r['scope']==scope and r['name']==name]
            summaries.append({'scope':scope,'name':name,'queries':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('exact_KL','endpoint_Fisher_half_variance','integral64','adaptive_integral','postnorm_coordinate_relative_MSE','argmax_agreement')},
                'max_quadrature_identity_error':max(r['adaptive_absolute_error'] for r in rr),
                'max_quadrature_nodes':max(r['adaptive_nodes'] for r in rr),
                'KL_cluster':clustered([r['exact_KL'] for r in rr],[r['source_group'] for r in rr])})
    same_prefix=[]
    for scope in ('natural_reanalysis','QA_query_transfer'):
        for b in (6,16,34):
            for name in ('global','selected'):
                rr=[r for r in local_records if r['scope']==scope and r['block']==b and r['name']==name]
                same_prefix.append({'scope':scope,'block':b,'name':name,'queries':len(rr),'relative_MSE':float(np.mean([r['relative_MSE'] for r in rr])),
                    'raw_MSE':float(np.mean([r['raw_MSE'] for r in rr])),'relative_MSE_cluster':clustered([r['relative_MSE'] for r in rr],[r['source_group'] for r in rr])})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'hybrid':hybrid,'summaries':summaries,'same_prefix_local':same_prefix,'autonomous_sources':len(autonomous),
        'complete_readout_geometry':[read(p) for p in sorted((out/'readout_geometry').glob('*.json'))],
        'hybrid_autonomous_repetition':float(np.mean([r['repeated_4gram_fraction'] for r in autonomous])),
        'hybrid_autonomous_EOS':sum(r['stopped_by_native_EOS'] for r in autonomous),
        'mean_native_same_history_KL':float(np.mean([s['native_same_history_KL'] for r in autonomous for s in r['steps']])),
        'native_same_history_argmax_agreement':float(np.mean([s['native_same_history_argmax_agrees'] for r in autonomous for s in r['steps']])),
        'field_axes':['generation_step','block6/16/34 output + final_postnorm','all2560nativecoordinates'],
        'limits':'Known softmax log-partition identity and an empirical route-selection result, not new mathematics or a universal semantic metric. The straight interpolation z+s*delta is an analytic logit-space path, not an observed native layer/generation trajectory. Natural confirmation is re-analysis after the earlier output results were observed. QA is task-query transfer, not brand-new source/semantic holdout. Partial native network remains; repetition statistics cannot identify a reset mechanism.'}
    save(out/'result.json',result);ledger('output_metric_and_own_history_followup',time.monotonic()-start);guard()
    print('OUTPUT_METRIC_COMPLETE',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if args.plan_only:
        out,hybrid,_,_=protocol();print('FOLLOWUP_FROZEN',str(out),hybrid,flush=True)
    else:main()
