"""Natural question/context ordering and audited BF16 residual cancellation, no donor transport."""
import gc
from collections import defaultdict
from rdc_operator_common import *
from rdc_operator_qa import cases,prompt,QueryTrace,evaluate,repeated_ngrams
from phase2724_rdc_operator_qa_atlas import token_overlap


def correct_prefix_moments(model):
    import torch
    from rdc_operator_capture import AllFieldObserver
    out=BASE/'identity_audit'
    if (out/'correction_result.json').exists():return read(out/'correction_result.json')
    items=gzread(out/'all_cue_availability_mismatches.json.gz');material={r['sample_id']:r for r in rows()}
    observer=AllFieldObserver(model);corrected={};checks=[];device=model.get_input_embeddings().weight.device
    try:
        for item in items:
            r=material[item['sample_id']];p=item['position'];scope='confirmation' if r['split']=='confirmation' else 'main'
            group=r['split']+'_'+r['language']
            if group not in corrected:
                with np.load(BASE/'capture'/scope/'moments'/f'{group}.npz') as z:corrected[group]={k:z[k].copy() for k in z.files}
            observer.reset([p],False);observer.enabled=True
            model.model(input_ids=torch.tensor([r['prompt_ids']],device=device),use_cache=False)
            observer.enabled=False
            old=gzread(BASE/'capture'/scope/'commits'/f'{r["sample_id"]}.json.gz')
            assert json.loads(json.dumps(observer.hidden_hashes))==old['full_H_identities']
            h=unbits(np.stack([observer.hidden[l][:1] for l in range(37)])[:,0])
            f=torch.as_tensor(h,device=device);u=f/f.square().mean(-1).sqrt().clamp_min(1e-12)[:,None]
            hs=torch.stack([f,f.square(),u,u.square()],1).cpu().numpy().astype(float)
            us=[];raw={'H':np.stack([observer.hidden[l][0] for l in range(37)])}
            for b in (6,16,34):
                d=observer.factors[b];g,up,a=(d[k][0,p].float() for k in ('gate','up','activation'))
                phi=model.model.layers[b].mlp.act_fn(d['gate'])[0,p].float()
                us.append(torch.stack([g,up,a,phi,g.square(),up.square(),a.square(),phi.square(),phi*up]).cpu().numpy())
                for key in ('x','gate','up','activation','mlp'):raw[f'L{b}_{key}']=bits(d[key][0,p])
            us=np.stack(us).astype(float);before=7+item['saved_offset_mask'];after=7+item['known_ID_prefix_mask']
            z=corrected[group];z['H_sums'][:,:,before]-=hs;z['H_sums'][:,:,after]+=hs
            z['unit_sums'][:,:,before]-=us;z['unit_sums'][:,:,after]+=us
            z['counts'][before]-=1;z['counts'][after]+=1;assert np.all(z['counts']>=0)
            npz(out/'correction_fields'/f'{r["sample_id"]}.npz',**raw)
            checks.append({'sample_id':r['sample_id'],'position':p,'all_token_all_layer_original_SHA_exact':True,'from_column':before,'to_column':after,'group':group})
            del raw,d,g,up,a,phi,h,f,u,hs,us;observer.reset([],False)
    finally:observer.close()
    for group,z in corrected.items():npz(out/'corrected_moments'/f'{group}.npz',**z)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'corrected_token_positions':len(items),'affected_anchor_rows':0,'checks':checks,
        'meaning':'Four partial-byte pieces of Chinese reference cue其 were assigned an unavailable full-character cue by offsets. Replayed same full window, verified every layer raw SHA, and moved complete H/unit sufficient statistics to known-ID-prefix category.',
        'storage':'Only affected three group archives have corrected copies; unchanged groups reference original capture moments. No original result overwritten or silently relabeled.',
        'numeric_scope':'FP64 corrections to original FP32-per-source reduced moments, using the exact retained BF16 field values and FP32 elementwise terms. This is not a claim of bit-identical alternative reduction ordering.',
        'impact':'All4096operator anchors had zero cue-availability disagreement, so frozen operator banks/selection and actual causal compiler require no retraining. Original full-token cue summaries retain four-position annotation caveat; corrected sufficient statistics are separately available.'}
    save(out/'correction_result.json',result);return result


def precision(model):
    import torch
    out=BASE/'precision'
    if (out/'result.json').exists():return read(out/'result.json')
    target=read(BASE/'structure/result.json')['residual_precision_target']['target']
    r=next(row for row in rows() if row['sample_id']==target['sample_id']);b,p=target['block'],target['position']
    data={};layer=model.model.layers[b];handles=[]
    def before(m,a):data['r']=a[0][0,p].clone()
    def attention(m,a,o):data['a']=o[0][0,p].clone()
    def mlp(m,a,o):data['m']=o[0,p].clone()
    def after(m,a,o):data['y']=(o[0] if isinstance(o,tuple) else o)[0,p].clone()
    handles=[layer.register_forward_pre_hook(before),layer.self_attn.register_forward_hook(attention),
        layer.mlp.register_forward_hook(mlp),layer.register_forward_hook(after)]
    device=model.get_input_embeddings().weight.device
    try:
        model.model(input_ids=torch.tensor([r['prompt_ids']],device=device),use_cache=False)
    finally:
        for h in handles:h.remove()
    sequential=(data['r']+data['a'])+data['m'];assert torch.equal(sequential,data['y'])
    values={k:v.double().cpu().numpy() for k,v in data.items()}
    total=values['r']+values['a']+values['m'];q=values['y']-total
    E=lambda v:float(np.mean(v*v))
    real=E(total);native=E(values['y']);cross=2*float(np.mean(total*q));rounding=E(q)
    with np.load(BASE/'capture/main/energies'/f'{r["sample_id"]}.npz') as z:saved=z['block_terms'][b,:,p].astype(float)
    assert abs(native-saved[6])/native<1e-6
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'target':target,'native_sequential_BF16_sum_bitwise':True,
        'native_energy_FP64':native,'real_sum_energy_FP64':real,'real_vs_native_energy_relative':abs(real-native)/native,
        'rounding_vector_energy':rounding,'rounding_signed_cross_term':cross,
        'corrected_energy_relative_error':abs(real+cross+rounding-native)/native,
        'original_FP32_summary_energy':float(saved[:6].sum()),'FP32_summary_vs_real_FP64_relative':abs(saved[:6].sum()-real)/native,
        'all2560coordinates':True,
        'conclusion':'At the preselected worst site, native output exactly equals sequential BF16 residual additions. Real-valued sum differs due to observed rounding vector; signed cross term plus its energy closes the gap. No discrete reset or changed physical conservation law is implied.'}
    assert result['corrected_energy_relative_error']<1e-10
    npz(out/'full_native_residual_vectors.npz',**{k:bits(v) for k,v in data.items()},real_sum_FP64=total,rounding_vector_FP64=q)
    save(out/'result.json',result);return result


def main():
    import torch
    from rdc_operator_model import load
    out=BASE/'operations'
    if (out/'result.json').exists():return
    start=time.monotonic();selected=[]
    for lang in ('en','zh'):selected.extend([r for r in cases('qwen4','confirmation') if r['language']==lang][:32])
    assert len(selected)==64
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'question_ids':[r['question_id'] for r in selected],
        'operation':'Question-before-context versus original context-before-question, same original question, complete paragraph and gold. Only order changes; no new synthetic answers or grammar labels.',
        'reference':'Previously committed native qwen4/confirmation context-first outputs and fields, unchanged.',
        'decoding':'Native chat enable_thinking=False,greedy,max48,nativeEOS; no gold inserted.',
        'capture':'Final prompt query full37layers/all2560coordinates, selected3MLPs/all9728units and allsourceattention. Last prompt token identity checked; positions/history lengths can differ.',
        'probability_comparison':'Before execution: compare original-context-first versus question-first prompt-query probabilities using all151936logits inFP64. Original saved postnorm is read through unchanged nativeBF16head and argmax/entropy checked against originalQAcommit. Four predetermined indices0/1/32/33 retain both full logit arrays in their field archives. No gold answer participates in this comparison.',
        'interpretation':'Instruction-order robustness, not guaranteed arbitrary paraphrase invariance or a position-controlled pure semantic intervention.'})
    model,tok=load('qwen4',out);device=model.get_input_embeddings().weight.device
    observer=QueryTrace(model,[6,16,34]);records=[];moments=np.zeros((3,37,2560),np.float64)
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
    try:
      with torch.inference_mode():
        from phase2726_rdc_operator_scale_q4 import run as matched_scale
        matched=matched_scale(model,tok)
        cue_correction=correct_prefix_moments(model)
        audit=precision(model)
        for i,row in enumerate(selected):
            cp=out/'commits'/f'{row["question_id"]}.json'
            ref=read(BASE/'qa/qwen4/confirmation/commits'/f'{row["question_id"]}.json')
            user=prompt(row,question_first=True)
            actual=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(actual,add_special_tokens=False,return_offsets_mapping=True)
            ids=torch.tensor([enc['input_ids']],device=device)
            observer.data,observer.layers,observer.enabled={},{},True
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            observer.enabled=False
            field={**observer.data,'H':np.stack([observer.layers[l] for l in range(37)]),'postnorm':bits(post[0,-1])}
            with np.load(BASE/'qa/qwen4/confirmation/fields'/f'{row["question_id"]}.npz') as z:
                old=unbits(z['H']).astype(float)
                old_post=torch.as_tensor(unbits(z['postnorm']),device=device).to(model.dtype)
            old_logits=model.lm_head(old_post).float();new_logits=model.lm_head(post[0,-1]).float()
            old_lp32=old_logits.log_softmax(-1)
            assert int(old_logits.argmax())==ref['prompt_native_argmax']
            assert abs(float(-(old_lp32.exp()*old_lp32).sum())-ref['prompt_native_output_entropy'])<1e-5
            old_lp=old_logits.double().log_softmax(-1);new_lp=new_logits.double().log_softmax(-1)
            probability={'full_vocabulary':len(old_logits),'KL_context_first_to_question_first':float((old_lp.exp()*(old_lp-new_lp)).sum()),
                'original_head_argmax_replay_exact':True,'query_head_argmax_agrees':bool(old_logits.argmax()==new_logits.argmax()),
                'question_first_query_argmax':int(new_logits.argmax()),'context_first_entropy':float(-(old_lp.exp()*old_lp).sum()),
                'question_first_entropy':float(-(new_lp.exp()*new_lp).sum())}
            if i in (0,1,32,33):
                field['context_first_query_logits']=old_logits.cpu().numpy();field['question_first_query_logits']=new_logits.cpu().numpy()
            from rdc_operator_capture import persist_arrays
            persist_arrays(out/'fields'/f'{row["question_id"]}.npz',field)
            del old_post,old_logits,new_logits,old_lp32,old_lp,new_lp
            new=unbits(field['H']).astype(float);moments[0]+=old;moments[1]+=new;moments[2]+=(new-old)**2
            sequence=model.generate(input_ids=ids,do_sample=False,max_new_tokens=48,use_cache=True,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=eos)[0,ids.shape[1]:].tolist()
            text=tok.decode(sequence,skip_special_tokens=True);score=evaluate(text,row['answers'],row['language'])
            base=actual.find(row['full_context']);qstart=actual.find(row['question']);assert base>=0 and qstart>=0
            spans=[[base+a['answer_start'],base+a['answer_start']+len(a['text'])] for a in row['answers']]
            groups={'question':token_overlap(enc['offset_mapping'],[[qstart,qstart+len(row['question'])]]),
                'context':token_overlap(enc['offset_mapping'],[[base,base+len(row['full_context'])]]),'gold_answer':token_overlap(enc['offset_mapping'],spans)}
            paired_attention={}
            for b in (6,16,34):
                att=unbits(field[f'L{b}_attention_sources']).astype(float)
                paired_attention[str(b)]={k:float(att[:,v].sum(1).mean()) if v else None for k,v in groups.items()}
            r={k:row[k] for k in ('question_id','sample_id','source_group','language','question','question_type','answers')}
            r.update(actual_prompt=actual,prompt_ids=enc['input_ids'],token_offsets=enc['offset_mapping'],generated_ids=sequence,generated_text=text,
                normalized_full_EM=score['normalized_full_EM'],strict_full_answer=score['strict_full_answer'],answer_F1=score['answer_F1'],
                stopped_by_native_EOS=any(t in stop for t in sequence),repeated_4gram_fraction=repeated_ngrams(sequence),
                context_first={k:ref[k] for k in ('generated_text','generated_ids','normalized_full_EM','answer_F1','stopped_by_native_EOS')},
                same_final_prompt_token_id=enc['input_ids'][-1]==ref['prompt_ids'][-1],prompt_length_difference=len(enc['input_ids'])-len(ref['prompt_ids']),
                per_layer_full_coordinate_relative_MSE=np.mean((new-old)**2,1)/np.maximum(np.mean(old*old,1),1e-20),
                per_layer_full_coordinate_cosine=np.sum(new*old,1)/np.maximum(np.linalg.norm(new,axis=1)*np.linalg.norm(old,axis=1),1e-20),
                source_token_groups=groups,retrospective_attention=paired_attention)
            r['full_query_probability_comparison']=probability
            r['per_layer_full_coordinate_relative_MSE']=r['per_layer_full_coordinate_relative_MSE'].tolist()
            r['per_layer_full_coordinate_cosine']=r['per_layer_full_coordinate_cosine'].tolist()
            save(cp,r);records.append(r)
            observer.data,observer.layers={},{};del ids,post,field,old,new
            if i<2 or (i+1)%16==0:print('NATURAL_ORDER',i+1,64,'seconds',round(time.monotonic()-start,1),flush=True)
            guard()
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    npz(out/'all_query_order_moments.npz',reference_mean=moments[0]/64,question_first_mean=moments[1]/64,mean_coordinate_squared_difference=moments[2]/64)
    transitions=defaultdict(int)
    for r in records:transitions[str(int(r['context_first']['normalized_full_EM']))+'->'+str(int(r['normalized_full_EM']))]+=1
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'questions':len(records),'context_first_EM':float(np.mean([r['context_first']['normalized_full_EM'] for r in records])),
        'question_first_EM':float(np.mean([r['normalized_full_EM'] for r in records])),'normalized_EM_transitions':dict(transitions),
        'question_first_F1':float(np.mean([r['answer_F1'] for r in records])),'context_first_F1':float(np.mean([r['context_first']['answer_F1'] for r in records])),
        'paired_F1_difference_cluster':clustered([r['answer_F1']-r['context_first']['answer_F1'] for r in records],[r['source_group'] for r in records]),
        'question_first_EOS':sum(r['stopped_by_native_EOS'] for r in records),'same_last_token_ID':sum(r['same_final_prompt_token_id'] for r in records),
        'mean_full_vocab_query_KL':float(np.mean([r['full_query_probability_comparison']['KL_context_first_to_question_first'] for r in records])),
        'query_head_argmax_agreement':float(np.mean([r['full_query_probability_comparison']['query_head_argmax_agrees'] for r in records])),
        'mean_layer_relative_MSE':np.mean([r['per_layer_full_coordinate_relative_MSE'] for r in records],0).tolist(),
        'matched_Q4_scale':{'sources':matched['sources'],'blocks':matched['own_blocks'],'choices':matched['choices']},
        'precision_audit':audit,'cue_correction':cue_correction,'limits':'String-match change can include format/paraphrase variation. Operation changes order/position/cross-token causal availability, not only an isolated semantic variable. No internal invariance is inferred just from unchanged answer.'}
    save(out/'result.json',result);ledger('natural_QA_order_and_BF16_precision',time.monotonic()-start,questions=64);guard()
    print('NATURAL_ORDER_COMPLETE',result,flush=True)


if __name__=='__main__':main()
