"""Heldout prefix-only native fields, matched-shape future controls, frozen forecasts."""
import gc
from rdc_update_common import *


def material():
    rows=gzread(BASE/'language_material.json.gz');lookup={(r['source_group'],r['language'],r['answer_style']):r for r in rows}
    pairs=[];prefixes=[]
    for row in rows:
        if row['split']!='language_test' or row['answer_style']!='direct':continue
        other=lookup[row['source_group'],row['language'],'explain'];p=row['anchors'][0];ids=row['prompt_ids'][:p+1]
        assert ids==other['prompt_ids'][:other['anchors'][0]+1]
        prefixes.append(row|{'prompt_ids':ids,'anchors':[p],'field_anchor':0,'capture_mode':'causal_replay',
          'cohort':row['cohort']+'/body_prefix','token_offsets':row['token_offsets'][:p+1],
          'text':row['text'][:row['token_offsets'][p][1]],'actual_input':'Only saved prefix IDs; text is display metadata.',
          'same_prefix_expression_aliases':[row['sample_id'],other['sample_id']]})
        pairs.append((row,other))
    assert len(prefixes)==80 and len({tuple(r['prompt_ids']) for r in prefixes})==80
    return pairs,prefixes


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    out=BASE/'causal_replay';start=time.monotonic()
    if (out/'result.json').exists():return
    guard(200*1024**2)
    booked=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'))
    assert booked+600<read(BASE/'resources.json')['compute_ceiling_seconds'],'Reserve a bounded600second recovery test, not an unbounded new campaign.'
    pairs,rows=material();compressed(out/'material.json.gz',rows)
    frozen=read(BASE/'graph/frozen.json')
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'pairs':80,'semantic_groups':40,'logical_expression_aliases':160,
      'sample_ids':[r['sample_id'] for r in rows],'material_sha256':sha(BASE/'language_material.json.gz'),
      'predictor_freeze_sha256':sha(BASE/'graph/frozen.json'),'selection':'Every held-language group, both languages; direct/explain share one exact earlier body prefix. No success selection.',
      'native_calls_per_pair':4,'calls':['prefix-only unpadded','same prefix exact repeat','direct suffix padded to pair-common length','explain suffix padded to same length'],
      'matched_shape':'Right padding with native pad ID and explicit key attention mask. Same batch1 total length, same causal body positions, only future suffix differs.',
      'forecast':'Old320natural-train banks and selected decoders retained. Test inputs and native targets from actual prefix-only forward, not full future prompt.',
      'boundaries':'80unique prefixes, not160independent body inputs. Same held groups already observed; this is a numerical/causal-control recovery, not new semantic confirmation.',
      'resource_ceiling_seconds':600,'quantized':False,'dtype':'Original native BF16'}
    if (out/'protocol.json').exists():
        original_protocol=read(out/'protocol.json')
        for key,value in protocol.items():
            if key not in ('timestamp','source'):assert original_protocol[key]==value,(key,'Frozen recovery protocol changed')
    else:immutable(out/'protocol.json',protocol)
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    observer=Observer(model);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if isinstance(pad,list):pad=pad[0]
    records=[]
    def capture(ids,p,mask=None):
        observer.reset([p],False);observer.enabled=True
        args={'input_ids':torch.tensor([ids],device='cuda'),'use_cache':False}
        if mask is not None:args['attention_mask']=torch.tensor([mask],device='cuda')
        post=model.model(**args).last_hidden_state;observer.enabled=False
        packet={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,
          'embedding':observer.H[0],'postnorm':bits(post[0,[p]]),**observer.factors,
          'positions':np.array([p]),'token_ids':np.array(ids,np.int32)}
        hashes=observer.H_hashes.copy();observer.reset([],False);return packet,hashes
    try:
      with torch.inference_mode():
        for i,((direct,explain),row) in enumerate(zip(pairs,rows)):
            path=out/'fields'/f'{row["sample_id"]}.npz';cp=out/'commits'/f'{row["sample_id"]}.json'
            if cp.exists():
                r=read(cp);assert sha(path)==r['array_sha256'];records.append(r);continue
            tick=time.monotonic();p=row['anchors'][0];prefix,hashes=capture(row['prompt_ids'],p)
            repeat,_=capture(row['prompt_ids'],p);assert np.array_equal(prefix['H'],repeat['H'])
            del repeat;length=max(len(direct['prompt_ids']),len(explain['prompt_ids']));padded=[]
            for r in (direct,explain):
                ids=r['prompt_ids'];packet,_=capture(ids+[pad]*(length-len(ids)),p,[1]*len(ids)+[0]*(length-len(ids)))
                padded.append(packet['H']);del packet
            old=[]
            for r in (direct,explain):
                with np.load(BASE/'language_capture/fields'/f'{r["sample_id"]}.npz') as z:old.append(z['H'][:,0:1].copy())
            def relative(a,b):
                a,b=unbits(a).astype(float),unbits(b).astype(float)
                return np.sqrt(((a-b)**2).sum((1,2))/np.maximum((b*b).sum((1,2)),1e-30))
            same=(padded[0]==padded[1]).all((1,2))
            prefix.update(padded_direct_H=padded[0],padded_explain_H=padded[1],same_length_suffix_bit_equal=same,
              prefix_vs_old_direct_relative_RMS=relative(prefix['H'],old[0]),
              prefix_vs_old_explain_relative_RMS=relative(prefix['H'],old[1]),
              old_style_relative_RMS=relative(old[1],old[0]))
            npz(path,**prefix)
            r={'sample_id':row['sample_id'],'aliases':row['same_prefix_expression_aliases'],'source_group':row['source_group'],
              'family':row['family'],'language':row['language'],'prefix_tokens':len(row['prompt_ids']),'original_lengths':[len(direct['prompt_ids']),len(explain['prompt_ids'])],
              'pair_common_padded_length':length,'prefix_repeat_all37_layers_exact':True,
              'same_length_suffix_all37_layers_exact':bool(same.all()),'same_length_suffix_exact_layers':same.tolist(),
              'H12_prefix_vs_old_direct_relative_RMS':float(prefix['prefix_vs_old_direct_relative_RMS'][12]),
              'H12_prefix_vs_old_explain_relative_RMS':float(prefix['prefix_vs_old_explain_relative_RMS'][12]),
              'H12_old_direct_vs_explain_relative_RMS':float(prefix['old_style_relative_RMS'][12]),
              'prefix_alltoken_layer_identities':hashes,'array_sha256':sha(path),'seconds':time.monotonic()-tick}
            save(cp,r);records.append(r);del prefix,padded,old;guard(4*1024**2)
            assert time.monotonic()-start<600
            if i<2 or (i+1)%8==0:print('CAUSAL_PREFIX_REPLAY',i+1,80,'same-shape',r['same_length_suffix_all37_layers_exact'],round(time.monotonic()-start,1),flush=True)
        observer.close();observer=None;model=None;gc.collect();torch.cuda.empty_cache()
        # Local routing only; no edits to the original field selector or frozen data.
        import rdc_update_graph as graph
        from phase2736_rdc_update_prediction import decoders,reports
        from rdc_law_native import parameter
        original_path,original_sources=graph.native_path,graph.sources
        def path_for(r):return out/'fields'/f'{r["sample_id"]}.npz' if r.get('capture_mode')=='causal_replay' else original_path(r)
        def sources_for(r):
            if r.get('capture_mode')!='causal_replay':return original_sources(r)
            with np.load(path_for(r)) as z:return unbits(z['H12_sources'])
        graph.native_path,graph.sources=path_for,sources_for
        predictions=[];paired=[];protocol_shift=[]
        try:
            with np.load(BASE/'graph/head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
            oldrows=[r for r in gzread(PRIOR/'natural_discovery.json.gz') if r['split']=='train']
            left,targets,_=graph.pack(rows,coef);right,_,_=graph.pack(oldrows,coef);kk=graph.kernels(left,right);del left,right
            old_index={(r['sample_id'],r['field_anchor']):i for i,r in enumerate(gzread(BASE/'language_prediction/row_identity.json.gz'))}
            original_ix=np.array([old_index[r['sample_id'],0] for r in rows]);groups=[r['source_group'] for r in rows]
            for block,yy0 in targets.items():
                selected=frozen['selected'][str(block)];yy=torch.tensor(yy0[:,:2560],device='cuda');errors={}
                w={k:parameter(f'model.layers.{block}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
                for name in (selected['kernel'],'query'):
                    bank=f'b{block}_{name}_{selected["df"]}.npz';assert sha(BASE/'graph/banks'/bank)==frozen['banks_sha256'][bank]
                    with np.load(BASE/'graph/banks'/bank) as z:
                        center=torch.tensor(z['center'],device='cuda');value=kk[name]/float(z['scale'])@torch.tensor(z['coefficients'],device='cuda')+center
                    prediction=decoders(value,w)[selected['decoder']];err=(prediction-yy).square().sum(-1).cpu().numpy()
                    den=(yy-center[:2560]).square().sum(-1).cpu().numpy();errors[name]=err
                    with np.load(BASE/'language_prediction'/f'b{block}_{name}.npz') as z:
                        old_err=z['squared_error'][original_ix];old_den=z['baseline_squared_error'][original_ix]
                    npz(out/'prediction'/f'b{block}_{name}.npz',prediction=prediction.cpu().numpy(),actual=yy.cpu().numpy(),
                      squared_error=err,baseline_squared_error=den,old_full_prompt_squared_error=old_err,old_full_prompt_baseline_squared_error=old_den)
                    predictions.append({'block':block,'kernel':name,'decoder':selected['decoder'],'reports':reports(rows,err,den)})
                    protocol_shift.append({'block':block,'kernel':name,'relative_error_prefix_minus_old_full_prompt':clustered(err/den.clip(1e-12)-old_err/old_den.clip(1e-12),groups),
                      'scope':'Both inputs AND native targets use the corresponding execution protocol. This is numerical-protocol sensitivity, not an independent semantic performance gain.'})
                for cohort in sorted({r['cohort'] for r in rows}):
                    ix=[i for i,r in enumerate(rows) if r['cohort']==cohort]
                    delta=(errors[selected['kernel']][ix]-errors['query'][ix])/den[ix].clip(1e-12)
                    paired.append({'block':block,'cohort':cohort,'unique_prefixes':len(ix),'directed_minus_query':clustered(delta,[groups[i] for i in ix])})
                del w,yy
        finally:graph.native_path,graph.sources=original_path,original_sources
        result={'timestamp':stamp(),'source':snapshot(__file__),'unique_prefixes':80,'semantic_groups':40,'logical_expression_aliases':160,
          'native_forward_calls':320,'generation_calls':0,'all_prefix_repeats_exact':all(r['prefix_repeat_all37_layers_exact'] for r in records),
          'same_length_future_suffix_all37_exact_pairs':sum(r['same_length_suffix_all37_layers_exact'] for r in records),
          'records':records,'fixed_predictor_reports':predictions,'directed_minus_query':paired,'cross_protocol_error_shift':protocol_shift,
          'frozen_predictor_sha256':sha(BASE/'graph/frozen.json'),'causal_anchor_audit_sha256':sha(BASE/'causal_anchor/result.json'),
          'seconds':time.monotonic()-start,'scope':'Executed prefix-only native inputs/targets and matched-length future suffix controls on all held80unique body prefixes. Prior predictors not refit, no future question/style/answer input. Reuses existing held groups, not new semantic confirmation; original training banks still retain their recorded full-prompt numerical protocol.'}
        assert result['seconds']<600
        save(out/'result.json',result);ledger('held_prefix_only_native_and_frozen_forecast',result['seconds'])
        print('CAUSAL_REPLAY_DONE',result['same_length_future_suffix_all37_exact_pairs'],result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if observer is not None:observer.close()
        model=None;observer=None;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
