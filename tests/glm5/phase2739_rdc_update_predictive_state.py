"""Reachable token-prefix probes and finite native KV bookkeeping."""
import gc
from rdc_update_common import *

def freeze():
    out=BASE/'predictive_state'
    if (out/'protocol.json').exists():return read(out/'protocol.json')
    rows=gzread(BASE/'natural_material.json.gz');features=[]
    for row in rows:
        p=row['anchors'][0]
        with np.load(native_path(row)) as z:
            h=unbits(z['H12_sources'])[:p+1];e=unbits(z['embedding'])[0]
        u=h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8)
        features.append(np.r_[u[-1],u.mean(0),e/np.sqrt(np.mean(e*e)).clip(1e-8)])
    f=np.array(features);distance=np.mean((f[:,None,:]-f[None,:,:])**2,axis=-1);pairs=[];centers=[]
    for cohort in ('gum','ewt'):
        centers.extend(sorted([i for i,r in enumerate(rows) if r['cohort']==cohort],key=lambda i:ranked('PSRcenter/'+rows[i]['sample_id']))[:16])
    for i in centers:
        rr=rows[i];length=rr['anchors'][0]+1
        available=[j for j,r in enumerate(rows) if r['cohort']==rr['cohort'] and r['source_group']!=rr['source_group'] and abs(r['anchors'][0]+1-length)<=8]
        allowance=8
        if len(available)<4:
            allowance=16;available=[j for j,r in enumerate(rows) if r['cohort']==rr['cohort'] and r['source_group']!=rr['source_group'] and abs(r['anchors'][0]+1-length)<=16]
        assert len(available)>=4
        order=sorted(available,key=lambda j:(distance[i,j],ranked(rows[j]['sample_id'])))
        for name,j in [('near',order[0]),('far',order[-1])]:
            pairs.append({'center':rr['sample_id'],'other':rows[j]['sample_id'],'condition':name,'source_group':rr['source_group'],
              'descriptor_squared_RMS_distance':float(distance[i,j]),'length_difference':rows[j]['anchors'][0]-rr['anchors'][0],'maximum_length_difference':allowance})
    p={'timestamp':stamp(),'source':snapshot(__file__),'pairs':pairs,'probes':['',' the',' because',' However,','\nThe'],
      'selection':'32 fixed centers, nearest/farthest actual prefix within same corpus and bounded length difference, different documents. No future logits, target labels or responses used for pair selection.',
      'descriptor':'Concatenated complete2560-coordinate H12 current vector, mean of every RMS-normalized visible H12 source, and current embedding; no PCA/Top-K. This declared finite summary is the object being tested, not claimed sufficient state.',
      'experiment':'Append the same predeclared token suffix to each actual prefix. Re-run original model on each prefix; compare complete vocabulary distributions. Fixed probe insertion is an experiment, not model free generation.',
      'KV_lengths':[512,1024,2048],'KV_input':'Known concatenation of retained authentic passages, clipped at fixed lengths; not a single authentic long document.',
      'limits':'Near is not equal; pairs reuse prefixes and are dependent. No exact reachable collision/lumpability proof, no guarantee for all continuations, no100K compression claim.',
      'full_vocab_fixtures':'All5 probe logprob vectors for first2centers and their near/far counterparts; other complete vectors compared before discarding, with exact prompt IDs and postnorm retained.'}
    immutable(out/'protocol.json',p);npz(out/'selection_full_coordinates.npz',descriptors=f,pairwise_squared_distance=distance);return p

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    p=freeze();out=BASE/'predictive_state';start=time.monotonic()
    if (out/'result.json').exists():return
    guard(70*1024**2);rows={r['sample_id']:r for r in gzread(BASE/'natural_material.json.gz')}
    ids=sorted({s for pair in p['pairs'] for s in (pair['center'],pair['other'])});fixture_centers={pair['center'] for pair in p['pairs'][:4]}
    fixtures={s for pair in p['pairs'] if pair['center'] in fixture_centers for s in (pair['center'],pair['other'])}
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);all_lp={};all_h={};records=[]
    try:
      with torch.inference_mode():
        for i,sid in enumerate(ids):
            row=rows[sid];prefix=row['prompt_ids'][:row['anchors'][0]+1];lp=[];hs=[];promptids=[]
            for probe in p['probes']:
                known=prefix+tok(probe,add_special_tokens=False)['input_ids'];promptids.append(known)
                post=model.model(input_ids=torch.tensor([known],device='cuda'),use_cache=False).last_hidden_state[0,-1]
                logits=model.lm_head(post).float();lp.append(logits.double().log_softmax(-1).cpu().numpy());hs.append(bits(post))
                del post,logits
            all_lp[sid]=np.array(lp);all_h[sid]=np.array(hs)
            if sid in fixtures:npz(out/'full_vocab_fixtures'/f'{sid}.npz',logprobs=all_lp[sid])
            npz(out/'postnorm_fields'/f'{sid}.npz',postnorm=all_h[sid])
            rec={'sample_id':sid,'source_group':row['source_group'],'cohort':row['cohort'],'actual_prefix_and_probes':promptids,
              'all_vocab_logprob_identity':identity(all_lp[sid]),'all_probe_top_ids':all_lp[sid].argmax(-1).tolist()}
            save(out/'commits'/f'{sid}.json',rec);records.append(rec)
            if (i+1)%16==0:print('REACHABLE_PREFIX_PROBE',i+1,len(ids),flush=True)
        comparisons=[]
        for pair in p['pairs']:
            a=all_lp[pair['center']];b=all_lp[pair['other']];kl=.5*((np.exp(a)*(a-b)).sum(-1)+(np.exp(b)*(b-a)).sum(-1))
            v=unbits(all_h[pair['center']]);u=unbits(all_h[pair['other']])
            comparisons.append(pair|{'symmetric_KL_by_probe':kl.tolist(),'postnorm_relative_MSE_by_probe':(np.mean((v-u)**2,axis=-1)/np.mean(v*v,axis=-1).clip(1e-8)).tolist(),
              'argmax_agrees_by_probe':(a.argmax(-1)==b.argmax(-1)).tolist()})
        del all_lp
        summary=[]
        for j,probe in enumerate(p['probes']):
            near=[r for r in comparisons if r['condition']=='near'];far=[r for r in comparisons if r['condition']=='far']
            assert [r['center'] for r in near]==[r['center'] for r in far]
            summary.append({'probe':probe,'near_mean_KL':float(np.mean([r['symmetric_KL_by_probe'][j] for r in near])),
              'far_mean_KL':float(np.mean([r['symmetric_KL_by_probe'][j] for r in far])),
              'paired_near_minus_far':clustered([a['symmetric_KL_by_probe'][j]-b['symmetric_KL_by_probe'][j] for a,b in zip(near,far)],[r['source_group'] for r in near])})
        stream=[t for r in rows.values() for t in r['prompt_ids']];kv=[]
        for length in p['KV_lengths']:
            prompt=stream[:length];tick=time.monotonic();output=model.model(input_ids=torch.tensor([prompt],device='cuda'),use_cache=True)
            cache=output.past_key_values;details=[]
            for b,layer in enumerate(cache.layers):
              for kind in ('keys','values'):
                value=getattr(layer,kind);assert value.shape[-2]==length and torch.isfinite(value).all()
                details.append({'block':b,'array':kind,'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.numel()*value.element_size(),
                  'identity':identity(bits(value)),'RMS':float(value.float().square().mean().sqrt())})
            actual=sum(r['bytes'] for r in details);expected=2*len(cache.layers)*length*model.config.num_key_value_heads*model.config.head_dim*2
            assert actual==expected
            kv.append({'length':length,'actual_KV_bytes':actual,'formula_KV_bytes':expected,'details':details,'seconds':time.monotonic()-tick,
              'native_argmax':int(model.lm_head(output.last_hidden_state[0,-1]).float().argmax())})
            compressed(out/f'KV_input_{length}.json.gz',prompt);del cache,output;torch.cuda.empty_cache()
        result={'timestamp':stamp(),'source':snapshot(__file__),'actual_prefixes':len(ids),'pairs':len(comparisons),'probes':p['probes'],'comparisons':comparisons,'summary':summary,
          'native_KV':kv,'seconds':time.monotonic()-start,
          'scope':'Observed similarity-versus-future-distribution test on actual model prefixes and fixed continuations. No exact collision proof or lossless memory extraction. Full cached arrays audited and released; complete input/config/identities allow bounded recomputation.'}
        save(out/'result.json',result);ledger('reachable_predictive_state_and_KV',result['seconds']);print('PREDICTIVE_STATE_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
