"""Same causal prefix, different later instruction: all-layer numerical audit."""
from collections import defaultdict
from rdc_update_common import *


def main():
    start=time.monotonic();out=BASE/'causal_anchor';rows=gzread(BASE/'language_material.json.gz')
    config_path=ROOT/'models/hf/qwen3-4b/config.json';config=read(config_path)
    assert config['model_type']=='qwen3' and config.get('rope_scaling') is None
    assert not config.get('use_sliding_window') and max(len(r['prompt_ids']) for r in rows)<config['max_position_embeddings']
    lookup={(r['source_group'],r['language'],r['answer_style']):r for r in rows}
    pairs=[];relative=[];equal=[];maxabs=[];prefix_length=[];lengths=[];evidence={}
    for row in rows:
        if row['answer_style']!='direct':continue
        other=lookup[row['source_group'],row['language'],'explain'];p,q=row['anchors'][0],other['anchors'][0]
        assert row['body']==other['body'] and row['question']==other['question']
        ids=row['prompt_ids'][:p+1];assert ids==other['prompt_ids'][:q+1] and p==q
        assert row['prompt_ids']!=other['prompt_ids']
        hh=[]
        for r in (row,other):
            path=BASE/'language_capture/fields'/f'{r["sample_id"]}.npz'
            digest=read(BASE/'language_capture/commits'/f'{r["sample_id"]}.json')['array_sha256']
            assert sha(path)==digest;evidence[path.relative_to(BASE).as_posix()]=digest
            with np.load(path,allow_pickle=False) as z:
                assert z['token_ids'].tolist()==r['prompt_ids']
                hh.append(z['H'][:,0].copy())
        same=(hh[0]==hh[1]).all(-1);assert same[0]
        a,b=[unbits(v).astype(float) for v in hh];difference=b-a
        rel=np.sqrt((difference*difference).sum(-1)/np.maximum((a*a).sum(-1),1e-30))
        relative.append(rel);equal.append(same);maxabs.append(abs(difference).max(-1));prefix_length.append(len(ids));lengths.append([len(row['prompt_ids']),len(other['prompt_ids'])])
        pairs.append({'direct_id':row['sample_id'],'explain_id':other['sample_id'],'source_group':row['source_group'],
          'language':row['language'],'family':row['family'],'split':row['split'],'same_prefix_token_IDs':True,
          'prefix_tokens':len(ids),'prefix_ID_sha256':identity(np.array(ids,np.int32))['sha256'],
          'full_input_lengths':[len(row['prompt_ids']),len(other['prompt_ids'])]})
    assert len(pairs)==320
    relative=np.array(relative);equal=np.array(equal);maxabs=np.array(maxabs);summaries=[]
    strata=defaultdict(list)
    for i,r in enumerate(pairs):strata[r['family'],r['language']].append(i)
    for (family,language),ix in sorted(strata.items()):
        for layer in range(37):
            values=relative[ix,layer]
            summaries.append({'family':family,'language':language,'layer':layer,'pairs':len(ix),
              'same_body_H_bit_exact':int(equal[ix,layer].sum()),'relative_RMS_mean':float(values.mean()),
              'relative_RMS_range':[float(values.min()),float(values.max())],
              'max_absolute_coordinate_difference':float(maxabs[ix,layer].max())})
    npz(out/'all_pair_all_layer_differences.npz',relative_RMS=relative,bit_equal=equal,
      max_absolute_coordinate_difference=maxabs,prefix_tokens=np.array(prefix_length),full_input_lengths=np.array(lengths))
    compressed(out/'pair_identity.json.gz',pairs)
    result={'timestamp':stamp(),'source':snapshot(__file__),'pairs':320,'semantic_groups':160,
      'all320_causal_prefix_IDs_exact':True,'all320_embedding_body_states_bit_exact':bool(equal[:,0].all()),
      'full_coordinate_count_per_pair':37*2560,'summaries':summaries,
      'H12_mean_relative_RMS':float(relative[:,12].mean()),'H12_max_relative_RMS':float(relative[:,12].max()),
      'H12_bit_exact_pairs':int(equal[:,12].sum()),'material_sha256':sha(BASE/'language_material.json.gz'),
      'unchanged_capture_archive_sha256':evidence,'seconds':time.monotonic()-start,
      'native_position_config':{k:config.get(k) for k in ('model_type','rope_scaling','rope_theta','max_position_embeddings','use_sliding_window')},
      'native_position_config_sha256':sha(config_path),
      'known_causal_boundary':'The direct/explain instruction follows the body/query anchor. In this configured Qwen3 fixed-RoPE, declared-range ideal causal computation, identical earlier token prefixes cannot semantically depend on a later style instruction.',
      'observed_protocol':'Stored fields came from independent B1, no-padding, full-prompt BF16 forwards with different full sequence lengths, not from a common truncated-prefix execution shape.',
      'interpretation':'Earlier body differences must not be called a causal style effect. Execution shape/finite precision and collection alignment are control issues; this archive audit alone does not isolate their numerical contributions.',
      'not_executed':'No new640case truncated-prefix GPU replay in this post-observation audit. A strictly deployable available-prefix forecast still needs matching prefix execution shape; later question/style effects are studied at the final instruction anchor.',
      'scope':'Every coordinate and all37body-anchor layers used, no Top-K. Post-observation same-material numerical/causal-position audit; no new independent semantic confirmation, predictor refit or changed input/output.'}
    if (out/'result.json').exists():
        old=read(out/'result.json')
        for k in ('H12_bit_exact_pairs','H12_mean_relative_RMS','H12_max_relative_RMS'):assert old[k]==result[k]
        save(out/'prior_receipts'/f'{time.time_ns()}.json',old)
    save(out/'result.json',result);ledger('same_body_prefix_all_layer_audit',result['seconds'])
    print('CAUSAL_ANCHOR_AUDIT',result['pairs'],result['H12_bit_exact_pairs'],result['H12_mean_relative_RMS'],result['H12_max_relative_RMS'],flush=True)


if __name__=='__main__':main()
