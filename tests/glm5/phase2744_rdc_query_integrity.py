"""Independent all-pair, deployed-model and scorer audit for the continuation."""
from phase2744_rdc_query_identifiability import *


def audit():
    assert read(OUT/'queue/status.json')['all_passed']
    protocol=read(OUT/'protocol.json');material=gzread(OUT/'material.json.gz');rows=material['controlled'];checks=[]
    lookup={r['sample_id']:r for r in rows};assert len(lookup)==320
    assert len(material['pairs'])==160 and len({r['source_group'] for r in rows})==80
    for pair in material['pairs']:
        a,b=[lookup[s] for s in pair['sample_ids']]
        assert Counter(a['prompt_ids'])==Counter(b['prompt_ids'])
        assert a['question']==b['question'] and a['target']!=b['target'] and a['truth']!=b['truth']
    checks.append({'check':'Every complete actual-token multiset pair retains the same question and opposite external recipe answer','pairs':160,'semantic_groups':80})
    frozen=read(BASE/'followup/protocol.json')['unchanged_main_decoder_sha256']
    assert frozen==sha(BASE/'rules/decoder.npz')
    for stage in ['relations','calibration','behavior','analysis']:
        assert read(OUT/stage/'result.json')['all_passed'],stage
    for stage in ['relations','calibration']:
        result=read(OUT/stage/'result.json')
        assert result['variants']==VARIANTS and len(result['deployment_checks'])==5
        assert all(c['first4_original_formation_losses_exact'] and c['max_NLL_error']==0 for c in result['deployment_checks'])
    checks.append({'check':'All actual deployed parameter variants reproduce the original four formation losses exactly in both native capture stages','variants':5,'stages':2})
    cfg=read(ROOT/'models/hf/qwen3-4b/generation_config.json');stop=cfg['eos_token_id'];stop=set(stop if isinstance(stop,list) else [stop])
    bit_checks=0;trajectory_count=0;token_count=0
    for variant in VARIANTS:
        root=OUT/'relations'/variant
        commits=list((root/'commits').glob('*.json'));assert len(commits)==320
        for cp in commits:
            r=read(cp);row=lookup[r['sample_id']];fp=root/'fields'/cp.with_suffix('.npz').name
            assert r['variant']==variant and r['pair_id']==row['pair_id'] and r['sha256']==sha(fp)
            assert r['prefix_cache_not_modified'];bit_checks+=r['whole_prefix_cache_bitchecked']
            with np.load(fp) as z:
                assert np.array_equal(z['matched_query_indices'],QUERIES)
                assert z['matched_subset_postnorm'].shape==(6,2560)
                assert z['postnorm_original_prompt'].shape==(2560,)
                for block in [16,35]:
                    for field in ['gate_proj','up_proj','activation']:assert z[f'L{block}_{field}'].shape==(9728,)
                if variant=='native':
                    assert z['prefix_layers'].shape==(37,2560) and z['postnorm'].shape==(100,2560)
                    assert z['frozen_rule_all_query_MSE'].shape==z['frozen_rule_all_query_KL'].shape==(5,100)
                else:assert z['prefix_selected_layers_H16_H17_H36'].shape==(3,2560)
            cp2=OUT/'behavior'/variant/'commits'/cp.name;behavior=read(cp2)
            assert behavior['variant']==variant and behavior['pair_id']==row['pair_id']
            assert 1<=len(behavior['generated_ids'])<=128
            assert language_score(row,behavior['generated_text'],behavior['generated_ids'],stop,128)==behavior['answer_scoring']
            batch=behavior['batch_ids'];index=next(i for i,item in enumerate(rows) if item['sample_id']==row['sample_id'])
            assert batch==[x['sample_id'] for x in rows[index//8*8:index//8*8+8]] and batch[behavior['batch_row']]==row['sample_id']
            fp2=OUT/'behavior'/variant/'fields'/cp.with_suffix('.npz').name
            assert sha(fp2)==behavior['field_sha256']
            with np.load(fp2) as z:assert z['first_and_final_postnorm'].shape in [(1,2560),(2,2560)]
            trajectory_count+=1;token_count+=len(behavior['generated_ids'])
        root=OUT/'calibration'/variant;cp=read(root/'result.json')
        assert cp['field_sha256']==sha(root/'all_fields.npz') and cp['examples']==384
        observations=gzread(root/'observations.json.gz')
        assert [r['sample_id'] for r in observations]==[r['sample_id'] for r in material['natural']]
        with np.load(root/'all_fields.npz') as z:
            assert z['postnorm'].shape==(384,2560) and z['temperature_prior_mixture_NLL'].shape==(384,7,6)
            assert np.array_equal(z['temperature_prior_mixture_NLL'][:,2,0],[r['native_NLL'] for r in observations])
    assert bit_checks==50 and trajectory_count==1600
    checks.append({'check':'All320expressions times5actual models: native response axes, all9728units and independent own-history rescoring','trajectories':trajectory_count,'generated_tokens':token_count,'whole_prefix_cache_bitchecked':bit_checks})
    parent={r['sample_id']:r for r in [*gzread(BASE/'material/natural.json.gz'),*gzread(BASE/'followup/material.json.gz')]}
    splits=defaultdict(set)
    for row in material['natural']:
        p=parent[row['parent_id']];pos=row['position']
        assert row['ids']==p['prompt_ids'][:pos+1] and row['target']==p['prompt_ids'][pos+1]
        splits[row['source_group']].add(row['split'])
    assert all(len(v)==1 for v in splits.values())
    assert len({r['source_group'] for r in material['natural'] if r['split']=='prospective_natural'})==96
    expected=Counter(r['target'] for r in gzread(BASE/'formation/material.json.gz')['train'])
    assert {int(k):v for k,v in material['train_token_counts'].items()}==expected
    checks.append({'check':'Calibration and prospective documents do not overlap, targets are the actual next tokens, and fixed vocabulary prior uses only original training labels','prospective_positions':192,'prospective_documents':96})
    return {'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
      'scope':'Metadata, axes, original-loss reconstruction and independent scoring verification; not a proof of semantic identifiability or a universal language law.'}


if __name__=='__main__':
    start=time.monotonic();result=audit();result['seconds']=time.monotonic()-start
    save(OUT/'verification.json',result);ledger('identity_independent_integrity',result['seconds']);print('IDENTITY_INTEGRITY_PASS',result['seconds'],flush=True)
