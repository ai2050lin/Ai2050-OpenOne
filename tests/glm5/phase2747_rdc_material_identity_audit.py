"""Independent full material/trace checks, not just recorded completion flags."""
from collections import Counter,defaultdict
from itertools import combinations
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays


def main():
    start=time.monotonic();protocol=read(OUT/'material/protocol.json')
    material=OUT/'material/rows.json.gz';assert sha(material)==protocol['material_sha256']
    data=gzread(material);checks=[];counts={k:len(v) for k,v in data.items()}
    assert counts==protocol['counts']=={'train':2048,'validation':192,'diagnostic':512,'fresh':192}
    natural=[];control=[]
    for split,rows in data.items():
        assert len({r['sample_id'] for r in rows})==len(rows)
        for r in rows:
            if r['kind']=='natural_content':
                original=r['source_input_ids'];position=r['position']
                assert r['ids']==original[:position+1] and r['target']==original[position+1]
                assert position+1<len(original);natural.append(r)
            else:
                assert r['kind']=='controlled_relation' and r['ids']==r['prompt_ids'];control.append(r)
    group_sets={k:{r['source_group'] for r in v} for k,v in data.items()}
    for a,b in combinations(group_sets,2):assert not group_sets[a]&group_sets[b],(a,b)
    excluded=gzread(OUT/'material/excluded_inventory.json.gz')
    excluded_groups={r['source_group'] for r in excluded}
    excluded_inputs={tuple(r['prompt_ids']) for r in excluded if r.get('prompt_ids') is not None}
    excluded_components={c for r in excluded for c in (r.get('component_ids') or [])}
    assert not group_sets['fresh']&excluded_groups
    for r in data['fresh']:
        assert tuple(r['source_input_ids']) not in excluded_inputs
        assert not set(r.get('component_ids',[]))&excluded_components
    checks.append('Every authentic nexttoken/prefix boundary, allsplit sourcegroup exclusions and each freshsource/component/fullinput exclusion recomputed from the frozen material')
    vocab=checked_arrays(protocol['vocabulary_receipt']);train=data['train'];classes=vocab['classes'];permutation=vocab['permutation']
    assert sorted(permutation.tolist())==list(range(2048))
    def key(r):return r['kind'],r['cohort'],r['language'],int(r['surface_class'])
    bins=defaultdict(list)
    for i,r in enumerate(train):
        other=train[int(permutation[i])]
        assert key(r)==key(other) and r['permuted_target']==other['target']
        assert classes[r['target']]==classes[r['permuted_target']]==r['surface_class']
        bins[key(r)].append(r)
    for rr in bins.values():assert Counter(r['target'] for r in rr)==Counter(r['permuted_target'] for r in rr)
    changed=sum(r['target']!=r['permuted_target'] for r in train)
    assert changed/len(train)==protocol['label_permutation_changed_fraction']
    checks.append('Full2048permutation is bijective within kind/cohort/language/class; exact empirical target histograms retained perbin, including unchanged-label coincidences')
    tprotocol=read(OUT/'training/protocol.json');result=read(OUT/'training/result.json');traces={};trace_checks=[]
    assert result['all_passed'] and len(result['runs'])==6
    for run in result['runs']:
        trace=run['trace'];assert [r['step'] for r in trace]==list(range(1,129))
        assert all(len(r['examples'])==16 and r['nominal_FP32_step_norm']==tprotocol['step_FP32_norm'] for r in trace)
        drawn=[i for r in trace for i in r['examples']]
        assert sorted(drawn)==list(range(2048))
        assert [r['step'] for r in run['checkpoints']]==[1,8,32,128]
        assert all(np.isfinite([r['objective'],r['full_gradient_norm']]).all() and r['full_gradient_norm']>0 for r in trace)
        order=np.random.default_rng(run['seed']).permutation(2048).tolist()
        assert order==drawn,('Actual trace order differs from frozen seeded permutation',run['condition'],run['seed'])
        traces[(run['condition'],run['seed'])]=drawn
        trace_checks.append({'condition':run['condition'],'seed':run['seed'],'actual_update_records':len(trace),
            'actual_examples':len(drawn),'each_material_index_seen_exactly_once':True,'seeded_order_recomputed':True})
    for seed in tprotocol['seeds']:
        orders=[traces[(condition,seed)] for condition in tprotocol['conditions']]
        assert all(v==orders[0] for v in orders)
    assert traces[(tprotocol['conditions'][0],2747)]!=traces[(tprotocol['conditions'][0],2748)]
    checks.append('All768actual update traces and12288example indices independently reconstructed; same seeded order across3conditions, distinct orders across2seeds, no unused advertised pool')
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
        'material_counts':counts,'authentic_natural_positions':len(natural),'controlled_positions':len(control),
        'fresh_source_groups':len(group_sets['fresh']),'permutation_bins':len(bins),'changed_permuted_labels':changed,
        'traces':trace_checks,'material_sha256':sha(material),'training_result_sha256':sha(OUT/'training/result.json'),
        'seconds':time.monotonic()-start,
        'scope':'Evidence identity and executed data consumption, not proof of label semantics, absence from pretraining or everyhistoricalPhase, or full128parameter snapshots.'}
    save(OUT/'verification/material_trace_identities.json',value);ledger('phase2747_material_trace_audit',value['seconds'])
    print('FORMATION_MATERIAL_TRACE_IDENTITIES',len(natural),len(control),len(trace_checks),changed,flush=True)


if __name__=='__main__':main()
