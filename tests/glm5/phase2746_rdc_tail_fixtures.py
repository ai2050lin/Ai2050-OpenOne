"""Same-history all-layer KV fixtures for remaining-network differential tests."""
import argparse
from collections import Counter
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_runtime_observer import Observer
from rdc_native_tail import block_call,cuda_singleton,CUDA_TASKS

OUT=BASE/'phase2746/differential'
STARTS=[12,24,35]


def freeze():
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json'),gzread(OUT/'material.json.gz')
    raw=gzread(BASE/'phase2746/runtime/material.json.gz')
    runs={r['sample_id']:r for r in gzread(BASE/'phase2746/runtime/records.json.gz')}
    selected=[]
    for family in sorted({r['family'] for r in raw}):
        items=[r for r in raw if r['family']==family]
        groups=sorted({r['source_group'] for r in items},key=lambda s:rank('tail2746/'+s))
        if items[0]['kind']=='natural':
            selected += [min([r for r in items if r['source_group']==s],key=lambda r:rank(r['sample_id'])) for s in groups[:12]]
        else:
            selected += [r for r in items if r['source_group'] in groups[:3]]
    assert len(selected)==96 and all(n==12 for n in Counter(r['family'] for r in selected).values())
    rows=[]
    for r in selected:
        for step in [0,1]:
            row={k:r[k] for k in ['sample_id','source_group','family','kind','split','original_text']}
            row.update(fixture_id=r['sample_id']+'_t'+str(step),step=step,
                prompt_ids=r['prompt_ids']+runs[r['sample_id']]['generated_ids'][:step],
                history_reference_sha256=runs[r['sample_id']]['field_sha256'],
                old_history_token_ids=runs[r['sample_id']]['generated_ids'][:step])
            rows.append(row)
    pilot_ids=[next(r['fixture_id'] for r in rows if r['family']==f and r['step']==0) for f in sorted({r['family'] for r in rows})]
    compressed(OUT/'material.json.gz',rows)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'phase':2746,'model':'qwen4',
        'question':'Can complete remaining-network derivatives and native-parameter directions explain finite conditioned cross-layer response changes?',
        'source_rows':96,'same_history_endpoints':192,'families':dict(Counter(r['family'] for r in rows)),
        'selection':'Hash-selected12sources per cohort or3semanticgroups times2languages times2worlds per controlled family; no outcome or activation selection.',
        'selection_scope':'Existing exposed discovery material, not new independent confirmation; steps0and1 use originalB8history but B1 replay generates its own measured fixture.',
        'starts':STARTS,'directions':['seeded_dense_Rademacher','native_previous_MLP_write','one_hash_selected_native_down_column'],
        'direction_scale':'Each completeDdirection rescaled to RMS(x_start). No parameter or coordinate magnitude ranking.',
        'relative_epsilons':[.001,.01,.1],
        'smooth_reference':'Identical BF16 checkpoint scalar values represented in FP32; nativeQwen modules and fixed nativeBF16cos/sin/prefixKV promoted to FP32. Not an assertion of bit-exact BF16 differentiation.',
        'complete_tail':'All intervening blocks including dynamic current-query Q/K/V, softmax, residuals, both RMSNorms, SiLU products, finalnorm and full vocabulary projection. Only earlier-token prefixKV frozen.',
        'VJP_JVP':'Exact AD products without materializing fullDbyD Jacobian. Check adjoint identity and symmetric finite differences at all3eps. Report full-coordinate norm/gain/cosine and nativeBF16 finite-change floor separately.',
        'controls':'Bare final-readout compatibility and identity residual transport are controls, not accepted substitutes for remaining-network propagation.',
        'native_numerical_check':'Loaded original BF16 tail fixtures must reproduce their complete native tail postnorm vectors bit-for-bit; fullprefill versus tail execution-shape differences are measured separately.',
        'forecast_scope':'Directions and current state available at injection. Native-previous-write is an observed local input, not an earlier-layer prediction; no future state/token or gold used to construct directions.',
        'pilot_fixture_ids':pilot_ids,'material_sha256':sha(OUT/'material.json.gz'),
        'resource':'Sequential one CUDA model. All-layerKV stored once per fixture using registered physicalCarchive; stream one native parameter block at a time for FP32 derivatives.'}
    immutable(OUT/'protocol.json',protocol);return protocol,rows


def main(pilot):
    import torch
    cuda_singleton(CUDA_TASKS);verify_storage(8*1024**3);start=time.monotonic()
    protocol,rows=freeze();selected=[r for r in rows if r['fixture_id'] in protocol['pilot_fixture_ids']] if pilot else rows
    model=None;observer=None;handles=[];writes={}
    execution={'source':snapshot(__file__),'helper':snapshot(Path(__file__).with_name('rdc_native_tail.py')),
        'protocol_sha256':sha(OUT/'protocol.json')}
    try:
        model,tok=load('qwen4',OUT/'native_loader');observer=Observer(model)
        for index in STARTS:
            handles.append(model.model.layers[index-1].mlp.register_forward_hook(lambda m,a,o,i=index:writes.__setitem__(i,bits(o[0,-1]))))
        records=[]
        with torch.inference_mode():
            for row in selected:
                receipt=OUT/'commits'/(row['fixture_id']+'.json')
                if receipt.exists():
                    rec=read(receipt);assert rec['execution']==execution and sha(BASE/rec['field_path'])==rec['field_sha256']
                    records.append(rec);continue
                ids=torch.tensor([row['prompt_ids']],device='cuda');observer.reset(False);writes.clear()
                value=model.model(input_ids=ids,use_cache=True);states=observer.collect()['hidden'][0]
                cache=value.past_key_values;post=bits(value.last_hidden_state[0,-1])
                previous=np.stack([writes[i] for i in STARTS]);observer.active=False
                p=torch.tensor([[len(row['prompt_ids'])-1]],device='cuda')
                cos,sin=model.model.rotary_emb(torch.zeros((1,1,2560),device='cuda',dtype=torch.bfloat16),p)
                keys=np.stack([bits(l.keys[0,:,:-1]) for l in cache.layers])
                values=np.stack([bits(l.values[0,:,:-1]) for l in cache.layers])
                native=[];shape=[]
                for begin in STARTS:
                    h=torch.from_numpy(unbits(states[begin]).copy()).to(device='cuda',dtype=torch.bfloat16)[None,None]
                    for block in range(begin,36):
                        h=block_call(model.model.layers[block],h,cache.layers[block].keys[:,:,:-1],cache.layers[block].values[:,:,:-1],cos,sin,block)
                    out=model.model.norm(h)[0,-1];native.append(bits(out))
                    shape.append({'start':begin,'postnorm_MSE_vs_fullprefill':float(np.mean((unbits(native[-1]).astype(float)-unbits(post).astype(float))**2)),
                        'bit_equal_to_fullprefill':bool(np.array_equal(native[-1],post))})
                arrays={'hidden':states,'postnorm':post,'prefix_keys':keys,'prefix_values':values,
                    'cos':bits(cos),'sin':bits(sin),'starts':np.array(STARTS),
                    'native_previous_MLP_writes':previous,'native_BF16_tail_postnorm':np.stack(native)}
                verify_storage(sum(a.nbytes for a in arrays.values()))
                path=FIELD_STORE/'differential'/(row['fixture_id']+'.npz');npz(path,**arrays)
                rec={**row,'timestamp':stamp(),'execution':execution,'field_path':path.relative_to(BASE).as_posix(),
                    'field_sha256':sha(path),'field_bytes':path.stat().st_size,'native_shape_controls':shape,
                    'input_text':tok.decode(row['prompt_ids'],skip_special_tokens=False)}
                save(receipt,rec);records.append(rec)
                del value,cache,keys,values,h,out,ids
                if len(records)%8==0:print('TAIL_FIXTURES',len(records),len(selected),round(time.monotonic()-start,1),flush=True)
        if not pilot:
            assert len(records)==192;compressed(OUT/'fixtures.json.gz',records)
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'fixtures':len(records),
            'execution':execution,'field_bytes':sum(r['field_bytes'] for r in records),'seconds':time.monotonic()-start,
            'maximum_native_tail_vs_prefill_MSE':max(s['postnorm_MSE_vs_fullprefill'] for r in records for s in r['native_shape_controls']),
            'scope':'Original same-history native fixtures, not learned predictor results; derivative/finite checks not yet executed.'}
        save(OUT/('fixture_pilot.json' if pilot else 'fixture_result.json'),result)
        ledger('phase2746_tail_fixtures',result['seconds'],pilot=pilot)
        print('TAIL_FIXTURES_COMPLETE',len(records),result['field_bytes'],flush=True)
    except Exception as exc:
        failure(OUT,start,exc);raise
    finally:
        for handle in handles:handle.remove()
        if observer:observer.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
