"""Frozen token-order controls for genuine native last-MLP continuation training."""
import gc
from rdc_law_common import *
from phase2728_rdc_law_formation import protocol
from phase2728_rdc_law_capture import Observer, persist


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    p=protocol();out=BASE/'formation/controlled_capture'
    if (out/'result.json').exists():return
    assert (BASE/'capture/main/result.json').exists()
    start=time.monotonic();guard(800*1024**2)
    source=snapshot(Path(__file__))
    model,tok=load_native('qwen4');observer=Observer(model);device=model.get_input_embeddings().weight.device
    records=[]
    try:
      with torch.inference_mode():
        for i,row in enumerate(p['controlled_views']):
            sid=row['sample_id'];positions=row['training_positions']
            observer.reset(positions,False,i==0);observer.enabled=True
            ids=torch.tensor([row['controlled_ids']],device=device)
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            observer.enabled=False
            packet={'H':np.stack([observer.H[k] for k in range(37)]),'postnorm':bits(post[0,positions]),
                'x':observer.training['x'][positions],'residual':observer.training['residual'][positions],
                'native_output':observer.training['native_output'][positions],
                'H12_sources':observer.source_H12,'positions':np.array(positions),
                'targets':np.array(row['target_ids'],dtype=np.int32),'controlled_ids':np.array(row['controlled_ids'],dtype=np.int32),
                'order':np.array(row['order'],dtype=np.int32)}
            for key,a in observer.factors.items():packet[key]=a
            for b,a in observer.attention.items():packet[f'L{b}_attention_sources']=a
            checks=dict(observer.checks)
            if i==0:
                # Same-shape original input must reproduce the independently captured frozen prefix.
                observer.reset(positions,False);observer.enabled=True
                original=model.model(input_ids=torch.tensor([row['original_ids']],device=device),use_cache=False).last_hidden_state
                observer.enabled=False
                with np.load(BASE/'capture/main/sources'/f'{sid}.npz') as z:
                    assert np.array_equal(observer.training['x'],z['x'])
                    assert np.array_equal(observer.training['residual'],z['residual'])
                    assert np.array_equal(observer.training['native_output'],z['native_output'])
                checks['original_full_source_x_residual_output_exact']=True
                del original
            persist(out/'fields'/f'{sid}.npz',packet)
            record={'sample_id':sid,'split':row['split'],'source_group':row['source_group'],'positions':len(positions),
                'tokens':len(row['controlled_ids']),'field_sha':sha(out/'fields'/f'{sid}.npz'),'checks':checks,
                'all_coordinate_all_unit':True,'unmodified_targets_and_suffix':True}
            save(out/'commits'/f'{sid}.json',record);records.append(record)
            observer.reset([],False);del post,packet,ids
            if i<2 or (i+1)%16==0:print('LAW_CONTROL_CAPTURE',i+1,len(p['controlled_views']),'elapsed',round(time.monotonic()-start,1),flush=True)
            guard(64*1024**2)
    except Exception as exc:
        import traceback
        save(out/'failure.json',{'timestamp':stamp(),'source':source,'error':str(exc),'traceback':traceback.format_exc(),'completed':len(records),'seconds':time.monotonic()-start})
        ledger('failed_control_capture',time.monotonic()-start)
        raise
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'source':source,'rows':len(records),'positions':sum(r['positions'] for r in records),
        'tokens':sum(r['tokens'] for r in records),'checks':[r for r in records if r['checks']],'seconds':time.monotonic()-start,
        'scope':'Actual native forward on a deliberately unnatural early-prefix order control. All target token IDs and suffix input IDs match coherent counterparts. Both conditions get their own exact upstream hidden states, not donor patching.'}
    save(out/'result.json',result);ledger('controlled_prefix_native_capture',result['seconds'],rows=len(records))
    print('LAW_CONTROL_CAPTURE_COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
