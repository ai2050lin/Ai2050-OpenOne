"""Serial CUDA capture and frozen signed-moment prediction on unseen material."""
import gc
from rdc_binding_common import *

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    from phase2735_rdc_binding_signed import evaluate
    out=BASE/'signed_source';frozen=read(out/'frozen.json');rows=signed_rows()
    assert sha(out/'natural_material.json.gz')==frozen['material_sha256']
    start=time.monotonic()
    if not (out/'capture_result.json').exists():
        guard(300*1024**2);model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
        observer=Observer(model);device=model.get_input_embeddings().weight.device;records=[]
        fixtures={next(r['sample_id'] for r in rows if r['cohort']==cohort and r['split']==split)
          for cohort in ('gum','ewt') for split in ('signed_connected','signed_matched')}
        save(out/'runtime.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'dtype':str(model.dtype),
          'quantized':bool(getattr(model,'is_quantized',False)),'device_map':getattr(model,'hf_device_map',{}),
          'batch':1,'padding':False,'cache':False,'model':'qwen3-4b','attention':'native eager','fixtures':sorted(fixtures)})
        try:
          with torch.inference_mode():
            for i,row in enumerate(rows):
                sid=row['sample_id'];commit=out/'commits'/f'{sid}.json';path=out/'fields'/f'{sid}.npz'
                if commit.exists():
                    r=read(commit);assert sha(path)==r['array_sha'];records.append(r);continue
                tick=time.monotonic();ids=torch.tensor([row['prompt_ids']],device=device)
                assert tok(row['text'],add_special_tokens=False)['input_ids']==row['prompt_ids']
                observer.reset(row['anchors'],sid in fixtures,i==0);observer.enabled=True
                post=model.model(input_ids=ids,use_cache=False).last_hidden_state;observer.enabled=False
                field={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,
                  'embedding':observer.H[0],**observer.factors,'token_ids':np.array(row['prompt_ids'],np.int32),
                  'positions':np.array(row['anchors']),'postnorm':bits(post[0,row['anchors']]),
                  'alltoken_layer_moments':np.stack([observer.moments[k] for k in range(37)])}
                r={k:row[k] for k in ('sample_id','source_group','cohort','split')};r['checks']=observer.checks
                r['all_token_layer_identities']=observer.H_hashes
                if i==0:
                    plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
                    assert torch.equal(plain,post);r['observer_matched_shape_exact_noop']=True;del plain
                if sid in fixtures:npz(out/'full_fields'/f'{sid}.npz',H=np.stack([observer.full_H[k] for k in range(37)]))
                npz(path,**field);r.update(array_sha=sha(path),seconds=time.monotonic()-tick,full_fixture=sid in fixtures)
                save(commit,r);records.append(r);observer.reset([],False);del ids,post,field
                if (i+1)%16==0:print('SIGNED_SOURCE_NATIVE_CAPTURE',i+1,len(rows),flush=True)
                guard(30*1024**2);assert time.monotonic()-start<7200
        finally:
            observer.close();del model,observer;gc.collect();torch.cuda.empty_cache()
        save(out/'capture_result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'rows':len(records),
          'frozen_predictors_sha256':sha(out/'frozen.json'),'seconds':time.monotonic()-start,
          'scope':'Untouched BF16 CUDA model; complete37layer anchor field and allH12source coordinates. Four declared full-layer/alltoken fixtures.'})
        ledger('signed_source_native_confirmation',time.monotonic()-start)
    evaluate()

if __name__=='__main__':main()
