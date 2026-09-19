"""Bit-preserved full-source H12/H23 and six-position output checkpoints on native CUDA."""
import argparse,gc,sys
from rdc_relation_common import *


class Observer:
    def __init__(self,model):
        self.enabled=False;self.positions=[];self.data={};self.handles=[]
        for layer in (12,23,24,36):
            def hook(m,a,o,layer=layer):
                if self.enabled:self.data['h'+str(layer)]=bits(o[0] if layer in (12,23) else o[0,self.positions])
            self.handles.append(model.model.layers[layer-1].register_forward_hook(hook))
        def norm(m,a,o):
            if self.enabled:self.data['postnorm']=bits(o[0,self.positions])
        self.handles.append(model.model.norm.register_forward_hook(norm))
    def close(self):
        for h in self.handles:h.remove()


def finish(fresh,pilot=False):
    rr=rows(fresh);out=BASE/('fresh' if fresh else 'main');commits=sorted((out/'commits').glob('*.json'))
    if pilot:
        sizes=sum((out/f'fields/{r["sample_id"]}.npz').stat().st_size for r in rr[:4]);n=sum(len(r['prompt_ids']) for r in rr[:4])
        total_tokens=sum(len(r['prompt_ids']) for r in rows(False)+rows(True));reserve=280*1024**2
        estimate=sizes/n*total_tokens+reserve+usage();seconds=np.mean([read(p)['seconds'] for p in commits[:4]])*640
        passed=bool(estimate<CEILING and shutil.disk_usage(ROOT).free-(estimate-usage())>FLOOR and seconds<5400)
        save(BASE/'pilot_audit.json',{'timestamp':stamp(),'passed':passed,'pilot_units':4,'pilot_file_bytes':sizes,'pilot_tokens':n,
          'bytes_per_token_including_six_anchor_fields':sizes/n,'total_source_tokens_main_fresh':total_tokens,
          'capture_plus_all_analysis_estimated_bytes':estimate,'analysis_native_generation_scale_client_reserve_bytes':reserve,
          'preexisting_new_campaign_bytes':usage(),'estimated_capture_seconds':float(seconds),'ceiling':CEILING,
          'note':'All main/fresh field estimate includes H12/H23 at every token and six H24/H36/postnorm, not all layers at all positions.'})
        assert passed,('Pilot resource gate',estimate,CEILING)
    if len(commits)==len(rr):
        groups=['en/train','zh/train','en/validation','zh/validation','en/test','zh/test','en/confirmation','zh/confirmation']
        sums=np.zeros((8,2,2560),np.float64);squares=np.zeros_like(sums);counts=np.zeros(8,np.int64)
        for r in rr:
            g=groups.index(r['language']+'/'+r['split'])
            with np.load(out/f'fields/{r["sample_id"]}.npz') as z:
                for j,k in enumerate(('h12','h23')):
                    h=unbits(z[k]).astype(np.float64);sums[g,j]+=h.sum(0);squares[g,j]+=(h*h).sum(0)
                counts[g]+=len(r['prompt_ids'])
        npz(out/'all_token_moments.npz',sums=sums,squares=squares,counts=counts)
        save(out/'capture_result.json',{'timestamp':stamp(),'units':len(rr),'tokens':int(counts.sum()),'groups':groups,
          'full_source_layers':[12,23],'six_anchor_layers':[24,36,'postnorm'],'native_width':2560,'all_commits_passed':True,
          'material_sha':sha(BASE/('fresh_material.json' if fresh else 'material.json'))})
    status('fresh' if fresh else 'main',state='captured' if len(commits)==len(rr) else 'pilot_complete',commits=len(commits),total=len(rr))


def main(pilot=False,fresh=False):
    import torch
    rr=rows(fresh);out=BASE/('fresh' if fresh else 'main')
    if fresh:assert (BASE/'frozen.json').exists(),'Freeze before fresh model outputs'
    elif not pilot:assert read(BASE/'pilot_audit.json')['passed']
    wanted=rr[:4] if pilot else rr
    if all((out/f'commits/{r["sample_id"]}.json').exists() for r in wanted):finish(fresh,pilot);return
    guard(16*1024**2);code=snapshot(Path(__file__));protocol={'timestamp':stamp(),'code':code,'material_sha':sha(BASE/('fresh_material.json' if fresh else 'material.json')),
      'checkpoints':'H12=block11 output,H23=block22 output,H24=block23 output,H36=block35 raw output; postnorm separately.',
      'execution':'Native BF16 CUDA eager, batch1, natural length, no padding/truncation, use_cache=False, raw corpus text without chat template.',
      'full_source_layers':[12,23],'six_positions':'Each quantile anchor and following two positions; every2560 native coordinate.',
      'controls':'First4 natural shapes: hooked/unhooked final states bitwise; replacing suffix after first query cannot change any saved prefix H12/H23 or first H24/H36/postnorm.',
      'fresh_frozen_sha':sha(BASE/'frozen.json') if fresh else None}
    ppath=out/f'protocols/{code["sha256"][:16]}.json'
    if not ppath.exists():save(ppath,protocol)
    from phase2662_symmetric_mapping_contract import load_native
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    save(out/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'model_config':model.config.to_dict(),
      'native_model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),'model':'qwen3-4b','dtype':'bfloat16','quantized':False,
      'device_map':getattr(model,'hf_device_map',{'actual':str(device)}),'tokenizer_sha':sha(ROOT/'models/hf/qwen3-4b/tokenizer.json')})
    start=time.monotonic()
    try:
      with torch.inference_mode():
       for index,r in enumerate(wanted):
        cp=out/f'commits/{r["sample_id"]}.json'
        if cp.exists():continue
        t=time.monotonic();assert tok(r['text'],add_special_tokens=False)['input_ids']==r['prompt_ids']
        x=torch.tensor([r['prompt_ids']],device=device);observer.positions=r['positions'];observer.data={};observer.enabled=True
        y=model.model(input_ids=x,use_cache=False).last_hidden_state;observer.enabled=False;packet=observer.data
        logits=model.lm_head(y[:,r['anchors']])[0];lp=logits.float().log_softmax(-1);ids=[r['prompt_ids'][p+1] for p in r['anchors']]
        checks={'all_saved_coordinates_finite':all(bool(np.isfinite(unbits(a)).all()) for a in packet.values())};assert checks['all_saved_coordinates_finite']
        if index<4:
            plain=model.model(input_ids=x,use_cache=False).last_hidden_state;assert torch.equal(y,plain)
            changed=x.clone();p=r['anchors'][0];changed[0,p+1:]=tok.eos_token_id
            observer.data={};observer.enabled=True;altered=model.model(input_ids=changed,use_cache=False).last_hidden_state;observer.enabled=False
            assert torch.equal(y[0,:p+1],altered[0,:p+1])
            for k in ('h12','h23'):assert np.array_equal(packet[k][:p+1],observer.data[k][:p+1])
            for k in ('h24','h36','postnorm'):assert np.array_equal(packet[k][0],observer.data[k][0])
            checks.update(same_shape_hook_noop=True,rewritten_future_suffix_bitwise_invariant=True)
            del plain,changed,altered
        packet['positions']=np.array(r['positions']);path=out/f'fields/{r["sample_id"]}.npz';npz(path,**packet)
        bp=out/f'behavior/{r["sample_id"]}.json';save(bp,{'timestamp':stamp(),'native_argmax':logits.argmax(-1).cpu().tolist(),'observed_next_token_ids':ids,
          'observed_next_token_NLL':[-float(lp[j,v]) for j,v in enumerate(ids)],'teacher_forced':True,'full_logits_persisted':False,
          'reference_recompute':'Saved native postnorm and unchanged real BF16 head, reporting head execution shape; original observed two-anchor behavior separately retained.'})
        save(cp,{'timestamp':stamp(),'source_sha':code['sha256'],'protocol':str(ppath.relative_to(BASE)),'protocol_sha':sha(ppath),
          'files':{str(q.relative_to(BASE)):sha(q) for q in (path,bp)},'checks':checks,'tokens':len(r['prompt_ids']),'seconds':time.monotonic()-t})
        print('RELATION_CAPTURE','fresh' if fresh else 'main',index+1,len(wanted),r['sample_id'],path.stat().st_size,flush=True)
        del x,y,logits,lp,packet;observer.data={};gc.collect();guard(16*1024**2)
        assert time.monotonic()-start<5400
    finally:observer.close();del model;gc.collect();torch.cuda.empty_cache()
    finish(fresh,pilot);print('RELATION_CAPTURE_COMPLETE',fresh,pilot,usage(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();main(a.pilot,a.fresh)
