"""Same-generation immutable KV replay, on all96 previously exposed families."""
import sys,time,gc
import numpy as np
import b36_switch14 as M
import b36_switch14_engine as E
S=M.S


def prepare():
    S.configure(M.PEAK);assert not (S.OUT/'calibration-plan.json').exists()
    old=S.read(M.PRIOR/'plan.json');rows=[r for r in old['rows'] if r['arm']=='full'];assert len(rows)==96
    S.save('calibration-plan.json',dict(rows=rows,prior_plan_sha256=S.file_sha(M.PRIOR/'plan.json'),
        model_files=old['model_files'],source_hashes=M.sources(['b36_switch14.py','b36_switch14_engine.py','b36_switch14_calibrate.py','b36_transition14.py','b36_semantic14.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']),versions=M.versions(),
        cut_generated_tokens=8,max_new_tokens=64,batch=96,
        gate='All96 alive at fixed cut8; full original-vs-cached replay token sequences identical96/96; first resumed full-vocabulary logits bitwise identical; CPU KV snapshot hashes unchanged. Failure stops new-family mechanism work.',
        evidence='Previously exposed material only, calibration not independent confirmation. Snapshot comes from the original HF generate forward before processing the8th generated token, with8 generated tokens in sequence and7 in KV.',
        retention='CPU snapshot only while both branches run; preserve per-layer exact hashes, pending inputs, prefix IDs, all generated tokens and first-branch BF16 logits. Full multi-GiB KV not written to disk.',
        budget=dict(new_peak_bytes=M.PEAK,reserve_bytes=20*2**30,expected_minutes=30,hard_minutes=80)))
    print('KV calibration frozen:96 old families, cut8, exact96/96 gate',flush=True)


def run():
    S.configure(M.PEAK);p=S.read(S.OUT/'calibration-plan.json');M.admit(p);start=time.monotonic()
    import torch
    with S.GpuLease(),torch.inference_mode():
        for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('cache_identity',i+1,len(p['model_files']),start)
        engine=E.Stream(start,p['budget']['hard_minutes']);inputs=engine.inputs(p['rows']);width=inputs['input_ids'].shape[1]
        state=dict(index=0,tokens=[],snapshot=None,mode='original');logits={}
        def before(m,args,kw):
            if state['mode']=='original' and state['index']==p['cut_generated_tokens']:
                prefix=torch.cat([inputs['input_ids'],torch.stack(state['tokens'],1)],1)
                assert torch.equal(prefix[:,-1:],kw['input_ids'])
                data=E.copy_cache(kw['past_key_values']);desc=E.cache_description(data)
                prepared={k:v.detach().cpu().clone() for k,v in kw.items() if torch.is_tensor(v)}
                state['snapshot']=dict(cache=data,description=desc,prepared=prepared,prefix=prefix.cpu().clone(),alive=not bool((prefix[:,width:]==engine.tok.eos_token_id).any()))
        def after(m,args,kw,out):
            if state['mode']=='original':
                if state['index']==p['cut_generated_tokens']:logits['original']=E.bits(out.logits[:,-1])
                if state['index']<p['cut_generated_tokens']:state['tokens'].append(out.logits[:,-1].argmax(-1).detach().clone())
            elif state['index']==0:logits['replay']=E.bits(out.logits[:,-1])
            state['index']+=1
        hooks=[engine.model.register_forward_pre_hook(before,with_kwargs=True),engine.model.register_forward_hook(after,with_kwargs=True)]
        original=engine.generate('cache_original',inputs,p['max_new_tokens']);original_records=engine.records(p['rows'],original,width)
        S.save('cache-original.json',dict(results=original_records,plan_sha256=S.file_sha(S.OUT/'calibration-plan.json')))
        snap=state['snapshot']
        if snap is None or not snap['alive']:
            S.save('calibration.json',dict(passed=False,reason='Missing fixed checkpoint or at least1 sequence already ended. No success-subset selection.'));return
        del original;state['tokens']=[];gc.collect();torch.cuda.empty_cache()
        state.update(index=0,mode='replay')
        resumed=dict(input_ids=snap['prefix'].cuda(),attention_mask=snap['prepared']['attention_mask'].cuda(),position_ids=snap['prepared']['position_ids'].cuda())
        cache=E.restore_cache(snap['cache'],engine.cfg)
        replay=engine.generate('cache_replay',resumed,p['max_new_tokens']-p['cut_generated_tokens'],past_key_values=cache)
        replay_records=engine.records(p['rows'],replay,width)
        exact=[a['tokens']==b['tokens'] for a,b in zip(original_records,replay_records)]
        immutable=E.cache_description(snap['cache'])==snap['description'];bitwise=np.array_equal(logits['original'],logits['replay'])
        S.save('cache-replay.json',dict(results=replay_records,plan_sha256=S.file_sha(S.OUT/'calibration-plan.json')))
        np.savez_compressed(S.OUT/'cache-first-logits.npz',**logits)
        S.save('cache-snapshot.json',dict(layers=snap['description'],prefix_ids=snap['prefix'].tolist(),prepared_inputs={k:v.tolist() for k,v in snap['prepared'].items()},immutable_after_replay=immutable,scope='Original live generation checkpoint. Full KV discarded only after immutable comparison; no prefill reconstruction.'))
        S.save('calibration.json',dict(passed=all(exact) and immutable and bitwise,n=96,exact=sum(exact),mismatch_ids=[r['id'] for r,ok in zip(p['rows'],exact) if not ok],first_logits_bitwise=bitwise,cache_immutable=immutable,seconds=time.monotonic()-start,source_hashes=p['source_hashes'],plan_sha256=S.file_sha(S.OUT/'calibration-plan.json'),logits_sha256=S.file_sha(S.OUT/'cache-first-logits.npz'),snapshot_sha256=S.file_sha(S.OUT/'cache-snapshot.json')))
        for h in hooks:h.remove()
        del replay,cache,snap;state.clear();engine.close();M.admit(p)
        print('Cache replay gate:',all(exact) and immutable and bitwise,'exact',sum(exact),'/96',flush=True)


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
