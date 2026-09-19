"""Long multi-operation natural trajectories with full-coordinate current states."""
import argparse,gc,shutil,sys
from rdc_conditional_common import *
from rdc_long_material import build,score,FAMILIES
from phase2693_rdc_language_capture import Capture
RUN='k_long';OUT=CAMPAIGN/RUN
ANALYSIS_STEPS=tuple(sorted(set((0,1,2)+tuple(range(0,128,4)))))


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'prefixes.json',rows)
    immutable(OUT/'protocol.json',{'phase':2705,'capture_code_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_long_material.py'),
      'prefixes':128,'families':list(FAMILIES),'max_generated_tokens':128,'prefix_material_sha':sha(OUT/'prefixes.json'),
      'split':'Eightentitygroups;0..3train64prefixes,4..5val32,6..7test32. Bothlanguages andalloperations ofentitygroup stay together. These names occur in earlier factorial material; heldout is within this new long-generator analysis, not wholecampaign novelty.',
      'model':'Qwen3-4b nonquantizedBF16 CUDA eager,batch1,no sampling,KVcache,naturalpromptlength; no truncation of input',
      'capture':'Allsteps H0..36 currentquery full2560 BF16; L23 all9728gate/up/a and all2560mlp_x/down/attention_x. At predeclared analysissteps additionally L11/35 same fullnative quantities and actual fullvocabBF16logits. Fullprefill alltokenH for16prefixes entity0, selected before outputs.',
      'analysis_steps':list(ANALYSIS_STEPS),'analysis_sampling':'Predeclared generation-step subsample to bound exact Gram memory. Every generated step still has complete currentqueryH andL23 native, not HiddenState dimension reduction.',
      'noops':'First16 prefixes native model.generate greedy max128 must match entire captured token sequence.',
      'behavior':'Independent content/order/format/EOS constraints; translation andstyle lexical checklists explicitly not wholemeaning judgments.',
      'limits':['Only16prefixes peroperation.','Max128 is a censoring limit, not learnedEOS for truncated cases.','Prefill fulltokenpersistence is16/128.','Saved query state does not include the full projected KV history.','No assumption that logical hops equal generated steps.']})
    return rows


def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare()[:limit]
    if all((OUT/f'prefix_commits/{r["sample_id"]}.json').exists() for r in rows):return
    assert shutil.disk_usage(OUT).free>8*1024**3+6*1024**3*(limit/128)
    announce(RUN,state='loading',completed=0,total=128,requested=limit)
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cap=Capture(model);noops=[]
    save(OUT/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'dtype':str(model.dtype),'cache':True,
      'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),'vocab_size':model.config.vocab_size})
    try:
      with torch.inference_mode():
       for ri,r in enumerate(rows):
        if (OUT/f'prefix_commits/{r["sample_id"]}.json').exists():continue
        start=time.monotonic();prefix=r['prompt_ids'].copy();generated=[];cache=None;stepids=[];step_rows=[]
        for step in range(128):
            cap.arrays={};cap.positions=[len(prefix)-1 if step==0 else 0];cap.enabled=True
            state=model.model(input_ids=torch.tensor([prefix if step==0 else prefix[-1:]],device='cuda'),use_cache=True,past_key_values=cache)
            cache=state.past_key_values;rawlogits=model.lm_head(state.last_hidden_state[:,-1])[0];cap.enabled=False
            nextid=int(rawlogits.argmax());fullh=np.stack([cap.arrays.pop(f'H{l}') for l in range(37)]);h=fullh[:,-1]
            assert h.shape==(37,2560) and np.isfinite(unbits(h)).all()
            selected=step in ANALYSIS_STEPS;pack={'h_c':h,'postnorm_c':cap.arrays['postnorm'][-1]}
            for l in ((11,23,35) if selected else (23,)):
                for key in ('gate','up','a','down','mlp_x','attention_x'):
                    pack[f'L{l}_{key}']=cap.arrays[f'L{l}_{key}'][0]
                    assert np.isfinite(unbits(pack[f'L{l}_{key}'])).all()
            if step==0 and r['full_prefill_panel']:pack['h_prefill']=fullh
            if selected:pack['logits']=bits(rawlogits)
            sid=f'{r["sample_id"]}-s{step}';stepids.append(sid)
            row=dict(sample_id=sid,prefix_id=r['sample_id'],generation_step=step,query_position=len(prefix)-1,
              family=r['family'],unit=r['unit'],language=r['language'],word_split=r['word_split'],analysis_selected=selected,
              visible_last_token=prefix[-1],next_token_id=nextid,next_token=tok.decode([nextid]),
              prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),source_mode='live_model')
            fp=OUT/f'fields/{sid}.npz';npz(fp,**pack)
            rp=OUT/f'steps/{sid}.json';save(rp,row)
            save(OUT/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,rp)}})
            step_rows.append(row);generated.append(nextid);prefix.append(nextid)
            if nextid==tok.eos_token_id:break
        text=tok.decode(generated,skip_special_tokens=True);eos=tok.eos_token_id in generated
        if ri<16:
            plain=model.generate(input_ids=torch.tensor([r['prompt_ids']],device='cuda'),do_sample=False,max_new_tokens=128,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
            assert plain==generated;noops.append({'prefix_id':r['sample_id'],'greedy_token_ids_bitwise_equal':True})
        bp=OUT/f'behavior/{r["sample_id"]}.json'
        save(bp,{'sample_id':r['sample_id'],'generated':text,'generated_ids':generated,'reference':r['reference'],
          'scores':score(r,text,eos,len(generated)==128 and not eos),'steps':len(generated)})
        elapsed=time.monotonic()-start
        save(OUT/f'prefix_commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'steps':stepids,'behavior_sha':sha(bp),'eos':eos,'elapsed_seconds':elapsed})
        announce(RUN,state='running',completed=ri+1,total=128,requested=limit,last_prefix_seconds=elapsed)
        print('LONG',ri+1,128,r['family'],r['language'],len(generated),round(elapsed,2),flush=True)
        cap.arrays={};del cache,state,rawlogits,pack,fullh;gc.collect()
        assert shutil.disk_usage(OUT).free>8*1024**3
    finally:cap.close()
    material=[]
    for r in read(OUT/'prefixes.json'):
        cp=OUT/f'prefix_commits/{r["sample_id"]}.json'
        if cp.exists():material.extend(read(OUT/f'steps/{sid}.json') for sid in read(cp)['steps'])
    save(OUT/'material.json',material);save(OUT/f'noop_{limit}.json',noops)
    announce(RUN,state='captured' if limit==128 else 'pilot_complete',completed=len(list((OUT/'prefix_commits').glob('*.json'))),total=128,states=len(material))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,128),default=16);main(p.parse_args().limit)
