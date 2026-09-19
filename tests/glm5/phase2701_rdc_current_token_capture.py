"""Current-decision cached trajectories; all past sources, all native units, no forced output."""
import argparse,gc,sys,shutil
from rdc_continuity_common import *
from phase2697_rdc_generation_capture import Capture,LAYERS
OUT=CAMPAIGN/'g_generation';RUN='g_generation'

def prepare():
    rows=[r for r in read(CAMPAIGN/'e_confirmation/material.json') if r['unit'] in (0,1,8,12,13)]
    assert len(rows)==320
    immutable(OUT/'prefixes.json',rows)
    immutable(OUT/'protocol.json',{'phase':2701,'source_sha':sha(Path(__file__)),'cases':320,'max_steps':8,
      'selection':'All8 families, units0/1 train128,8validation64,12/13 test128; allfact/query/language variants grouped. No filtering on behavior.',
      'capture':'BF16 nonquantized Qwen4 eager, natural batch1 cache; all37 current-query H full2560; native11/23/35 full9728 and all32head/source P and allsourceK/V.',
      'current_target':'Full-vocabulary probability mass of fixed token shortlist Yes/No/是/否/period/Chinese period/exclamation/newline/EOS plus remaining probability. Actual argmax category separately.',
      'noops':'first16 same-shape native generate max8, tokens must match',
      'splits':'Prefix-level grouping across steps; no future token or target used as predictor input.',
      'limits':['Short answer task; observed period/EOS performance does not imply general stopping mechanism.','Subset of E: new generation evidence, not an additional independent relation confirmation.']})
    return rows

def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare()[:limit];allrows=read(OUT/'material.json') if (OUT/'material.json').exists() else []
    if all((OUT/f'prefix_commits/{r["sample_id"]}.json').exists() for r in rows):return
    model,tok=load_native('qwen4');cap=Capture(model);assert not getattr(model,'is_quantized',False)
    ids_short=sorted(set([tok.eos_token_id]+[x for s in ('Yes','No','是','否','.','。','!','！','\n') for x in tok.encode(s,add_special_tokens=False)]))
    immutable(OUT/'shortlist.json',{'token_ids':ids_short,'tokens':tok.convert_ids_to_tokens(ids_short),'other_index':len(ids_short),'eos_id':tok.eos_token_id})
    save(OUT/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'dtype':str(model.dtype),'cache':True,'model_code_sha':sha(Path(cap.module.__file__))})
    noops=[]
    try:
      with torch.inference_mode():
       for ri,r in enumerate(rows):
        if (OUT/f'prefix_commits/{r["sample_id"]}.json').exists():continue
        start=time.monotonic();prefix=r['prompt_ids'].copy();generated=[];cache=None;stepids=[]
        for step in range(8):
          cap.data={};cap.enabled=True
          state=model.model(input_ids=torch.tensor([prefix if step==0 else prefix[-1:]],device='cuda'),use_cache=True,past_key_values=cache);cache=state.past_key_values
          logits=model.lm_head(state.last_hidden_state[:,-1]).float()[0];cap.enabled=False
          nextid=int(logits.argmax());h=np.stack([cap.data.pop(f'H{l}') for l in range(37)]);assert h.shape==(37,1,2560)
          assert np.isfinite(unbits(h)).all()
          assert all(np.isfinite(unbits(a)).all() for a in cap.data.values())
          prob=torch.softmax(logits,dim=-1);short=prob[ids_short].cpu().numpy();grouped=np.r_[short,max(0.,1-float(short.sum()))]
          sid=f'g-{r["sample_id"]}-s{step}';stepids.append(sid)
          row=dict(r,sample_id=sid,prefix_id=r['sample_id'],generation_step=step,query_position=len(prefix)-1,prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),prompt=tok.decode(prefix),next_token_id=nextid,next_token=tok.decode([nextid]),visible_last_token=prefix[-1])
          beh={'generated':tok.decode(generated+[nextid],skip_special_tokens=True),'next_id':nextid,'eos':nextid==tok.eos_token_id,'expected_target':r['target'],'correct':tok.decode(generated+[nextid],skip_special_tokens=True).strip().rstrip('.。!！').casefold()==r['target'].casefold(),'group_probability':grouped.tolist(),'argmax_group':ids_short.index(nextid) if nextid in ids_short else len(ids_short)}
          fp=OUT/f'fields/{sid}.npz';npz(fp,h=h,logits=logits.cpu().numpy(),**cap.data)
          bp=OUT/f'behavior/{sid}.json';save(bp,beh)
          save(OUT/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,bp)}})
          allrows.append(row);save(OUT/'material.json',allrows)
          announce(RUN,state='running',completed=len(allrows),total=2560,current_sample=sid,completed_prefixes=ri)
          events(RUN,'generation_step_committed',sample_id=sid,step=step)
          generated.append(nextid);prefix.append(nextid)
          if nextid==tok.eos_token_id:break
        if ri<16:
          plain=model.generate(input_ids=torch.tensor([r['prompt_ids']],device='cuda'),do_sample=False,max_new_tokens=8,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
          assert plain==generated;noops.append({'prefix_id':r['sample_id'],'same_shape_generated_ids_equal':True})
        save(OUT/f'prefix_commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'steps':stepids,'generated_ids':generated,'eos':tok.eos_token_id in generated,'elapsed_seconds':time.monotonic()-start})
        print('CURRENT',ri+1,len(rows),len(generated),flush=True)
        cap.data={};del cache,state,logits;gc.collect();assert shutil.disk_usage(OUT).free>8*1024**3
    finally:cap.close()
    save(OUT/f'noop_{limit}.json',noops)
    announce(RUN,state='captured' if limit==320 else 'pilot_complete',completed=len(allrows),total=len(allrows),prefixes=len(list((OUT/'prefix_commits').glob('*.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,320),default=16);main(p.parse_args().limit)
