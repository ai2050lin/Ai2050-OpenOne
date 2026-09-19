"""Bounded automatic next-stage capture: output order, whole states, actual source accounting."""
import argparse,gc,sys,shutil
from rdc_conditional_common import *
from rdc_order_material import build,score,result_boundary,source_labels
from phase2697_rdc_generation_capture import Capture
OUT=CAMPAIGN/'m_order';MAX_STEPS=128


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'prefixes.json',rows)
    immutable(OUT/'protocol.json',{'phase':2707,'source_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_order_material.py'),'material_sha':sha(OUT/'prefixes.json'),
      'status':'Automatic same-goal follow-up selected from K wrongResult/correctTrace counterexamples, not an independent confirmation of a prespecified liquid drift theory.',
      'cases':'3operations *8entities *2truthstates *2languages *3orders =288prefixes. Same record and same required Result/Trace/Neutral content within each3order pair-set.',
      'split':'entities0..3train144prefixes,4..5val72,6..7test72; keep alltranslations/truthstates/orders together. Entities overlap previous I/K; novelty is output-order design, not unseen project-wide names.',
      'condition':'Result-first, Trace-first, neutral-markers-first. All three fields requested in every condition; instruction order and generated histories necessarily differ. Neutral control is not exactly length/token matched; record lengths and no causal purity claim.',
      'model':'Qwen4 nonquantizedBF16 CUDAeager,batch1,naturalshape,KVcache,greedy,max128; oneCUDAmodel only after alignedmodels exit.',
      'capture':'Every step all37 H checkpoints/currentquery full2560 and allL23 gate/up/a/down/mlp_x/attention_x. At step0 and after generated Result label, additionally L11/35 native; at Result boundary allhistory actual Q/K/V/P andattentionout/headinput atL11/23/35, fullvocablogits, and source field labels. All arrays nativeBF16 bits; no hidden-coordinate reduction.',
      'boundary':'Before forward: generated output ends in line label Result: with optional spaces. Persist consecutive candidate boundary states; select first with alphanumeric next token for comparison. Missing/malformed label reported, not silently repaired or replaced by expectedanswer.',
      'scores':'Frozen deterministic Result/Trace/Neutral, order, format, EOS separate; expected values are never injected into free generation.',
      'pilot':'First36 full crossed entity0 prefixes, noops4 entire generated sequences; expand only after scores/reference/adversarial checks and projected physicalbytes<=6GiB within30GiBcampaign,8GiBfreefloor.',
      'resource':'288prefixes,max128tokens,<=3600captureseconds,6GiB M-phase ceiling,30GiBcampaign ceiling. Stop with explicit resourcebound and preservedstate if exceeded; no infinite loop.',
      'analysis':'All-native layer readers for external binary state at prefill/Result-boundary; full-coordinate means and conditional trajectories; exact source-group attention vector accounting, not attention weight as causal importance; pairedbehavior acrosssame records.'})
    for r in rows:
        s=score(r,r['reference'],True,False);assert all(s[k] for k in ('all_content_correct','field_order_correct','format_structure','exact_reference'))
        bad=r['reference'].replace('Result: '+r['expected_fields']['Result'],'Result: WRONG');s=score(r,bad,True,False)
        assert not s['result_correct'] and s['trace_correct'] and s['format_structure']
        swapped=r['reference'].splitlines();swapped[0],swapped[1]=swapped[1],swapped[0];s=score(r,'\n'.join(swapped),True,False)
        assert s['all_content_correct'] and not s['field_order_correct']
    assert result_boundary('Trace: X0 -> X1\nResult: ') and not result_boundary('Result: A')
    save(OUT/'material_checks.json',{'timestamp':stamp(),'references':288,'all_passed':True,'wrong_result_and_order_adversarial':True,'same_record_and_required_content_pairs':96})
    return rows


def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare();selected=rows[:limit];existing=list((OUT/'prefix_commits').glob('*.json'))
    if len(existing)>=limit:return
    if limit==288:
        pilot=read(OUT/'pilot_audit.json');assert pilot['passed'] and pilot['projected_bytes']<6*1024**3
    assert shutil.disk_usage(OUT).free>8*1024**3
    model,tok=load_native('qwen4');cap=Capture(model);wall=time.monotonic();noops=[];newbytes=0
    save(OUT/'runtime.json',{'timestamp':stamp(),'dtype':str(model.dtype),'quantized':False,'torch':torch.__version__,'cache':True,'native_code_sha':sha(Path(cap.module.__file__))})
    announce('m_order',state='running',completed=len(existing),total=288,requested=limit)
    try:
     with torch.inference_mode():
      for ix,r in enumerate(selected):
        if (OUT/f'prefix_commits/{r["sample_id"]}.json').exists():continue
        assert time.monotonic()-wall<3600,'Declared capture time budget reached'
        start=time.monotonic();prefix=r['prompt_ids'].copy();generated=[];cache=None;stepids=[];boundaries=[];chosen=None
        for step in range(MAX_STEPS):
            before=tok.decode(generated,skip_special_tokens=True);boundary=result_boundary(before)
            cap.data={};cap.enabled=True
            state=model.model(input_ids=torch.tensor([prefix if step==0 else prefix[-1:]],device='cuda'),past_key_values=cache,use_cache=True)
            cache=state.past_key_values;logits=model.lm_head(state.last_hidden_state[:,-1])[0];cap.enabled=False
            nextid=int(logits.argmax());token=tok.decode([nextid]);lexical=any(c.isalnum() for c in token) and nextid not in tok.all_special_ids
            h=np.stack([cap.data.pop(f'H{l}')[0] for l in range(37)]);pack={'h_c':h}
            for l in ((11,23,35) if step==0 or boundary else (23,)):
                for key in ('gate','up','a','down','mlp_x','attention_x'):pack[f'L{l}_{key}']=cap.data[f'L{l}_{key}'][0]
            if boundary:
                for l in (11,23,35):
                    for key in ('q','k','v','p','head_output','attention_out'):
                        v=cap.data[f'L{l}_{key}'];pack[f'L{l}_{key}']=v[0] if key in ('head_output','attention_out') else v
                pack['logits']=bits(logits);pack['postnorm']=cap.data['postnorm'][0]
            assert all(np.isfinite(unbits(a)).all() for a in pack.values())
            sid=r['sample_id']+f'-s{step}';stepids.append(sid)
            if boundary:
                boundaries.append(sid)
                if chosen is None and lexical:chosen=sid
            row=dict(sample_id=sid,prefix_id=r['sample_id'],base_id=r['base_id'],family=r['family'],unit=r['unit'],truth=r['truth'],language=r['language'],
              order=r['order'],word_split=r['word_split'],generation_step=step,query_position=len(prefix)-1,initial_prompt_length=len(r['prompt_ids']),
              prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),next_token_id=nextid,next_token=token,
              result_boundary=boundary,boundary_next_lexical=boundary and lexical,source_labels=source_labels(tok,r,generated) if boundary else None)
            fp=OUT/f'fields/{sid}.npz';npz(fp,**pack);rp=OUT/f'steps/{sid}.json';save(rp,row)
            save(OUT/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,rp)}})
            newbytes+=fp.stat().st_size+rp.stat().st_size;generated.append(nextid);prefix.append(nextid)
            if nextid==tok.eos_token_id:break
        text=tok.decode(generated,skip_special_tokens=True);eos=tok.eos_token_id in generated
        if ix<4:
            plain=model.generate(input_ids=torch.tensor([r['prompt_ids']],device='cuda'),do_sample=False,max_new_tokens=MAX_STEPS,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
            assert plain==generated;noops.append({'sample_id':r['sample_id'],'entire_greedy_tokens_equal':True})
        bp=OUT/f'behavior/{r["sample_id"]}.json';save(bp,{'sample_id':r['sample_id'],'generated':text,'generated_ids':generated,'reference':r['reference'],
          'scores':score(r,text,eos,len(generated)==MAX_STEPS and not eos),'steps':len(generated),'result_boundary_state':chosen,'boundary_candidates':boundaries})
        save(OUT/f'prefix_commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'steps':stepids,'behavior_sha':sha(bp),'result_boundary_state':chosen,'elapsed_seconds':time.monotonic()-start})
        print('ORDER_CAPTURE',ix+1,limit,r['family'],r['order'],len(generated),chosen,round(time.monotonic()-start,2),flush=True)
        announce('m_order',state='running',completed=ix+1,total=288,requested=limit)
        cap.data={};del state,cache,logits,h,pack;gc.collect();assert shutil.disk_usage(OUT).free>8*1024**3
        if ix%12==11:
            size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
            assert size<6*1024**3 and total<30*1024**3,'Declared physical storage ceiling reached'
    finally:cap.close()
    material=[]
    for r in rows:
        cp=OUT/f'prefix_commits/{r["sample_id"]}.json'
        if cp.exists():material.extend(read(OUT/f'steps/{sid}.json') for sid in read(cp)['steps'])
    save(OUT/'material.json',material);save(OUT/f'noop_{limit}.json',noops)
    if limit==36:
        physical=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file());projection=physical*8
        save(OUT/'pilot_audit.json',{'timestamp':stamp(),'prefixes':36,'states':len(material),'bytes':physical,'projected_bytes':projection,'projected_campaign_bytes':total-physical+projection,
          'noops':len(noops),'passed':len(noops)==4 and projection<6*1024**3 and total-physical+projection<30*1024**3,
          'missing_result_boundary':sum(read(OUT/f'behavior/{r["sample_id"]}.json')['result_boundary_state'] is None for r in rows[:36])})
    announce('m_order',state='pilot_complete' if limit==36 else 'captured',completed=len(list((OUT/'prefix_commits').glob('*.json'))),total=288,states=len(material))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--limit',type=int,choices=(36,288),default=36);a=p.parse_args();prepare() if a.prepare else main(a.limit)
