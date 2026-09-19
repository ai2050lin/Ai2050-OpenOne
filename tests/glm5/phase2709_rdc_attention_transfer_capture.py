"""Fresh material, frozen N predictors, bounded native cached-decoding capture."""
import argparse,gc,shutil,sys
from rdc_conditional_common import *
from rdc_attention_transfer_material import build
from phase2697_rdc_generation_capture import Capture
OUT=CAMPAIGN/'o_generalization';MAX_STEPS=12;ANALYSIS_STEPS=(1,4,8)


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'prefixes.json',rows)
    models=CAMPAIGN/'n_cached_attention';assert (models/'result.json').exists()
    frozen={str(p.relative_to(models)):sha(p) for p in sorted((models/'models').glob('*.npz'))}
    for name in ('features.npz','input_grams.npz','selected_rows.json','protocol.json','result.json'):frozen[name]=sha(models/name)
    immutable(OUT/'frozen_predictors.json',{'source_run':'n_cached_attention','files':frozen,'fitting_in_this_phase':False})
    immutable(OUT/'protocol.json',{'phase':2709,'source_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_attention_transfer_material.py'),
      'material_sha':sha(OUT/'prefixes.json'),'frozen_predictors_sha':sha(OUT/'frozen_predictors.json'),
      'reason':'Automatic same-goal independent-material challenge of the Phase2708 cache-conditional attention predictor, selected after N main results. No tuning on O.',
      'material':'8operations *16freshentitypairs *2languages *2crossedwording/source-order forms =512prefixes. Complete entity strings absent from I/K/M fit materials; tokenizer units and common nouns may overlap, and absence from all historical/pretraining text is not claimed. All O heldout. Two forms change both wording and source/instruction order, not isolated form causality.',
      'execution':'Native Qwen4 BF16 CUDAeager,batch1,naturalshape,KVcache,greedy,max12 newtokens. Do not force Result labels or wait for a content boundary. No model overlap or quantization.',
      'capture':'Every actual step H0..36/currentquery full2560. At declared existing steps1/4/8, full L23 gate/up/a/down/mlp_x/attention_x, Q/currentquery, every historical K/V/P, head/O output, full151936 logits. Not alltokenprefill field and not whole generation beyond12tokens.',
      'target_alignment':'Only actual existing cached decode steps1/4/8. EOS-shorter prefixes stay in coverage reporting; no manufactured future states. Past KV excludes the current last source. Predict native L23attention, not full-answer success.',
      'scores':'Actual tokens, EOS, cap, entropy and selected-state coverage only. Deliberately capped12token outputs are not assigned full-task correctness or interpreted as failures to reason.',
      'source_labels':'Record-span tokens / otherprompt / generated_other. No inferred semantic Trace/Result labels in free-form output.',
      'pilot':'First32 cover all8operations/bothlanguages/bothforms atentity0. Four entire generated12token-orEOS sequences compared to model.generate. Only collector/length/storage checked, no predictor scores used to choose the design.',
      'resources':{'maximum_prefixes':512,'maximum_steps':12,'analysis_steps':list(ANALYSIS_STEPS),'maximum_capture_seconds':2400,'maximum_phase_bytes':int(3.2*1024**3),'forecast_reserve_bytes':int(.55*1024**3),'campaign_ceiling':30*1024**3,'free_floor':8*1024**3},
      'limits':['Independent of N training examples but not a population-random sample.','16entitygroups, two forms share facts; token states correlated.','New tasks/wording/output styles and generation steps shift together; no single factor explains transfer failure.','Past native high-layer KV remains required, so this is not an end-to-end decoder replacement.']})
    save(OUT/'material_checks.json',{'passed':True,'prefixes':512,'paired_form_groups':256,'new_entity_strings':64,'tokenizers_may_reuse_tokens':True,
      'max_prompt_tokens':max(len(r['prompt_ids']) for r in rows),'all_material_heldout':True,'timestamp':stamp()})
    return rows


def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare();protocol=read(OUT/'protocol.json');resources=protocol['resources'];selected=rows[:limit]
    if all((OUT/f'prefix_commits/{r["sample_id"]}.json').exists() for r in selected):return
    if limit==512:assert read(OUT/'pilot_audit.json')['passed'],'Pilot did not qualify declared full scope'
    assert shutil.disk_usage(OUT).free>resources['free_floor']
    model,tok=load_native('qwen4');cap=Capture(model);started=time.monotonic();noops=[]
    save(OUT/'runtime.json',{'timestamp':stamp(),'dtype':str(model.dtype),'quantized':False,'native_code_sha':sha(Path(cap.module.__file__)),'torch':torch.__version__,'cache':True,'max_steps':12})
    announce('o_generalization',state='running',completed=len(list((OUT/'prefix_commits').glob('*.json'))),total=512)
    try:
     with torch.inference_mode():
      for ix,r in enumerate(selected):
        if (OUT/f'prefix_commits/{r["sample_id"]}.json').exists():continue
        assert time.monotonic()-started<resources['maximum_capture_seconds']
        begin=time.monotonic();prefix=r['prompt_ids'].copy();generated=[];cache=None;sids=[];selected_ids=[]
        for step in range(MAX_STEPS):
            cap.enabled=True;cap.data={}
            state=model.model(input_ids=torch.tensor([prefix if step==0 else prefix[-1:]],device='cuda'),past_key_values=cache,use_cache=True)
            cache=state.past_key_values;logits=model.lm_head(state.last_hidden_state[:,-1])[0];cap.enabled=False
            nextid=int(logits.argmax());h=np.stack([cap.data.pop(f'H{l}')[0] for l in range(37)]);pack={'h_c':h};analysis=step in ANALYSIS_STEPS
            if analysis:
                for part in ('gate','up','a','down','mlp_x','attention_x','head_output','attention_out'):
                    pack['L23_'+part]=cap.data['L23_'+part][0]
                for part in ('q','k','v','p'):pack['L23_'+part]=cap.data['L23_'+part]
                pack['logits']=bits(logits);pack['postnorm']=cap.data['postnorm'][0]
            assert all(np.isfinite(unbits(v)).all() for v in pack.values())
            sid=r['sample_id']+f'-s{step}';sids.append(sid)
            if analysis:selected_ids.append(sid)
            row=dict(sample_id=sid,prefix_id=r['sample_id'],base_id=r['base_id'],family=r['family'],unit=r['unit'],language=r['language'],form=r['form'],
              word_split='test',u=r['u'],v=r['v'],target=r['target'],generation_step=step,query_position=len(prefix)-1,initial_prompt_length=len(r['prompt_ids']),
              prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),next_token_id=nextid,next_token=tok.decode([nextid]),analysis_selected=analysis,
              source_labels=(['record' if j in r['record_token_positions'] else 'prompt_other' for j in range(len(r['prompt_ids']))]+['generated_other']*len(generated)) if analysis else None)
            fp=OUT/f'fields/{sid}.npz';rp=OUT/f'steps/{sid}.json';npz(fp,**pack);save(rp,row)
            save(OUT/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,rp)}})
            generated.append(nextid);prefix.append(nextid)
            if nextid==tok.eos_token_id:break
        if ix<4:
            plain=model.generate(input_ids=torch.tensor([r['prompt_ids']],device='cuda'),do_sample=False,max_new_tokens=MAX_STEPS,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
            assert plain==generated;noops.append({'sample_id':r['sample_id'],'full_capped_greedy_equal':True})
        eos=generated[-1]==tok.eos_token_id
        bp=OUT/f'behavior/{r["sample_id"]}.json';save(bp,{'sample_id':r['sample_id'],'generated':tok.decode(generated,skip_special_tokens=True),'generated_ids':generated,
          'steps':len(generated),'eos':eos,'cap_without_eos':len(generated)==MAX_STEPS and not eos,'analysis_states':selected_ids,
          'scope':'Only partial-output observation, not a whole-answer correctness score'})
        save(OUT/f'prefix_commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'steps':sids,'analysis_states':selected_ids,'behavior_sha':sha(bp),'elapsed_seconds':time.monotonic()-begin})
        print('O_CAPTURE',ix+1,limit,r['family'],r['language'],r['form'],len(generated),round(time.monotonic()-begin,2),flush=True)
        announce('o_generalization',state='running',completed=ix+1,total=512)
        cap.data={};del state,cache,logits,h,pack;gc.collect()
        if ix%32==31:
            physical=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
            assert physical+resources['forecast_reserve_bytes']<resources['maximum_phase_bytes'] and total+resources['forecast_reserve_bytes']<resources['campaign_ceiling']
            assert shutil.disk_usage(OUT).free>resources['free_floor']
    finally:cap.close()
    material=[]
    for r in rows:
        cp=OUT/f'prefix_commits/{r["sample_id"]}.json'
        if cp.exists():material.extend(read(OUT/f'steps/{sid}.json') for sid in read(cp)['steps'])
    save(OUT/'material.json',material);save(OUT/f'noops_{limit}.json',noops)
    if limit==32:
        physical=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file());projected=physical*16+resources['forecast_reserve_bytes']
        save(OUT/'pilot_audit.json',{'timestamp':stamp(),'prefixes':32,'states':len(material),'analysis_states':sum(r['analysis_selected'] for r in material),'physical_bytes':physical,
          'projected_phase_bytes_including_forecasts':projected,'projected_campaign_bytes':total-physical+projected,'noops':len(noops),
          'passed':len(noops)==4 and projected<resources['maximum_phase_bytes'] and total-physical+projected<resources['campaign_ceiling']})
    announce('o_generalization',state='pilot_complete' if limit==32 else 'captured',completed=len(list((OUT/'prefix_commits').glob('*.json'))),total=512,states=len(material))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--limit',type=int,choices=(32,512),default=32);a=p.parse_args();prepare() if a.prepare else main(a.limit)
