"""Full source-position H12 organization, finite autonomous follow-up; no coordinate compression."""
import argparse
import gc
from collections import Counter
from rdc_prefix_estimators import *
from phase2711_rdc_prefix_material import parse,aligned_words
OUT=CAMPAIGN/'full_source_history'
RULES=('current','mean_history','absolute_history','relative_history')


def prepare():
    if (OUT/'protocol.json').exists():return
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    prior=read(CAMPAIGN/'material_stratified.json')+read(CAMPAIGN/'confirmation_material.json')
    used_groups={r['source_group'] for r in prior};used_text={re.sub(r'\W','',r['text']).casefold() for r in prior};fresh=[]
    for lang in ('en','zh'):
        eligible=[];seen=set()
        for r in parse(CAMPAIGN/f'sources/{lang}_test.conllu',lang):
            norm=re.sub(r'\W','',r['text']).casefold()
            if r['source_group'] in used_groups or r['source_group'] in seen or norm in used_text:continue
            enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
            if not 24<=len(enc['input_ids'])<=96:continue
            seen.add(r['source_group']);r.update(prompt_ids=enc['input_ids'],token_offsets=enc['offset_mapping'],tokens=tok.convert_ids_to_tokens(enc['input_ids']))
            eligible.append(r)
        buckets={g:sorted([r for r in eligible if r['genre']==g],key=lambda r:hashlib.sha256(('2714:'+r['source_group']).encode()).hexdigest()) for g in sorted({r['genre'] for r in eligible})}
        chosen=[]
        for i in range(max(map(len,buckets.values()))):
            for g in buckets:
                if i<len(buckets[g]):chosen.append(buckets[g][i])
            if len(chosen)>=32:break
        assert len(chosen)>=32
        for i,r in enumerate(chosen[:32]):
            n=len(r['prompt_ids']);r.update(sample_id=f'fresh-{lang}-{i:04d}',split='confirmation',anchors=[max(3,n//3),min(n-4,2*n//3)])
            r['positions']=r['anchors'];r['retrospective_ud']=aligned_words(r['text'],r['retrospective_ud'])
            r['actual_anchor_graphs']=[prefix_graph(tok.decode(r['prompt_ids'][:p+1],skip_special_tokens=False,clean_up_tokenization_spaces=False),p,lang) for p in r['positions']]
            fresh.append(r)
    fresh.sort(key=lambda r:(r['sample_id'][-4:],r['language']))
    immutable(OUT/'fresh_material.json',fresh)
    immutable(OUT/'protocol.json',{'phase':2714,'timestamp':stamp(),'source_sha':sha(Path(__file__)),
      'same_goal':'Identify reusable full native-coordinate source-position organization missing from a prefix mean, using the same shared natural corpus.',
      'completed_prerequisites':['2711 natural full-coordinate atlas','2712 shared rules/native probability','2713 frozen confirmation, three-model matched fit, client and integrity'],
      'main_material_sha':sha(CAMPAIGN/'material_stratified.json'),'fresh_material_sha':sha(OUT/'fresh_material.json'),
      'main_units':512,'fresh_units':64,'fresh_sources_excluded':'Every previous main/confirmation source group and normalized text excluded; new IDs and content-hash rank, no model outcomes.',
      'native_input':'All H12 coordinates at every source position up to query token, plus full current H12; no H36 target/norm/future tokens in any kernel.',
      'rules':list(RULES),'kernels':'current:1+k_current; other:1+(k_current+k_history)/2. Mean history is all-source mean. Absolute history aligns equal source indices; relative history aligns equal distance behind each own query. Both ordered kernels sum every valid aligned full2560 inner product and divide sqrt(prefix lengths)*training average source vector energy.',
      'selection':'Same640/192/192 main train/validation/test anchors; H36-only full2560 target; one ridge per rule chosen by validation, select one rule before fresh capture. Never train on fresh.',
      'numeric_validation':'For16 existing full panels reuse original H12 directly. For remaining496 recapture ONLY all-token H12 and compare ALL six saved anchor H12 plus two H36 and postnorm bitwise against original before commit. First4 recaptures also compare same-shape hook/nohook. No coordinate averaging in retained H12.',
      'resources':{'parent_ceiling':CEILING,'remaining_at_design':CEILING-usage(),'physical_free_at_design':shutil.disk_usage(ROOT).free,'reserve_final_audit_bytes':12*1024**2,'maximum_gpu_capture_seconds':1800,'maximum_fit_seconds':1800},
      'planned_scope':'4-rule source-organization comparison, full-coordinate/condition/source errors, fresh frozen64-source test and complete vocabulary KL, client whole-source query and final retention audit.',
      'limits':['This feature alignment tests ordered similarity, not an identified attention kernel or semantic role binding.','Natural single sentences remain24–96tokens, not arbitrary-depth generation.','Failure rejects selected source kernels, not all possible use of source history.']})
    save(CAMPAIGN/'continuation_decision.json',{'timestamp':stamp(),'completed_phases':[2711,2712,2713],'same_goal':True,'next_phase':2714,
      'state':'pilot_before_bounded_extension','reason':'History mean and single-state layer update lose source organization; test all-source position kernels without narrowing to one sentence template.',
      'resource_policy':'Do not increase3GiB ceiling. Continue only after pilot estimate includes four fitted rules, fresh64 confirmation, complete probability checks and final audit reserve.'})


def main_rows():return read(CAMPAIGN/'material_stratified.json')


def recapture(pilot=False,fresh=False):
    import torch
    prepare();run='fresh' if fresh else 'main';out=OUT/run
    if fresh:assert (OUT/'frozen.json').exists(),'Freeze all rules before fresh capture'
    rows=read(OUT/'fresh_material.json') if fresh else main_rows()
    if pilot:
        rows=[r for r in rows if not r.get('full_panel')][:4]
    if not pilot and not fresh:assert read(OUT/'pilot_audit.json')['passed']
    wanted=[r for r in rows if not (out/f'commits/{r["sample_id"]}.json').exists() and (fresh or not r['full_panel'])]
    if not wanted:print('FULL_SOURCE_CAPTURE_ALREADY_COMPLETE',run,len(rows),flush=True);return
    guard(12*1024**2)
    from phase2662_symmetric_mapping_contract import load_native
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device;captured={}
    def hook(m,a,o):captured['h12']=bits(o[0])
    handle=model.model.layers[11].register_forward_hook(hook);start=time.monotonic();records=[]
    try:
      with torch.inference_mode():
       for index,r in enumerate(wanted):
        captured.clear();ids=torch.tensor([r['prompt_ids']],device=device);positions=r['anchors']
        final=model.model(input_ids=ids,use_cache=False,output_hidden_states=True)
        # output_hidden_states[-1] is final norm; raw H36 is taken from a temporary final-block hook below only if needed.
        h12=captured['h12'];post=bits(final.last_hidden_state[0,positions]);h36raw=None
        # Qwen output_hidden_states includes H0..H35 then normalized output; reconstruct raw final block via a dedicated hook in a second audited forward.
        raw={}
        def last_hook(m,a,o):raw['h36']=bits(o[0,positions])
        last_handle=model.model.layers[-1].register_forward_hook(last_hook)
        second=model.model(input_ids=ids,use_cache=False)
        last_handle.remove();h36raw=raw['h36'];assert np.array_equal(bits(second.last_hidden_state[0,positions]),post)
        check={'sample_id':r['sample_id'],'same_shape_repeat_postnorm_bitwise':True}
        if not fresh:
            with np.load(CAMPAIGN/f'qwen4/fields/{r["sample_id"]}.npz') as old:
                assert np.array_equal(h12[r['positions']],old['h'][12]),(r['sample_id'],'H12 recapture changed')
                assert np.array_equal(h36raw,old['h'][36,[0,3]]),(r['sample_id'],'H36 recapture changed')
                assert np.array_equal(post,old['postnorm'][[0,3]]),(r['sample_id'],'postnorm changed')
            packet={'h12':h12};check['all_six_H12_two_H36_postnorm_match_original']=True
        else:
            h0=bits(model.get_input_embeddings()(ids)[0,positions]);packet={'h12':h12,'h0':h0,'h36':h36raw,'postnorm':post,'positions':np.array(positions)}
            # Native BF16 behavior is measured now without archiving redundant full vocab logits for every fresh sentence.
            logits=model.lm_head(second.last_hidden_state[:,positions])[0];lp=logits.float().log_softmax(-1);nextids=[r['prompt_ids'][p+1] for p in positions]
            save(out/f'behavior/{r["sample_id"]}.json',{'native_BF16_argmax':logits.argmax(-1).cpu().tolist(),'observed_next_ids':nextids,
              'observed_next_NLL':[-float(lp[j,t]) for j,t in enumerate(nextids)],'teacher_forced':True,'full_logits_retained':False,
              'recompute':'Original model and exact IDs/config retained; probability analysis can also use saved native postnorm and real BF16 head with explicitly stated execution shape.'})
        if pilot:
            handle.remove();unhooked=model.model(input_ids=ids,use_cache=False).last_hidden_state
            assert torch.equal(unhooked,second.last_hidden_state);handle=model.model.layers[11].register_forward_hook(hook)
            check['same_shape_hook_noop']=True
        p=out/f'fields/{r["sample_id"]}.npz';npz(p,**packet)
        save(out/f'commits/{r["sample_id"]}.json',{'files':{str(p.relative_to(OUT)):sha(p)},'source_sha':sha(Path(__file__)),
          'positions':positions,'tokens':len(r['prompt_ids']),'checks':check,'fresh_frozen_hash':sha(OUT/'frozen.json') if fresh else None})
        records.append(check);print('FULL_SOURCE_CAPTURE',run,index+1,len(wanted),r['sample_id'],p.stat().st_size,flush=True)
        assert time.monotonic()-start<1800;guard(12*1024**2)
        del final,second,ids,post,h12,packet;gc.collect()
    finally:handle.remove();del model;gc.collect();torch.cuda.empty_cache()
    if pilot:
        sizes=sum((out/f'fields/{r["sample_id"]}.npz').stat().st_size for r in rows);tokens=sum(len(r['prompt_ids']) for r in rows)
        pending_tokens=sum(len(r['prompt_ids']) for r in main_rows() if not r['full_panel'] and r['sample_id'] not in {x['sample_id'] for x in rows})
        freshrows=read(OUT/'fresh_material.json');fresh_tokens=sum(len(r['prompt_ids']) for r in freshrows)
        capture_estimate=(pending_tokens+fresh_tokens)*sizes/tokens+64*2*3*2560*2
        analysis_estimate=4*640*2560*4+4*1024**2*4+4*(192+128)*2560*4+12*1024**2
        remaining=CEILING-usage();projected=capture_estimate+analysis_estimate
        passed=bool(all(r.get('same_shape_hook_noop') and r.get('all_six_H12_two_H36_postnorm_match_original') for r in records) and projected<remaining-12*1024**2)
        save(OUT/'pilot_audit.json',{'timestamp':stamp(),'passed':passed,'pilot_rows':[r['sample_id'] for r in rows],'pilot_checks':records,
          'bytes_per_source_token':sizes/tokens,'remaining_capture_estimate':capture_estimate,'analysis_estimate':analysis_estimate,
          'total_estimated_remaining_bytes':projected,'remaining_parent_bytes':remaining,'final_reserve_bytes':12*1024**2,
          'mean_seconds_per_pilot':(time.monotonic()-start)/len(rows)})
        print('FULL_SOURCE_PILOT',passed,projected,remaining,flush=True)
    status('full_source_history_'+run,state='pilot_complete' if pilot else 'captured',completed=len(list((out/'commits').glob('*.json'))),reused_panels=0 if fresh else 16)


def source_field(row,fresh=False):
    if not fresh and row.get('full_panel'):
        with np.load(CAMPAIGN/f'qwen4/full_panels/{row["sample_id"]}.npz') as z:return unbits(z['h'][12])
    with np.load(OUT/('fresh' if fresh else 'main')/f'fields/{row["sample_id"]}.npz') as z:return unbits(z['h12'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args()
    prepare() if a.prepare else recapture(a.pilot,a.fresh)
