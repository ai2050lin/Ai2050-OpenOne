"""Diagnose same-source prefix discrepancies without reinterpreting them as semantics."""
import argparse,itertools
import numpy as np
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2676_native_mlp_delivery/prefix_replay';FIELD=RESULT/'phase2671_native_mlp_field'


def prepare():
    if (OUT/'protocol/frozen.json').exists():return read(OUT/'material/groups.json')
    rows=read(RESULT/'phase2670_native_mlp_contract/material/cases.json');index={r['case_index']:r for r in rows};groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('family','language','unit','content_instance','form','target_index','mention_order')),[]).append(r)
    prior=read(OUT.parent/'analysis/source_prefix_audit.json');known={r['base_case'] for r in prior['changed_frozen_candidate_pairs']};selected=[]
    for fam,lang in itertools.product(sorted({r['family'] for r in rows}),('en','zh')):
        gg=[r for key,r in groups.items() if key[:2]==(fam,lang)];exceptions=[r for r in gg if r[0]['case_index'] in known][:2];rest=[r for r in gg if r not in exceptions];chosen=(exceptions+rest)[:4]
        assert len(chosen)==4
        for gi,rr in enumerate(chosen):
            base=rr[0];other=max(rr[1:],key=lambda r:(len(r['prompt_ids']),-r['case_index']));body=base['prompt_ids'][:base['body_end_token']+1]
            assert other['prompt_ids'][:other['body_end_token']+1]==body
            selected.append({'index':len(selected),'base':base,'other':other,'pad_length':max(len(base['prompt_ids']),len(other['prompt_ids']))+1,
                'published':fam in ('chronology','word_sense') and lang=='en' and gi==0,'selected_known_exception':base['case_index'] in known})
    assert len(selected)==64 and sum(r['published'] for r in selected)==2
    save(OUT/'material/groups.json',selected)
    save(OUT/'protocol/frozen.json',{'groups':64,'variants':['base','repeat','other','base_padded','other_padded'],'precisions':['bf16','fp32'],
        'selection':'4groups per8family/2language; up to2first previouslyobserved candidate-prefix discrepancies plus firstremaininggroups. Posthoc numeric diagnostic, not unbiased semantic prevalence test.',
        'controls':'Same source actualtokenprefix. Exactrepeat, futurequery/code variant, and both sequences rightpadded to identicalmaxlength+1 withEOS and explicitzero attentionmask. Realbody tokens have mask1. No donor or weight changes.',
        'all_coordinates':'EverylayerbodyH andallactualMLPproductunits. BF16 uses original fullfield Capture including float64moment instrumentation; FP32 uses samecheckpointvalues and ownsimplebodyhooks, so instrumentation difference explicitlyrecorded.',
        'rounding_boundary':'Both padded states matching while unpadded states differ supports shape-dependent numerics, not backwardsemanticcausation. OriginalBF vs replay disagreement and exactrepeat instability reported separately. FP32 is not exact realarithmetic.',
        'native_forward_total':640,'published_raw_packs':4,'material_sha256':sha(OUT/'material/groups.json'),'source_audit_sha256':sha(OUT.parent/'analysis/source_prefix_audit.json')})
    return selected


def execute(model,tok,precision):
    import torch
    from phase2671_native_mlp_field import Capture,unbits
    groups=prepare();folder=OUT/precision;assert not (folder/'analysis/completion.json').exists();folder.joinpath('field').mkdir(parents=True,exist_ok=True)
    hooks=[];cap=None;captured={};bodypos=0
    if precision=='bf16':cap=Capture(model)
    else:
        def take(key,l,out):
            if isinstance(out,tuple):out=out[0]
            captured.setdefault(key,{})[l]=out.detach()[0,bodypos].float().cpu().numpy().copy()
        hooks.append(model.get_input_embeddings().register_forward_hook(lambda m,a,b:take('h',0,b)))
        for l,block in enumerate(model.model.layers):
            hooks.append(block.register_forward_hook(lambda m,a,b,l=l:take('h',l+1,b)))
            hooks.append(block.mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:take('a',l,a[0])))
    maps={};records=[];raw_manifest=[]
    def measure(r,pad):
        nonlocal bodypos,captured
        bodypos=r['body_end_token'];captured={};ids=list(r['prompt_ids']);mask=[1]*len(ids)
        if pad is not None:ids += [r['eos_token_id']]*(pad-len(ids));mask += [0]*(pad-len(mask))
        if cap:cap.reset(bodypos,False);cap.enabled=True
        input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device)
        em=model.get_input_embeddings()(input_ids)
        device=model.model.layers[0].input_layernorm.weight.device;em=em.to(device)
        output=model.model(inputs_embeds=em,attention_mask=torch.tensor([mask],device=device),use_cache=False)
        if cap:
            cap.enabled=False;pack=cap.pack();result={k:unbits(pack[k])[:,0].copy() for k in ('h','a')};cap.reset(0,False)
        else:result={k:np.stack([v[l] for l in sorted(v)]) for k,v in captured.items()}
        assert all(np.isfinite(v).all() for v in result.values());del output,em,input_ids;return result
    try:
        with torch.inference_mode():
            for group in groups:
                base,other=group['base'],group['other'];variants={}
                for label,r,pad in [('base',base,None),('repeat',base,None),('other',other,None),('base_padded',base,group['pad_length']),('other_padded',other,group['pad_length'])]:variants[label]=measure(r,pad)
                with np.load(FIELD/f'field/case_{base["case_index"]:04d}.npz') as z:original={k:unbits(z[k])[:,0].copy() for k in ('h','a')}
                comparisons={};pairs=[('exact_repeat','base','repeat'),('raw_future_variant','base','other'),('same_shape_future_variant','base_padded','other_padded'),('same_prompt_different_shape','base','base_padded')]
                variants['original']=original;pairs.append(('original_vs_replay','original','base'))
                for label,left,right in pairs:
                    comparisons[label]={}
                    for metric in ('h','a'):
                        a,b=variants[left][metric],variants[right][metric];d=np.abs(a.astype('float64')-b);bits=a.view(np.uint32)!=b.view(np.uint32);key=label+'__'+metric
                        if key+'__max_abs' not in maps:maps[key+'__max_abs']=np.zeros_like(d);maps[key+'__different_bits_count']=np.zeros_like(bits,dtype='int32')
                        np.maximum(maps[key+'__max_abs'],d,out=maps[key+'__max_abs']);maps[key+'__different_bits_count']+=bits
                        comparisons[label][metric]={'different_coordinates':int(bits.sum()),'numerically_different_coordinates':int((d>0).sum()),'max_abs':float(d.max()),'max_abs_by_layer':d.max(-1).tolist(),'different_by_layer':bits.sum(-1).tolist()}
                record={'group':group['index'],'base_case':base['case_index'],'other_case':other['case_index'],'family':base['family'],'language':base['language'],
                    'source_tokens':base['body_end_token']+1,'lengths':[len(base['prompt_ids']),len(other['prompt_ids']),group['pad_length']],
                    'known_exception_selection':group['selected_known_exception'],'comparisons':comparisons};records.append(record)
                if group['published']:
                    path=folder/f'field/group_{group["index"]:02d}.npz';np.savez_compressed(path,**{label+'__'+k:v for label,dd in variants.items() for k,v in dd.items()});raw_manifest.append({'path':str(path),'group':group['index'],'bytes':path.stat().st_size,'published':True})
                save(folder/'analysis/progress.json',{'groups':len(records),'total':64});print('PREFIX REPLAY',precision,len(records),'/64',flush=True)
    finally:
        if cap:cap.close()
        for h in hooks:h.remove()
    folder.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'maps/full_coordinate_comparisons.npz',**maps);save(folder/'analysis/records.json',records);save(folder/'analysis/raw_manifest.json',raw_manifest)
    summary={label:{metric:{'nonzero_groups':sum(r['comparisons'][label][metric]['numerically_different_coordinates']>0 for r in records),'bit_difference_groups':sum(r['comparisons'][label][metric]['different_coordinates']>0 for r in records),'max_abs':max(r['comparisons'][label][metric]['max_abs'] for r in records)} for metric in ('h','a')} for label,_,_ in pairs}
    checks={'64groups_320forwards':len(records)==64,'two_published_packs':len(raw_manifest)==2,'material_immutable':sha(OUT/'material/groups.json')==read(OUT/'protocol/frozen.json')['material_sha256']}
    save(folder/'analysis/completion.json',{'checks':checks,'all_checks_passed':all(checks.values()),'summary':summary,'precision':precision,
        'original_reference':'The original arrays are previouslysavedBF16 in BOTH folders. ForFP32, original_vs_replay mixesprecision/instrumentation differences; onlyBF16original_vs_replay tests sameprecision recovery oftheoldmeasurement.',
        'semantic_or_causal_claim':False});assert all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare']);p.parse_args();prepare();print('64 PREFIX CONTROL GROUPS FROZEN')
