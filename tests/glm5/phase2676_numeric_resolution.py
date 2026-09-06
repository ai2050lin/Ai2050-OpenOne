"""Same FP32 core, FP64 readout and frozen dose ladder. Not a full-FP64-model test."""
import argparse,gc,os
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
import phase2674_native_mlp_scalar as engine
from phase2666_multitoken_parameter_engine import branch_inputs,PARTS,load_fp
from phase2624_scalar_forward_validation import digest_tensor

OUT=RESULT/'phase2676_native_mlp_delivery/numeric_resolution'


def effect_summary(effects):
    assert effects
    n=len(effects);den=sum(abs(e['effect64']) for e in effects)
    error=sum(abs(e['effect64']-e['predicted64']) for e in effects)
    return {'n':n,'sum_abs_effect64':den,'mean_abs_effect64':den/n,
            'mean_abs_prediction64':sum(abs(e['predicted64']) for e in effects)/n,
            'mean_abs_error64':error/n,'relative_L1_error64':error/den if den else None,
            'ratio_defined':den>0}

def prepare():
    if (OUT/'protocol/frozen.json').exists():return read(OUT/'material/cases.json'),read(OUT/'protocol/frozen.json')
    cases=[r for r in read(engine.OUT/'material/cases.json') if r['published']];source=read(engine.OUT/'protocol/frozen.json');sites=source['sites']
    assert len(cases)==16;indices=[i for i,s in enumerate(sites) if s['control']=='ordinary'];assert len(indices)==15
    save(OUT/'material/cases.json',cases);plan={'prefixes':16,'sites':sites,'indices':indices,'scales':[1,4,16],'conditions':1440,
        'trigger':'Interim2674 completed41prefixes shows meanabsscalareffects near1e-5 with large relative finiteprediction errors. Original128prefix contract unchanged; all16previouslypublishedprefixes chosen independently ofnumericoutcomes.',
        'precision':'Transformer blocks remainFP32, TF32off; actual checkpointvalues unchanged. Two fullvocabulary readouts from identicalhiddenstates: FP32 andFP64 matrixaccumulation/logsoftmax. This does NOT establish fullFP64core accuracy.',
        'dose':'For15pre-frozenordinarysites, +/- originaldelta times1,4,16. 16x correspondsabout1.6vectorRMS, not a microscopic perturbation; quantifycurvature, do not relabel as infinitesimal.',
        'validation':'Recompute fulltoken nativegradients for FP64readout objective. Actualmemoryweights changed oneatatime, finallyrestore plus12fullmatrixhashes eachprefix. Compare total/content/format/EOS and legacyFP32 effects.',
        'source_protocol_sha256':sha(engine.OUT/'protocol/frozen.json')};save(OUT/'protocol/frozen.json',plan);return cases,plan

def both_values(model,r,bi,grad):
    em,mask,ids,targets=branch_inputs(model,r,bi,grad);out=model.model(inputs_embeds=em,attention_mask=mask,use_cache=False)
    start=len(r['prompt_ids'])-1;states=out.last_hidden_state[0,start:start+len(targets)];cpu=states.cpu();idx=torch.arange(len(targets));target=torch.tensor(targets)
    logits64=cpu.double()@model._resolution_head64.T;lp64=logits64.log_softmax(-1)[idx,target]
    # FP32 side is only a finite-forward diagnostic; do not attach an unnecessary graph to it.
    with torch.no_grad():lp32=(cpu.detach()@model.lm_head.weight.T).log_softmax(-1)[idx,target]
    return em,states,lp64,lp32,ids,targets

def values64(model,r,bi,grad=False):
    em,states,lp,_,ids,targets=both_values(model,r,bi,grad);return em,states,lp,ids,targets

@torch.inference_mode()
def scores(model,r):
    branches=[]
    for bi in (0,1):
        _,_,lp64,lp32,_,_=both_values(model,r,bi,False);a,b=lp64.tolist(),lp32.tolist();cats=r['answer_token_categories'][bi]
        branches.append({'total64':sum(a),'total32':sum(b),'lp64':a,'lp32':b,**{p:sum(v for v,c in zip(a,cats) if c==p) for p in PARTS}})
    return {'branches':branches,**{p:branches[0][p]-branches[1][p] for p in ('total64','total32',)+PARTS}}

def execute():
    assert (RESULT/'phase2675_native_mlp_crossmodel/analysis/final.json').exists();assert not (OUT/'analysis/completion.json').exists()
    cases,plan=prepare();os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';model,info=load_fp();model._resolution_head64=model.lm_head.weight.detach().double();save(OUT/'protocol/model.json',info)
    sites=plan['sites'];matrices={(s['layer'],s['kind']):engine.module(model,s).weight for s in sites};before={f'{l}_{k}':digest_tensor(w) for (l,k),w in matrices.items()};oldvalues=engine.values;engine.values=values64
    path=OUT/'analysis/records.jsonl';path.parent.mkdir(parents=True,exist_ok=True);records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]]
    try:
        with path.open('a',encoding='utf-8') as stream:
            for r in cases[len(records):]:
                base=scores(model,r);gr,raw,meta=engine.native_gradients(model,{**r,'published':False},sites);noop=abs(base['total64']-meta['branches'][0]['total']+meta['branches'][1]['total']);assert noop<1e-7;del raw;effects=[]
                for si in plan['indices']:
                    s=sites[si];w=engine.module(model,s).weight;original=w[s['j'],s['k']].detach().clone()
                    for scale in plan['scales']:
                        for sign in (-1,1):
                            try:
                                with torch.no_grad():w[s['j'],s['k']]=float(original)+scale*(s['targets'][str(sign)]-s['original'])
                                delta=float(w[s['j'],s['k']])-float(original);now=scores(model,r)
                                effects.append({'site_index':si,'kind':s['kind'],'layer':s['layer'],'unit':s['unit'],'scale':scale,'sign':sign,'delta':delta,
                                    'effect64':now['total64']-base['total64'],'effect32':now['total32']-base['total32'],'predicted64':float(delta*gr['all'][si]),
                                    'parts':{p:{'effect64':now[p]-base[p],'predicted64':float(delta*gr[p][si])} for p in PARTS}})
                            finally:
                                with torch.no_grad():w[s['j'],s['k']].copy_(original)
                after={f'{l}_{k}':digest_tensor(w) for (l,k),w in matrices.items()};assert after==before
                rec={'case_index':r['case_index'],'family':r['family'],'language':r['language'],'base':base,'noop64':noop,'gradients64':{k:v.tolist() for k,v in gr.items()},'effects':effects,'whole_matrix_hashes':after}
                stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec);save(OUT/'analysis/progress.json',{'prefixes':len(records),'total':16});print('READOUT RESOLUTION',len(records),'/16',flush=True)
    finally:engine.values=oldvalues
    from phase2676_prefix_replay import execute as prefix_replay
    prefix_replay(model,None,'fp32')
    after={f'{l}_{k}':digest_tensor(w) for (l,k),w in matrices.items()};assert after==before
    del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/records.json',records);save(OUT/'analysis/restoration.json',{'before':before,'after':after})
    summary={}
    for scale in plan['scales']:
        rr=[e for r in records for e in r['effects'] if e['scale']==scale]
        summary[str(scale)]={**effect_summary(rr),
            'mean_abs_effect32_minus64':sum(abs(e['effect32']-e['effect64']) for e in rr)/len(rr),
            'parts':{p:effect_summary([e['parts'][p] for e in rr]) for p in PARTS}}
    checks={'16prefixes':len(records)==16,'1440actualchanges':sum(len(r['effects']) for r in records)==1440,'all_fullmatrix_hashes_restored':all(r['whole_matrix_hashes']==before for r in records),'original_protocol_immutable':sha(engine.OUT/'protocol/frozen.json')==plan['source_protocol_sha256']}
    assert all(checks.values());save(OUT/'analysis/completion.json',{'checks':checks,'all_checks_passed':True,'summary':summary,'boundary':'FP32core+FP64readout, not fullFP64network. Largestdose1.6vectorRMS can benonlinear. Numericexception analysis, not languagegearclosure.'});print('NUMERIC RESOLUTION COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);a=p.parse_args();globals()[a.action]()
