"""Paired behavior, all-coordinate readers, frozen gates, and full native source-group vectors."""
import gc
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel
from rdc_order_material import ORDERS,FAMILIES
OUT=CAMPAIGN/'m_order'
SOURCE_GROUPS=('record','prompt_other','generated_trace','generated_neutral','generated_result','generated_other')


def report(y,p,indices,rows):
    y=y[indices];p=np.asarray(p);correct=(p[:,0]>.5)==y[:,0]
    return {'mse':float(np.mean((p-y)**2)),'accuracy':float(correct.mean()),'correct':int(correct.sum()),'n':len(indices),
      'by_order':{o:{'n':int(mask.sum()),'correct':int(correct[mask].sum())} for o in ORDERS if (mask:=np.array([rows[i]['order']==o for i in indices])).any()},
      'by_family':{f:{'n':int(mask.sum()),'correct':int(correct[mask].sum())} for f in FAMILIES if (mask:=np.array([rows[i]['family']==f for i in indices])).any()}}


def source_account(rows,behavior):
    weights={l:checkpoint(f'model.layers.{l}.self_attn.o_proj.weight').float().numpy().astype(np.float64) for l in (11,23,35)}
    wd=checkpoint('model.layers.23.mlp.down_proj.weight').float().numpy().astype(np.float64);accounts=[]
    for i,(r,b) in enumerate(zip(rows,behavior)):
        sid=b['result_boundary_state']
        if sid is None:continue
        row=read(OUT/f'steps/{sid}.json');labels=np.array(row['source_labels']);fp=OUT/f'fields/{sid}.npz'
        assert len(labels)==len(row['prompt_ids']) and set(labels).issubset(SOURCE_GROUPS)
        with np.load(fp) as z:
            h=unbits(z['h_c']).astype(np.float64);a23=unbits(z['L23_a']).astype(np.float64);down23=unbits(z['L23_down']).astype(np.float64)
            native_down=a23@wd.T;down_round=down23-native_down;ledger={};layers=[]
            for l,wo in weights.items():
                p,v,head,actual=[unbits(z[f'L{l}_{k}']).astype(np.float64) for k in ('p','v','head_output','attention_out')]
                expanded=v[np.arange(32)//4];group_heads=[];mass=[]
                for group in SOURCE_GROUPS:
                    mask=labels==group
                    group_heads.append(np.einsum('hs,hsd->hd',p[:,mask],expanded[:,mask]).reshape(-1))
                    mass.append(float(p[:,mask].sum()/32))
                group_heads=np.stack(group_heads);vectors=group_heads@wo.T
                head_round=(head-group_heads.sum(0))@wo.T;o_round=actual-head@wo.T
                recon=vectors.sum(0)+head_round+o_round;error=float(np.max(np.abs(recon-actual)));assert error<1e-8
                norm2=max(float(actual@actual),1e-30);shares=(vectors@actual/norm2).tolist()
                down=unbits(z[f'L{l}_down']).astype(np.float64);residual_round=h[l+1]-h[l]-actual-down
                ledger.update({f'L{l}_source_vectors':vectors,f'L{l}_source_heads':group_heads,f'L{l}_matmul_round_vector':head_round,
                  f'L{l}_o_round_vector':o_round,f'L{l}_actual_attention':actual,f'L{l}_residual_add_round':residual_round})
                layers.append({'layer':l,'source_tokens':len(labels),'source_group_counts':{g:int((labels==g).sum()) for g in SOURCE_GROUPS},
                  'mean_head_attention_mass':dict(zip(SOURCE_GROUPS,mass)),'projection_share_along_actual_attention':dict(zip(SOURCE_GROUPS,shares)),
                  'group_vector_squared_norm':dict(zip(SOURCE_GROUPS,np.square(vectors).sum(1).tolist())),
                  'actual_attention_squared_norm':norm2,'matmul_round_rms':float(np.sqrt(np.mean(head_round**2))),'o_round_rms':float(np.sqrt(np.mean(o_round**2))),
                  'account_max_abs_error':error,'residual_add_round_rms':float(np.sqrt(np.mean(residual_round**2)))})
            ledger.update(L23_native_down_fp64=native_down,L23_down_round=down_round)
            npz(OUT/f'ledgers/{sid}.npz',**ledger)
        account={'sample_id':sid,'prefix_id':r['sample_id'],'family':r['family'],'language':r['language'],'unit':r['unit'],'order':r['order'],
          'generation_step':row['generation_step'],'result_correct':b['scores']['result_correct'],'layers':layers,
          'L23_down_round_rms':float(np.sqrt(np.mean(down_round**2)))}
        save(OUT/f'accounts/{sid}.json',account);accounts.append(account)
        if i%24==0:print('ORDER_SOURCE',i,len(rows),flush=True)
    summary=[]
    for l in (11,23,35):
      for f in FAMILIES:
       for order in ORDERS:
        rr=[next(x for x in a['layers'] if x['layer']==l) for a in accounts if a['family']==f and a['order']==order]
        if rr:summary.append({'layer':l,'family':f,'order':order,'prefixes':len(rr),
          'mean_head_attention_mass':{g:float(np.mean([x['mean_head_attention_mass'][g] for x in rr])) for g in SOURCE_GROUPS},
          'mean_projection_share':{g:float(np.mean([x['projection_share_along_actual_attention'][g] for x in rr])) for g in SOURCE_GROUPS}})
    save(OUT/'source_result.json',{'source_groups':list(SOURCE_GROUPS),'accounts':accounts,'summary':summary,
      'max_abs_account_error':max(a['account_max_abs_error'] for r in accounts for a in r['layers']),
      'limits':['Real V/P/Wo composition with observed conditions, not causal necessity.','Signed projection shares may be negative or exceed1; they are not independent probability or importance fractions.','Source field labels describe text positions, not unique latent semantic modules.','Changes in prompt instruction, position, generated text, norms andattentionallcooccur; neutral control does not exactly equalize tokenlength.']})
    return summary


def main():
    rows=read(OUT/'prefixes.json');assert len(list((OUT/'prefix_commits').glob('*.json')))==288
    immutable(OUT/'analysis_protocol_v2.json',{'phase':2707,'source_sha':sha(Path(__file__)),'estimator_sha':sha(ROOT/'tests/glm5/rdc_conditional_estimators.py'),
      'implementation_correction':'Initial analysis stopped after first H0 fit because dict() received duplicate mse/n keyword arguments from fit metadata and scores. Use explicit dictionary merge; no input, split, model fit, scoring or native capture changed. Original analysis_protocol.json preserved.',
      'prior_protocol_sha':sha(OUT/'analysis_protocol.json'),
      'behavior':'Paired per-record results for3orders, including allfailures and truncations; byfamily/lang, no independenttokenconfidence claims.',
      'readers':'Fullcoordinate C all37 layers linear, H12/24/36 quadratic; stage-prefill and Result-field onset separate. Use pilot-corrected byte-piece onset/parsed scores fromscoring_alignment_protocol_v2, rawcapture unchanged. Numeric task onset precedes firstname, not first numeric disambiguation. Binarylabel is externally defined truth/direction state, not completeanswer correctness. Entitiesgrouped4train/2val/2test; malformed-label missing boundaries excludedonlyinboundarymodels andreported.',
      'transfer':'One result-first prefill H24/H36 reader frozen across3orders atprefill andatResultboundary. Boundarycomparison is conditional observation withmodel-specific generatedhistory, not a deployable futuretimestep selector.',
      'gates':'Frozen I weightedglobal/language coefficients appliedtoobserved M L23 g/up atprefill/Resultboundary, allunits andenergy, no claim ofadvanceprediction.',
      'sources':'For everyselected Result-boundary, allsource tokens/32heads/128headdim/nativeWo coordinates summed intosix preregistered source-text groups atL11/23/35. Full2560 group-vectors androunding vectors retained. No top-K source selection.',
      'limits':'Controlinstructions andnative executionlength differ. Only8globalentitygroups,2heldout; generatedTrace can itselfbe incorrect. Labelreadability can reflectexplicitprioroutput, not nativecorrectinference. No proof of necessity, unique algorithm, or causal mediation.'})
    behavior=[read(OUT/f'behavior_scored/{r["sample_id"]}.json') for r in rows];assert all(b['scores']['score_version']==2 for b in behavior);scores=[];pairs=[]
    keys=('result_correct','trace_correct','neutral_correct','all_content_correct','field_order_correct','format_structure','eos','truncated')
    for f in FAMILIES:
     for language in ('en','zh'):
      for order in ORDERS:
        rr=[b for r,b in zip(rows,behavior) if (r['family'],r['language'],r['order'])==(f,language,order)]
        scores.append(dict(family=f,language=language,order=order,n=len(rr),**{k:sum(b['scores'][k] for b in rr) for k in keys},
          mean_steps=float(np.mean([b['steps'] for b in rr])),missing_result_boundary=sum(b['result_boundary_state'] is None for b in rr)))
    for bid in sorted({r['base_id'] for r in rows}):
        values={r['order']:b['scores'] for r,b in zip(rows,behavior) if r['base_id']==bid};pairs.append({'base_id':bid,**{k:{o:values[o][k] for o in ORDERS} for k in ('result_correct','trace_correct','all_content_correct')}})
    save(OUT/'behavior_result.json',{'timestamp':stamp(),'prefixes':288,'paired_records':96,'scores':scores,'pairs':pairs})
    allfeatures={};results=[];gate_reports=[]
    cg=read(CAMPAIGN/'j_predictive_gates/protocol.json')
    with np.load(CAMPAIGN/'j_predictive_gates/unit_errors/L23_weighted_global.npz') as z:global_c=z['coefficient'][0]
    with np.load(CAMPAIGN/'j_predictive_gates/unit_errors/L23_weighted_language.npz') as z:lang_c=z['coefficient']
    for stage in ('prefill','result_boundary'):
        ids=[i for i,b in enumerate(behavior) if stage=='prefill' or b['result_boundary_state'] is not None];rr=[rows[i] for i in ids];h=[];native={k:[] for k in ('gate','up','a')};steps=[]
        for i in ids:
            sid=rows[i]['sample_id']+'-s0' if stage=='prefill' else behavior[i]['result_boundary_state']
            with np.load(OUT/f'fields/{sid}.npz') as z:
                h.append(z['h_c'])
                for k in native:native[k].append(z['L23_'+k])
            steps.append(read(OUT/f'steps/{sid}.json')['generation_step'])
        hb=np.stack(h);h=unbits(hb);tr,va,te=splits(rr);y=np.array([[r['truth']] for r in rr],np.float32)
        npz(OUT/f'features/{stage}.npz',h_c=hb,original_indices=np.array(ids),steps=np.array(steps),**{k:np.stack(v) for k,v in native.items()})
        allfeatures[stage]=(rr,h,steps,ids)
        for l in range(37):
          for kind in (('linear','quadratic') if l in (12,24,36) else ('linear',)):
            f=FullKernel(h[:,l],tr,va,te,kind);mid=f'{stage}_H{l}_{kind}';p,m=f.fit(y,OUT/f'models/{mid}.npz')
            results.append({'stage':stage,'H':l,'algorithm':kind,'target':'external_binary_state',**m,**report(y,p,te,rr)})
            npz(OUT/f'predictions/{mid}.npz',prediction=p,target=y[te],test=te)
        g,u,a=[unbits(np.stack(native[k])).astype(np.float64) for k in ('gate','up','a')];energy=np.mean(a*a,0)
        for name,c in [('global',global_c),('language',lang_c[np.array([r['language']=='zh' for r in rr],int)])]:
            err=(g*u*c-a)**2;npz(OUT/f'unit_errors/{stage}_old_{name}.npz',coordinate_mse=err.mean(0),energy=energy)
            for order in ORDERS:
                mask=np.array([r['order']==order for r in rr]);gate_reports.append({'stage':stage,'old_gate':name,'order':order,'prefixes':int(mask.sum()),'all_unit_mse':float(err[mask].mean()),'energy_ratio':float(err[mask].sum()/max(np.sum(a[mask]**2),1e-30))})
        del native,g,u,a,err;gc.collect();print('ORDER_READERS',stage,len(rr),flush=True)
    transfer=[];sr,sh,_,_=allfeatures['prefill'];tr,va,_=splits(sr);tr=tr[[sr[i]['order']=='result_first' for i in tr]];va=va[[sr[i]['order']=='result_first' for i in va]]
    for stage,(rr,h,_,_) in allfeatures.items():
        _,_,te=splits(rr);combined=sr+rr;y=np.array([[r['truth']] for r in combined],np.float32)
        for l in (24,36):
            x=np.concatenate([sh[:,l],h[:,l]],0);test=te+len(sr);f=FullKernel(x,tr,va,test)
            p,m=f.fit(y,OUT/f'models/transfer_prefill_resultfirst_H{l}_{stage}.npz');transfer.append({'stage':stage,'H':l,**m,**report(y,p,test,combined)})
    source=source_account(rows,behavior)
    save(OUT/'result.json',{'phase':2707,'timestamp':stamp(),'prefixes':288,'states':len(read(OUT/'material.json')),
      'scores':scores,'paired_behavior':pairs,'readers':results,'frozen_resultfirst_prefill_readers':transfer,'prospective_gates':gate_reports,
      'native_source_summary':source,'limits':read(OUT/'analysis_protocol_v2.json')['limits']})
    announce('m_order',state='analysis_complete',completed=288,total=288,states=len(read(OUT/'material.json')))
    print('ORDER_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
