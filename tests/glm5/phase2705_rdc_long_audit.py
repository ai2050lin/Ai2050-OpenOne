"""Post-hoc stratification and capacity sensitivity; never revises frozen main outcomes."""
from phase2705_rdc_long_analysis import *


def main():
    rows,data=extract();tr,va,te=splits(rows)
    immutable(OUT/'extra_audit_protocol.json',{'source_sha':sha(Path(__file__)),
      'status':'post-hoc descriptive sensitivity after main result; not a new untouched confirmation',
      'gate':'Frozen I coefficients on every K selected state, split by family × input language × stage. All-unit MSE and energy, prefix paired preference; requested output language differs only for translation.',
      'history':'Compare same nominal 5120-coordinate H12+previousH0 control against H12+previousH36. Also fix effective degrees of freedom to128 for H12 alone and H12+previousH36. Full H36 prediction only; no new vocabulary decode during serial large-model GPU capture.',
      'limits':'Prior H0 controls token identity partly, not all information/capacity. Degrees of freedom constrain spectral shrinkage, not identical hypothesis classes. Only2testentitygroups.'})
    g,u,a=[unbits(data[k]).astype(np.float64) for k in ('L23_gate','L23_up','L23_a')]
    root=CAMPAIGN/'j_predictive_gates/unit_errors'
    with np.load(root/'L23_weighted_global.npz') as z:cg=z['coefficient'][0]
    with np.load(root/'L23_weighted_language.npz') as z:cl=z['coefficient']
    inp=np.array([r['language']=='zh' for r in rows],int);out=np.array([1-i if r['family']=='translation' else i for r,i in zip(rows,inp)])
    err=np.stack([np.mean((g*u*c-a)**2,1) for c in (cg,cl[inp],cl[out])],1);energy=np.mean(a*a,1)
    cells=[]
    for f in sorted({r['family'] for r in rows}):
      for lang in ('en','zh'):
       for stage in ('prefill','early','later'):
        mask=np.array([r['family']==f and r['language']==lang and ('prefill' if r['generation_step']==0 else 'early' if r['generation_step']<16 else 'later')==stage for r in rows])
        if mask.any():cells.append({'family':f,'input_language':lang,'stage':stage,'states':int(mask.sum()),'mse_global_input_output':err[mask].mean(0).tolist(),'target_energy':float(energy[mask].mean())})
    prefix=[]
    for pid in sorted({r['prefix_id'] for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if r['prefix_id']==pid]);r=rows[ix[0]]
        prefix.append({'prefix_id':pid,'family':r['family'],'language':r['language'],'word_split':r['word_split'],'unit':r['unit'],
          'states':len(ix),'mse_global_input_output':err[ix].mean(0).tolist()})
    npz(OUT/'unit_errors/transfer_state_errors.npz',mse_global_input_output=err,target_energy=energy)
    del g,u,a;gc.collect()
    previous_h0=[]
    for r in rows:
        if r['generation_step']==0:previous_h0.append(np.zeros(2560,np.uint16))
        else:
            with np.load(OUT/f'fields/{r["prefix_id"]}-s{r["generation_step"]-1}.npz') as z:previous_h0.append(z['h_c'][0])
    h12=unbits(data['H12']);prior=unbits(data['previous_H36']);prior0=unbits(np.stack(previous_h0));y=unbits(data['H36']);fits=[]
    for name,blocks in [('H12_previousH0',[h12,prior0]),('H12_fixeddf128',[h12]),('H12_previousH36_fixeddf128',[h12,prior])]:
        scales=[max(float(np.sqrt(np.mean(np.sum(np.asarray(b[tr],np.float64)**2,1)))),1e-12) for b in blocks];x=np.concatenate([b/s for b,s in zip(blocks,scales)],1)
        for kind in ('linear','quadratic'):
            f=FullKernel(x,tr,va,te,kind)
            if name.endswith('fixeddf128'):
                lo,hi=1e-12,1e8
                for _ in range(100):
                    mid=(lo*hi)**.5
                    if np.sum(f.e/(f.e+mid))>128:lo=mid
                    else:hi=mid
                ridge=(lo*hi)**.5;p=(f.tq@((f.q.T@y[tr])/(f.e[:,None]+ridge))).astype(np.float32)
                m={'ridge':ridge,'effective_degrees_of_freedom':float(np.sum(f.e/(f.e+ridge)))}
            else:p,m=f.fit(y,OUT/f'models/audit_{name}_{kind}.npz')
            e=np.mean((p.astype(np.float64)-y[te])**2,1)
            fits.append({'model':name+'_'+kind,**m,'mse':float(e.mean()),'test_states':len(te),
              'by_entity':{str(k):float(e[[rows[i]['unit']==k for i in te]].mean()) for k in (6,7)}})
            npz(OUT/f'predictions/audit_{name}_{kind}.npz',prediction=p,target=y[te],test=te)
            print('LONG_EXTRA',name,kind,float(e.mean()),flush=True)
    main=read(OUT/'result.json');summary={'phase':2705,'timestamp':stamp(),'prefixes':len(prefix),'selected_states':len(rows),
      'scores_by_family':main['scores_by_family'],'vocabulary':[dict(model=r['model'],mean_KL=r['mean_KL'],argmax_matches=r['argmax_matches'],states=r['states']) for r in main['full_vocabulary']],
      'gate_cells':cells,'gate_prefixes':prefix,'history_sensitivity':fits,'interpretation':'Post-hoc descriptive/capacity sensitivity; retain main results and two independent entity groups caveat.'}
    save(OUT/'summary_audit.json',summary)
    print('LONG_EXTRA_COMPLETE',flush=True)


if __name__=='__main__':main()
