"""Exploratory full-coordinate source-amplitude audit and controlled kernel reanalysis."""
import hashlib,argparse
from rdc_relation_common import *
from rdc_relation_estimators import features_at,PrefixRelations,Bank,select,predict,splits,errors,paired_gain


def main(verify=False):
    out=BASE/'source_normalization';guard(256*1024 if verify else 3*1024**2)
    if (out/'result.json').exists() and not verify:return
    existing=read(out/'reconstructible_fits.json') if verify else None
    if not verify:save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'phase':2718,
      'question':'Does native source-amplitude concentration confound the zero-history result, and does per-source RMS normalization change full-coordinate prediction?',
      'scope':'Exploratory reanalysis designed after viewing Phase2717 raw heatmaps and frozen results. Existing test/fresh are reused, not a new independent confirmation. No new LLM states collected.',
      'normalization':'Only each known past source vector is divided by its own full-coordinate RMS. Current H12 remains unchanged; all coordinates and sources remain. Source scaling is prediction-time observable, not fitted to targets.',
      'candidates':['source_RMS_mean','source_RMS_relation','source_RMS_relation_bilinear','source_RMS_permuted_relation'],
      'controls':'Raw prior candidates selected mix0. New mix0 included; same grid and validation target. Additional effective-df matched controls use frozen raw current baseline df.',
      'energy_diagnostic':'first-token squared norm divided by sum of squared norms across prefix. Separately ratio of squared norm of its message contribution to squared norm of full message: latter need not be <=1 due cross terms.',
      'retention':'All-coordinate error/energy profiles, per-source errors, complete material/rule reconstruction recipe and fitted-array hashes. Dual coefficients are reproducible, not top-K or rank-truncated.'})
    parser=PrefixRelations();data={k:[] for k in ('current','history_mean','relation_binding','source_permuted')};meta=[];target=[];diagnostics=[];profiles={}
    with np.load(BASE/'relation_atlas/training_scales.npz') as z:mu=z['mean'][0];sd=z['standard_deviation'][0]
    for fresh in (False,True):
      for r in rows(fresh):
        z=load_field(r,fresh);h=unbits(z['h12']);h23=unbits(z['h23']);h36=unbits(z['h36'])
        for j,p in enumerate(r['anchors']):
            prefix=h[:p+1];ids=r['prompt_ids'][:p+1];past=prefix[:-1];scaled=prefix.copy();rms=np.sqrt(np.mean(past.astype(float)**2,axis=1));scaled[:-1]=past/np.maximum(rms[:,None],1e-8)
            f=features_at(scaled,ids,r['language'],parser)
            for k in data:data[k].append(f[k])
            target.append(np.concatenate([h23[p],h36[3*j]]));meta.append({k:r[k] for k in ('sample_id','source_group','language','genre')}|{'split':'fresh' if fresh else r['split'],'anchor':j,'position':p})
            pp,_=parser.weights(ids,r['language']);rel=pp[:,1:];weights={'history_mean':np.full((1,len(past)),1/max(len(past),1)),
              'relation_binding':rel.T/(1+rel.sum(0))[:,None]}
            d={'sample_id':r['sample_id'],'split':'fresh' if fresh else r['split'],'language':r['language'],'anchor':j}
            for view,v in [('raw',past),('training_coordinate_z',(past-mu)/sd),('source_RMS',scaled[:-1])]:
                energy=np.sum(v.astype(float)**2,axis=1);d[view+'_first_source_energy_fraction']=float(energy[0]/energy.sum())
                for name,w in weights.items():
                    msg=w@v;first=w[:,0,None]*v[0];den=max(float(np.sum(msg*msg)),1e-30);d[view+'_'+name+'_first_contribution_norm_ratio']=float(np.sum(first*first)/den)
                key=d['split']+'/'+view
                if key not in profiles:profiles[key]={'first':np.zeros(2560,float),'rest':np.zeros(2560,float),'n':0}
                profiles[key]['first']+=v[0].astype(float)**2;profiles[key]['rest']+=np.mean(v[1:].astype(float)**2,axis=0);profiles[key]['n']+=1
            diagnostics.append(d)
    data={k:np.stack(v).astype(np.float32) for k,v in data.items()};y=np.stack(target).astype(np.float32);tr,va,te=splits(meta);fr=np.array([i for i,m in enumerate(meta) if m['split']=='fresh'])
    bank=Bank(data,tr);dd=bank.dots()
    if not verify:
        save(out/'rows.json',meta);save(out/'scales.json',bank.scales);save(out/'source_diagnostics.json',diagnostics)
        npz(out/'all_coordinate_source_energy.npz',**{key.replace('/','_')+'_'+name:p[name]/p['n'] for key,p in profiles.items() for name in ('first','rest')})
    df=read(BASE/'rules/current/result.json')['effective_df'];reports={};fits={};coord={};baseline={}
    with np.load(BASE/'rules/current/predictions.npz') as z:baseline['test']=z['test']
    with np.load(BASE/'confirmation/current_current.npz') as z:baseline['fresh']=z['prediction']
    names={'source_RMS_mean':'history_mean','source_RMS_relation':'relation_binding','source_RMS_relation_bilinear':'relation_bilinear','source_RMS_permuted_relation':'source_permuted'}
    for name,kind in names.items():
      for matched in (False,True):
        label=name+('_matched_df' if matched else '');model,info,grid=select(dd,kind,y,tr,va,[(0,2560),(2560,5120)],fixed_df=df if matched else None)
        fit={'kernel_kind':kind,**info,'validation_grid':grid,'target_blocks':[[0,2560],[2560,5120]],'fixed_df':df if matched else None,
          'array_sha':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in model.items()}}
        fits[label]=fit;report={}
        if verify:
            assert fit['array_sha']==existing[label]['array_sha'],('Recomputed coefficients differ',label);print('SOURCE_RMS_COEFFICIENTS_RECOMPUTED',label,flush=True);continue
        for split,idx in [('test',te),('fresh',fr)]:
            gram=Bank.gram(dd,kind,info['mix'])[np.ix_(idx,tr)];p=predict(model,gram);e=(p-y[idx])**2;bm=baseline[split];mm=[meta[i] for i in idx]
            report[split]={'anchors':len(idx),'H23_MSE':float(e[:,:2560].mean()),'H36_MSE':float(e[:,2560:].mean()),
              'raw_current_minus_candidate_H36':paired_gain((bm[:,2560:]-y[idx,2560:])**2,e[:,2560:],mm)}
            coord[label+'_'+split]=e.mean(0);save(out/f'{label}_{split}_source_errors.json',[{k:m[k] for k in ('sample_id','source_group','language','anchor')}|{'H23_MSE':float(e[j,:2560].mean()),'H36_MSE':float(e[j,2560:].mean())} for j,m in enumerate(mm)])
        reports[label]={'selection':info,'evaluation':report};print('SOURCE_RMS',label,info,report,flush=True)
    if verify:
        save(out/'recomputation_audit.json',{'timestamp':stamp(),'passed':True,'all_fitted_arrays_hash_equal':True,'fits':list(fits),'source':snapshot(Path(__file__))});return
    npz(out/'all_coordinate_prediction_MSE.npz',**coord);save(out/'reconstructible_fits.json',fits)
    values={}
    for split in ('train','validation','test','fresh'):
        rr=[r for r in diagnostics if r['split']==split];values[split]={k:{'mean':float(np.mean([r[k] for r in rr])),'median':float(np.median([r[k] for r in rr])),'p05_p95':np.quantile([r[k] for r in rr],[.05,.95]).tolist()} for k in rr[0] if k not in ('sample_id','split','language','anchor')}
    save(out/'result.json',{'timestamp':stamp(),'reports':reports,'source_energy':values,'fitted_arrays_persisted':False,'exact_reconstruction':'Run this archived script through identical feature construction and select(...fixed_df...) using saved scales, training IDs and native fields; fitted array SHA controls arithmetic identity.',
      'interpretation_limit':'A posthoc source-normalization result changes a candidate representation, not native attention or hidden states. It cannot prove that native high-norm sources are irrelevant, that all history is unnecessary, or that the extracted relation map is semantically complete.'});guard(1024**2)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):main(a.verify)
