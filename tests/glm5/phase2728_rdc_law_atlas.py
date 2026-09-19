"""Full-coordinate observational relationship atlas and nonorthogonal joint-state ledger."""
from collections import Counter, defaultdict
from rdc_law_common import *


def collect():
    rows=gzread(BASE/'material.json.gz');meta=[];hidden=[]
    data={b:defaultdict(list) for b in (6,16,35)}
    for r in rows:
        with np.load(BASE/'capture/main/fields'/f'{r["sample_id"]}.npz') as z:
            hidden.append(np.transpose(unbits(z['H']),(1,0,2)))
            for ai,p in enumerate(r['anchors']):
                graph=r.get('retrospective_graph',r.get('graph',[]))
                visible=[edge for edge in graph if edge.get('available_after_token',p+1)<=p]
                labelset=sorted({edge['type'] for edge in visible})
                meta.append({'id':r['sample_id']+f'_a{ai}','sample_id':r['sample_id'],'anchor_index':ai,'position':p,
                    'source_group':r['source_group'],'split':r['split'],'cohort':r['cohort'],'kind':r['kind'],'language':r['language'],
                    'genre':r['genre'],'token_id':r['prompt_ids'][p],'prefix_visible_gold_types':labelset,
                    'retrospective_full_window_types':r.get('relation_types',[]),
                    'label_scope':'Gold relations restricted to observed endpoint positions for analysis; annotations can depend on full sentence, therefore NEVER used as online predictor features.'})
            for b in data:
                for k in ('x','gate','up','activation','mlp'):
                    data[b][k].append(unbits(z[f'L{b}_{k}']))
    return rows,meta,np.concatenate(hidden),{b:{k:np.concatenate(v) for k,v in d.items()} for b,d in data.items()}


def main():
    out=BASE/'atlas'
    if (out/'result.json').exists():return
    assert (BASE/'capture/main/result.json').exists()
    start=time.monotonic();guard(700*1024**2)
    rows,meta,H,data=collect();train=np.array([i for i,r in enumerate(meta) if r['split']=='train'])
    mean=H[train].astype(float).mean(0);std=H[train].astype(float).std(0)
    npz(out/'training_coordinate_rulers.npz',mean=mean,std=std)
    masks={}
    for cohort in sorted({r['cohort'] for r in meta}):masks['cohort:'+cohort]=np.array([r['cohort']==cohort for r in meta])
    counts=Counter(t for r in meta for t in r['prefix_visible_gold_types'])
    for typ,n in sorted(counts.items()):
        if n>=32:masks['gold_endpoint_visible:'+typ]=np.array([typ in r['prefix_visible_gold_types'] for r in meta])
    profiles=[]
    for name,mask in masks.items():
        hh=H[mask].astype(float)
        raw=hh.mean(0);rms=np.sqrt(np.mean(hh**2,axis=-1,keepdims=True));unit=(hh/np.maximum(rms,1e-12)).mean(0)
        z=((hh-mean)/np.maximum(std,1e-6)).mean(0)
        key=rank(name)[:16]
        npz(out/'condition_profiles'/f'{key}.npz',raw_mean=raw,RMS_normalized_mean=unit,train_z_mean=z,
            raw_std=hh.std(0),coordinate_rms=np.sqrt(np.mean(hh**2,axis=0)))
        profiles.append({'id':key,'condition':name,'anchors':int(mask.sum()),'source_groups':len({meta[i]['source_group'] for i in np.flatnonzero(mask)}),
            'path':str((out/'condition_profiles'/f'{key}.npz').relative_to(BASE)),
            'scope':'Observational complete coordinate profile. No nuisance-adjusted semantic effect, no forced per-coordinate role assignment.'})
        del hh
    # All entries of two early-to-native-writeback covariance matrices per natural domain.
    covariance=[]
    for cohort in ('gum','ewt','cmrc'):
        ix=np.array([i for i,r in enumerate(meta) if r['split']=='train' and r['cohort']==cohort])
        a=H[ix,12].astype(float);a-=a.mean(0)
        for b in (16,35):
            y=data[b]['mlp'][ix].astype(float);y-=y.mean(0)
            cov=a.T@y/len(ix)
            scale=np.sqrt(np.maximum((a*a).mean(0),1e-20))[:,None]*np.sqrt(np.maximum((y*y).mean(0),1e-20))[None,:]
            corr=cov/scale
            path=out/'full_coordinate_relations'/f'{cohort}_H12_to_L{b}_MLP.npz'
            npz(path,covariance=cov.astype(np.float32),correlation=corr.astype(np.float32))
            covariance.append({'cohort':cohort,'block':b,'anchors':len(ix),'shape':list(cov.shape),'path':str(path.relative_to(BASE)),
                'covariance_frobenius':float(np.linalg.norm(cov)),'mean_abs_correlation':float(np.mean(abs(corr))),
                'meaning':'Complete empirical cross-coordinate association. Every native coordinate pair retained. Not a parameter connection or causal edge.'})
            del cov,corr,y,scale
    # Center-dependent complete finite-product expansion using all units and all write coordinates.
    from rdc_law_native import parameter
    ledgers=[]
    representatives=[next(i for i,r in enumerate(meta) if r['cohort']==cohort) for cohort in sorted({r['cohort'] for r in meta})]
    for b,d in data.items():
        g=d['gate'].astype(float);u=d['up'].astype(float)
        phi=g*np.exp(-np.logaddexp(0,-g))
        pm=phi[train].mean(0);um=u[train].mean(0)
        weight=parameter(f'model.layers.{b}.mlp.down_proj.weight','cpu').numpy()
        grams=[];errors=[];selected=[]
        for at in range(0,len(meta),96):
            aa=phi[at:at+96];uu=u[at:at+96];da=aa-pm;du=uu-um
            terms=np.stack([np.broadcast_to(pm*um,aa.shape),pm*du,da*um,da*du]).astype(np.float32)
            writes=terms@weight.T
            summed=writes.sum(0).astype(float)
            direct=(aa*uu).astype(np.float32)@weight.T
            gram=np.einsum('tnd,snd->nts',writes.astype(float),writes.astype(float))/H.shape[-1]
            reconstructed=gram.sum((1,2));native=(d['mlp'][at:at+96].astype(float)**2).mean(-1)
            error=np.stack([np.mean((direct-summed)**2,axis=-1)/np.maximum(np.mean(direct.astype(float)**2,axis=-1),1e-20),
                np.mean((summed-d['mlp'][at:at+96])**2,axis=-1)/np.maximum(native,1e-20),native,reconstructed],-1)
            grams.append(gram);errors.append(error)
            for ri in representatives:
                if at<=ri<at+len(aa):selected.append((ri,writes[:,ri-at].copy()))
            del terms,writes,direct,summed,aa,uu,da,du
        grams=np.concatenate(grams);errors=np.concatenate(errors)
        normalized=grams/np.maximum(grams.sum((1,2)),1e-20)[:,None,None]
        assert np.max(errors[:,0])<1e-8,('Expansion arithmetic failure',b,float(np.max(errors[:,0])))
        assert np.max(abs(normalized.sum((1,2))-1))<1e-8
        path=out/'joint_products'/f'L{b}_complete_unit_ledger.npz'
        npz(path,phi_center=pm,up_center=um,mean_product= (phi[train]*u[train]).mean(0),
            phi_up_covariance_diagonal=((phi[train]-pm)*(u[train]-um)).mean(0),signed_Gram=grams,
            normalized_signed_Gram=normalized,numeric_errors=errors,representative_indices=np.array([i for i,_ in selected]),
            representative_all_coordinate_writes=np.stack([w for _,w in selected]))
        reports=[]
        for cohort in sorted({r['cohort'] for r in meta}):
            ix=np.array([i for i,r in enumerate(meta) if r['cohort']==cohort])
            reports.append({'cohort':cohort,'anchors':len(ix),'mean_normalized_Gram':normalized[ix].mean(0).tolist(),
                'mean_native_FP32_analytic_relative_MSE':float(errors[ix,1].mean()),
                'interaction_term_squared_amplitude':float(normalized[ix,3,3].mean()),
                'signed_cross_total':float((normalized[ix].sum((1,2))-np.trace(normalized[ix],axis1=1,axis2=2)).mean())})
        ledgers.append({'block':b,'path':str(path.relative_to(BASE)),'reports':reports,'parameter_key':f'model.layers.{b}.mlp.down_proj.weight',
            'arithmetic':'Mathematical SiLU of stored BF16 gate interpreted in FP64; complete products/writes in FP32, Gram in FP64. Native BF16 discrepancy retained explicitly.',
            'max_finite_expansion_relative_MSE':float(errors[:,0].max()),'mean_native_numeric_relative_MSE':float(errors[:,1].mean()),
            'scope':'Four nonorthogonal terms with all16cross terms. Diagonal normalized values are NOT percentages of explained language.'})
        del g,u,phi,weight,grams,normalized,errors
        print('LAW_ATLAS_JOINT',b,'elapsed',round(time.monotonic()-start,1),flush=True)
    compressed(out/'anchor_catalog.json.gz',meta)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'anchors':len(meta),'tokens':sum(len(r['prompt_ids']) for r in rows),
        'main_source_groups':len({r['source_group'] for r in rows}),'layers':37,'coordinates':2560,'units':9728,
        'profiles':profiles,'all_observed_gold_relation_counts':dict(counts),'coordinate_covariances':covariance,'joint_ledgers':ledgers,
        'forecast_boundary':'This atlas is observation. Target gate and future-block states never enter an early predictor. Registered prediction experiments and controlled training are separate linked evidence.',
        'limits':['Condition profiles overlap and are not randomized treatments.','Relation co-occurrence is not necessarily a performed logical/semantic composition.',
            'Displayed covariance is an empirical association, not an inferred parameter edge.','Native coordinate ordering is preserved; index proximity is not semantic geometry.',
            'All token fields processed, but only selected full all-layer fixtures persist; every H12 source and final-MLP input does persist.'],
        'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('full_coordinate_relation_joint_atlas',result['seconds'])
    print('LAW_ATLAS_COMPLETE',len(profiles),len(covariance),flush=True)


if __name__=='__main__':main()
