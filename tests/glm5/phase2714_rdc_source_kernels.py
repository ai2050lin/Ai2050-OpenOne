"""All-source absolute/relative prefix kernels, common training selection, fresh frozen evaluation."""
from phase2714_rdc_full_source_history import *


def dataset(fresh=False):
    material=read(OUT/'fresh_material.json') if fresh else main_rows();sources=[];rows=[];target=[];current=[];means=[]
    for i,r in enumerate(material):
        h=source_field(r,fresh);sources.append(h)
        with np.load(OUT/f'fresh/fields/{r["sample_id"]}.npz' if fresh else CAMPAIGN/f'qwen4/fields/{r["sample_id"]}.npz') as z:
            y=unbits(z['h36']) if fresh else unbits(z['h'][36,[0,3]])
        for k,p in enumerate(r['anchors']):
            rows.append({q:r[q] for q in ('sample_id','source_group','language','genre','split')}|{'anchor':k,'position':p,'source_index':i,'prefix_length':p+1})
            current.append(h[p]);means.append(h[:p+1].mean(0,dtype=np.float64));target.append(y[k])
    return sources,rows,np.stack(current).astype(np.float64),np.stack(means),np.stack(target).astype(np.float64)


def scales_for(sources,rows,current,means,tr):
    energy=np.array([np.mean(np.sum(sources[r['source_index']][:r['position']+1].astype(float)**2,axis=1)) for r in rows])
    return {'current_energy':float(np.mean(np.sum(current[tr]**2,1))),'mean_energy':float(np.mean(np.sum(means[tr]**2,1))),
      'source_vector_energy':float(energy[tr].mean()),'source_full_coordinates':2560,'normalization':'Training-only average squared vector norms; ordered-history features divided by sqrt(each known prefix length).'}


def ordered(sources,rows,other_sources,other_rows,kind,scale,symmetric=False):
    n,m=len(rows),len(other_rows);k=np.zeros((n,m),np.float64)
    nl=np.array([r['prefix_length'] for r in rows]);ml=np.array([r['prefix_length'] for r in other_rows])
    left=np.zeros((n,2560),np.float64);right=np.zeros((m,2560),np.float64)
    for slot in range(int(min(nl.max(),ml.max()))):
        left.fill(0)
        for i,r in enumerate(rows):
            if slot<r['prefix_length']:
                ix=slot if kind=='absolute_history' else r['position']-slot
                left[i]=sources[r['source_index']][ix]/np.sqrt(r['prefix_length']*scale)
        if symmetric:right=left
        else:
            right.fill(0)
            for i,r in enumerate(other_rows):
                if slot<r['prefix_length']:
                    ix=slot if kind=='absolute_history' else r['position']-slot
                    right[i]=other_sources[r['source_index']][ix]/np.sqrt(r['prefix_length']*scale)
        k+=left@right.T
        if slot%16==15:print('FULL_SOURCE_GRAM',kind,slot+1,n,m,flush=True)
    return k


def kernel_set(sources,rows,current,means,scale,other_sources=None,other_rows=None,other_current=None,other_means=None):
    symmetric=other_sources is None
    if symmetric:other_sources=sources;other_rows=rows;other_current=current;other_means=means
    kc=current@other_current.T/scale['current_energy'];km=means@other_means.T/scale['mean_energy']
    out={'current':1+kc,'mean_history':1+(kc+km)/2}
    for name in ('absolute_history','relative_history'):
        kh=ordered(sources,rows,other_sources,other_rows,name,scale['source_vector_energy'],symmetric)
        out[name]=1+(kc+kh)/2
    return out


def main(fresh=False):
    started=time.monotonic();sources,rows,current,means,target=dataset(False);tr,va,te=splits(rows)
    assert (len(tr),len(va),len(te))==(640,192,192)
    if not fresh:
        assert len(list((OUT/'main/commits').glob('*.json')))==496
        scale=scales_for(sources,rows,current,means,tr);save(OUT/'scales.json',scale);save(OUT/'main_rows.json',rows)
        kernels=kernel_set(sources,rows,current,means,scale);reports=[];testrows=[rows[i] for i in te]
        for name,k in kernels.items():
            pred,val,meta=fit(k,tr,va,te,target,[(0,2560)],OUT/f'models/{name}.npz')
            report,arr=errors(target[te],pred,target[tr],testrows)
            reports.append({'rule':name,**meta,**report})
            npz(OUT/f'predictions/test_{name}.npz',prediction=pred,test=te,**arr)
            print('SOURCE_KERNEL_TEST',name,report['mse'],meta['normalized_validation_mse'],flush=True)
        npz(OUT/'input_grams.npz',**{k:v.astype(np.float32) for k,v in kernels.items()},train=tr,validation=va,test=te)
        best=min(reports,key=lambda r:r['normalized_validation_mse'])['rule']
        result={'phase':2714,'timestamp':stamp(),'source_units':512,'anchors':1024,'reports':reports,'selected_before_fresh':best,
          'information':'All coordinate inputs at every source position. Relative/absolute matching is an explicit typed-position similarity rule, not proof of native attention or semantic binding.',
          'scope':'Held-out main sources, full2560 target, H36-only validation selection; not identical selection objective to2712 joint three-layer rules.'}
        save(OUT/'result.json',result)
        files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').glob('*.npz')}
        for rel in ('scales.json','main_rows.json','result.json','protocol.json','fresh_material.json'):files[rel]=sha(OUT/rel)
        inputs={}
        for r in main_rows():
            p=CAMPAIGN/f'qwen4/full_panels/{r["sample_id"]}.npz' if r['full_panel'] else OUT/f'main/fields/{r["sample_id"]}.npz'
            inputs[str(p.relative_to(CAMPAIGN))]=sha(p)
            p=CAMPAIGN/f'qwen4/fields/{r["sample_id"]}.npz';inputs[str(p.relative_to(CAMPAIGN))]=sha(p)
        immutable(OUT/'frozen.json',{'timestamp':stamp(),'selected_by_validation':best,'files':files,'original_inputs':inputs,
          'fit_source_sha':sha(Path(__file__)),'capture_source_sha':sha(ROOT/'tests/glm5/phase2714_rdc_full_source_history.py'),
          'fresh_retraining_allowed':False,'fresh_capture_started':False})
        status('full_source_history',state='rules_frozen_before_fresh',selected=best)
    else:
        frozen=read(OUT/'frozen.json')
        for rel,digest in frozen['files'].items():assert sha(OUT/rel)==digest
        ns,nr,nc,nm,ny=dataset(True);scale=read(OUT/'scales.json')
        trainrows=[rows[i] for i in tr];kernels=kernel_set(ns,nr,nc,nm,scale,sources,trainrows,current[tr],means[tr])
        reports=[]
        for name,k in kernels.items():
            with np.load(OUT/f'models/{name}.npz') as z:pred=(k@z['alpha'])*z['target_scales']+z['means']
            report,arr=errors(ny,pred,target[tr],nr);reports.append({'rule':name,**report})
            npz(OUT/f'predictions/fresh_{name}.npz',prediction=pred.astype(np.float32),**arr)
            print('SOURCE_KERNEL_FRESH',name,report['mse'],flush=True)
        baselines={}
        for name,pred in [('train_mean',np.broadcast_to(target[tr].mean(0),ny.shape)),('copy_H12',nc)]:
            report,arr=errors(ny,pred,target[tr],nr);baselines[name]=report
        byname={r['rule']:r for r in reports};comparisons=[]
        for a,b in [('relative_history','absolute_history'),('relative_history','mean_history'),('absolute_history','mean_history'),(frozen['selected_by_validation'],'current')]:
            aa=byname[a]['by_source_group'];bb=byname[b]['by_source_group'];ids=sorted(aa);d=np.array([aa[k]['mse']-bb[k]['mse'] for k in ids]);rng=np.random.default_rng(2714)
            draws=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
            comparisons.append({'a':a,'b':b,'source_units':len(ids),'mean_MSE_a_minus_b':float(d.mean()),'CI95':np.quantile(draws,[.025,.975]).tolist(),'fraction_a_better':float(np.mean(d<0))})
        save(OUT/'fresh_rows.json',nr)
        save(OUT/'fresh_result.json',{'timestamp':stamp(),'phase':2714,'source_units':64,'anchors':128,'reports':reports,'baselines':baselines,
          'selected_before_capture':frozen['selected_by_validation'],'no_fresh_fit':True,'frozen_manifest_sha':sha(OUT/'frozen.json'),'paired_comparisons':comparisons,
          'limits':['64 new natural source units, not every language relation family.','Grouped inference does not verify pretraining absence.','Position kernels are candidate observational prediction rules, not proven native causal algorithms.']})
        status('full_source_history',state='fresh_frozen_evaluation_complete',source_units=64)
    assert time.monotonic()-started<1800;guard(12*1024**2);print('SOURCE_KERNEL_COMPLETE',fresh,usage(),CEILING-usage(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');a=p.parse_args();main(a.fresh)
