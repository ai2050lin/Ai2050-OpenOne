"""Post-discovery robustness analysis of existing full-coordinate fields, with train-only fitting."""
from itertools import combinations
from rdc_mechanism_common import *
from rdc_feature_extractors import fit_predict

OUT=CAMPAIGN/'a_native'
def main():
    rows=[];parts=[]
    for run in ('s1','s2pilot'):
        rr=read(PREVIOUS/run/'material.json');rows.extend([dict(r,origin_run=run) for r in rr])
        with np.load(PREVIOUS/run/'features/all_samples.npz') as z:
            parts.append({k:z[k] for k in z.files if k.startswith(('H0_','H12_','H24_','H36_')) or k=='lexical'})
    f={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tr=[i for i,r in enumerate(rows[:512]) if r['unit']<4];va=[i for i,r in enumerate(rows[:512]) if 4<=r['unit']<6];te=list(range(512,1024))
    y=np.eye(8)[[r['family_index'] for r in rows]]
    immutable(OUT/'reader_protocol.json',{'source_sha':sha(Path(__file__)),'train':tr,'validation':va,'evaluation':te,
        'status':'post-discovery existing-data robustness, not another blind confirmation',
        'no_coordinate_selection':True,'algorithms':'A1 all seven U/V/C subsets, norm-only, direction-only, norm+lexical; A2 U-only',
        'raw_sources':{run:sha(PREVIOUS/run/'protocol.json') for run in ('s1','s2pilot')}})
    immutable(OUT/'material.json',rows);results=[];books=[]
    announce('a_native',state='reader_controls',completed=0,total=44)
    def fit(layer,name,blocks,algo='A1_linear'):
        score,pred,p=fit_predict(blocks,y,tr,va,te,algo,True)
        mid=f'H{layer}__{name}__{algo}';result=dict(model_id=mid,representation=f'H{layer}',target='family',split='postdiscovery_s2',algorithm=name,**score)
        result['by_language']={lang:dict(n=sum(rows[i]['language']==lang for i in te),accuracy=float(np.mean(np.argmax(pred[[rows[i]['language']==lang for i in te]],1)==np.argmax(y[te][[rows[i]['language']==lang for i in te]],1)))) for lang in ('en','zh')}
        results.append(result);npz(OUT/f'predictions/{mid}.npz',prediction=pred,target=y[te],test_indices=np.asarray(te))
        npz(OUT/f'models/{mid}.npz',**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
    for l in (0,12,24,36):
        b=[f[f'H{l}_{key}'].astype(np.float64) for key in ('u','v','c')]
        for size in (1,2,3):
            for ids in combinations(range(3),size):fit(l,''.join('UVC'[j] for j in ids),[b[j] for j in ids])
        norms=np.stack([np.linalg.norm(x,axis=1) for x in b],axis=1)
        fit(l,'norm_only',[norms]);fit(l,'norm_lexical',[np.concatenate([norms,f['lexical']],axis=1)])
        fit(l,'unit_direction',[x/np.maximum(norms[:,i,None],1e-12) for i,x in enumerate(b)])
        fit(l,'U_quadratic',[b[0]],'A2_quadratic')
        # The old reader remains unchanged. Its centered block account is an identity, not ablation.
        with np.load(PREVIOUS/f's2pilot/coordinate_ledgers/word__family__H{l}__A1_linear.npz') as z:w=z['weights'];bias=z['bias']
        x=np.concatenate(b,axis=1);center=x[tr].mean(0);block=np.stack([(x[:,j*2560:(j+1)*2560]-center[j*2560:(j+1)*2560])@w[j*2560:(j+1)*2560] for j in range(3)],axis=1)
        base=bias+center@w;assert np.max(np.abs(base+block.sum(1)-(bias+x@w)))<1e-10
        npz(OUT/f'ledgers/H{l}_block_scores.npz',center=center,bias=base,block_scores=block,raw_weight=w,
            sample_ids=np.asarray([r['sample_id'] for r in rows]))
        books.append({'layer':l,'mean_abs_centered_block_score':np.abs(block[te]).mean((0,2)).tolist()})
        events('a_native','reader_layer_complete',layer=l,count=len(results));print('READER',l,len(results),flush=True)
    save(OUT/'reader_result.json',{'timestamp':stamp(),'results':results,'block_ledgers':books,
        'limits':['S2 is already known: these are post-discovery diagnostics with S1-only fitting.',
          'Whole-span means preserve coordinates but merge tokens; per-token native fields are independently available.',
          'Different subset-reader predictions do not constitute native causal ablations.'],
        'source_sha':sha(Path(__file__))})
    announce('a_native',state='reader_controls_complete_native_pending',completed=44,total=44)
    print('READER_DONE',len(results),flush=True)

if __name__=='__main__':main()
