"""Frozen grouped relation benchmark; full-coordinate kernels, no target-state inputs."""
import argparse
from rdc_mechanism_common import *
from rdc_feature_extractors import fit_predict,metrics,ALGORITHMS
OUT=CAMPAIGN/'b_relations'

def protocol():
    if (OUT/'analysis_protocol.json').exists():
        old=read(OUT/'analysis_protocol.json')
        if old['source_sha']!=sha(Path(__file__)):
            immutable(OUT/'analysis_implementation_v2.json',{'old_source_sha':old['source_sha'],'new_source_sha':sha(Path(__file__)),
                'reason':'Convert numpy integer count to JSON int; reuse already extracted identical features. No material, split, metric, or algorithm changes.',
                'incident':'All129 comparisons computed; result.json serialization failed before publishing. Frozen raw captures and predictions preserved.'})
        return
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),
        'targets':['positive_support','requested_answer','actual_margin'],
        'layers':[0,12,24,36],'algorithms':list(ALGORITHMS),
        'controls':'C-only, U/V-only, norm-only, direction-only; learned positive-support then observed query polarity XOR; native qnorm/knorm/v/a at L11/23/35',
        'splits':'base unit0..3 train256;4..5 validation128;6..7 held-out128; fact/query/language variants grouped',
        'selection':'ridge chosen only on validation MSE; fixed algorithms, no best-test selection claim',
        'contrast':'four-cell support x query interaction in every coordinate, per base/language; descriptive not an intervention',
        'limits':['Entire family remains in training; held-out base cases, not held-out task families.',
            'Templates shared; differing token lengths/position are not fully orthogonalized.',
            'Actual margin is observed model Yes-minus-No log probability, not ground-truth competence.']})

def main():
    protocol();rows=read(OUT/'material.json');assert len(list((OUT/'commits').glob('*.json')))==512
    features={};behavior=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in rows]
    cached=(OUT/'features/all_samples.npz').exists()
    for i,r in enumerate([] if cached else rows):
        with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
            for l in (0,12,24,36):
                for b in ('u','v','c'):
                    ix=r['spans'][b]['positions'] if b!='c' else [len(r['prompt_ids'])-1]
                    features.setdefault(f'H{l}_{b}',[]).append(unbits(z['h'][l,ix]).mean(0))
            positions=z['native_positions'].tolist()
            for l in (11,23,35):
                ix=positions.index(len(r['prompt_ids'])-1)
                for k in ('qnorm','knorm','v','a'):
                    features.setdefault(f'L{l}_{k}',[]).append(unbits(z[f'L{l}_{k}'][ix]).reshape(-1))
    if cached:
        with np.load(OUT/'features/all_samples.npz') as z:f={k:z[k] for k in z.files}
    else:
        f={k:np.stack(v) for k,v in features.items()};npz(OUT/'features/all_samples.npz',**f)
    tr,va,te=[[i for i,r in enumerate(rows) if r['word_split']==s] for s in ('train','validation','test')]
    ys={'positive_support':np.eye(2)[[int(r['fact_truth']) for r in rows]],
        'requested_answer':np.eye(2)[[int(r['expected_yes']) for r in rows]],
        'actual_margin':np.array([[b['yes_no_logprob'][0]-b['yes_no_logprob'][1]] for b in behavior])}
    results=[];index=[]
    def fit(rep,target,algo,blocks,label=None):
        y=ys[target];classification=target!='actual_margin'
        score,pred,p=fit_predict(blocks,y,tr,va,te,algo,classification)
        name=label or algo;mid=f'{rep}__{target}__{name}'
        result=dict(model_id=mid,split='base_heldout',representation=rep,target=target,algorithm=name,**score)
        result['by_family']={fam:metrics(y[te][mask],pred[mask],classification) for fam in sorted({r['family'] for r in rows}) if (mask:=np.array([rows[i]['family']==fam for i in te])).any()}
        results.append(result);npz(OUT/f'predictions/{mid}.npz',prediction=pred,target=y[te],test_indices=np.array(te))
        npz(OUT/f'models/{mid}.npz',**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
        save(OUT/f'models/{mid}.json',{'algorithm':algo,'scales':p['scales'],'ridge':p.get('ridge'),'inputs':rep,'target':target})
        index.append({'model_id':mid,'algorithm':algo,'path':f'models/{mid}.npz'})
        if target=='positive_support' and algo=='A1_linear' and label is None:
            # Query polarity is observed input, never the support/answer label at prediction time.
            polarity=np.array([rows[i]['negative_query'] for i in te]);choice=np.argmax(pred,1).astype(bool)^polarity
            cp=np.eye(2)[choice.astype(int)]
            results.append(dict(split='base_heldout',representation=rep,target='requested_answer',algorithm='learned_support_then_query_XOR',**metrics(ys['requested_answer'][te],cp,True)))
        return pred
    for l in (0,12,24,36):
        blocks=[f[f'H{l}_{b}'] for b in ('u','v','c')]
        for target in ys:
            for algo in ALGORITHMS:fit(f'H{l}',target,algo,blocks)
        norms=np.stack([np.linalg.norm(x,axis=1) for x in blocks],axis=1)
        for target in ('positive_support','requested_answer'):
            for label,bb in [('C_only',[blocks[2]]),('UV_only',blocks[:2]),('norm_only',[norms]),('direction',[x/np.maximum(norms[:,j,None],1e-12) for j,x in enumerate(blocks)])]:
                fit(f'H{l}',target,'A1_linear',bb,label)
        print('RELATION_ANALYSIS',l,len(results),flush=True)
    for l in (11,23,35):
        blocks=[f[f'L{l}_{k}'] for k in ('qnorm','knorm','v','a')]
        for target in ys:fit(f'native_L{l}',target,'A5_native',blocks)
    contrasts={};summary=[]
    for l in (0,12,24,36):
        for b in ('u','v','c'):
            interaction=[];support=[];query=[];meta=[]
            for fam in sorted({r['family'] for r in rows}):
                for unit in range(8):
                    for lang in ('en','zh'):
                        ids={(int(r['fact_truth']),int(r['negative_query'])):i for i,r in enumerate(rows) if (r['family'],r['unit'],r['language'])==(fam,unit,lang)}
                        x=f[f'H{l}_{b}'];a,c,d,e=[x[ids[t]].astype(np.float64) for t in ((0,0),(1,0),(0,1),(1,1))]
                        support.append(c-a);query.append(d-a);interaction.append(e-c-d+a);meta.append([fam,str(unit),lang])
            key=f'H{l}_{b}';interaction=np.array(interaction);support=np.array(support);query=np.array(query)
            contrasts[key+'_interaction']=interaction;contrasts[key+'_support']=support;contrasts[key+'_query']=query
            summary.append({'representation':key,'full_coordinate_interaction_rms':float(np.sqrt(np.mean(interaction**2))),
                'support_rms':float(np.sqrt(np.mean(support**2))),'query_rms':float(np.sqrt(np.mean(query**2))),'cases':128,
                'nonzero_coordinates':int(np.count_nonzero(np.any(interaction!=0,axis=0)))})
    npz(OUT/'ledgers/condition_full_coordinates.npz',**contrasts,case_keys=np.asarray(meta))
    by={}
    for fam in sorted({r['family'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['family']==fam]
        by[fam]={'n':len(ids),'format_correct':sum(behavior[i]['correct'] for i in ids),'parsed':sum(behavior[i]['parsed'] for i in ids),
            'eos':sum(behavior[i]['eos'] for i in ids),'first_pair_correct':int(sum((ys['actual_margin'][i,0]>0)==rows[i]['expected_yes'] for i in ids))}
    save(OUT/'result.json',{'timestamp':stamp(),'results':results,'models':index,'behavior_by_family':by,'contrasts':summary,
        'case_count':512,'test_n':128,'limits':read(OUT/'analysis_protocol.json')['limits']})
    announce('b_relations',state='analysis_complete',completed=512,total=512);events('b_relations','analysis_complete',comparisons=len(results))
    print('RELATIONS_DONE',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    protocol() if a.prepare else main()
