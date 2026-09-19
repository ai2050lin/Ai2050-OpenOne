"""Prospective predictions by pre-existing S1 coefficients; no fitting on S2 pilot."""
from rdc_feature_common import *
from rdc_feature_extractors import metrics

OUT=CAMPAIGN/'s2pilot';OLD=CAMPAIGN/'s1'

def main():
    rows=read(OUT/'material.json');contract=read(OUT/'protocol.json')
    assert len(list((OUT/'commits').glob('*.json')))==512
    for mid,digest in contract['frozen_models'].items():assert sha(OLD/f'models/{mid}.npz')==digest
    immutable(OUT/'readout_protocol.json',{'timestamp_policy':'algorithm frozen by S2 capture protocol before new forward',
        'source_sha':sha(Path(__file__)),'capture_protocol_sha':sha(OUT/'protocol.json'),
        'algorithms':['A0_mean','A0_distance','A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'],
        'fit_new_parameters':False,'coefficient_ledger':'Full7680x8 A1 readout matrix and bias in raw coordinate units; extractor weights not checkpoint weights'})
    dest=OUT/'features/all_samples.npz'
    if dest.exists():
        with np.load(dest) as z:features={k:z[k] for k in z.files}
    else:
        lists={f'H{l}_{b}':[] for l in (0,12,24,36) for b in ('u','v','c')};lists['lexical']=[]
        for i,r in enumerate(rows):
            c=read(OUT/f'commits/{r["sample_id"]}.json');assert c['protocol_sha']==sha(OUT/'protocol.json')
            for rel,d in c['files'].items():assert sha(OUT/rel)==d
            with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:h=z['h']
            for l in (0,12,24,36):
                layer=unbits(h[l])
                for b in ('u','v'):lists[f'H{l}_{b}'].append(layer[r['spans'][b]['positions']].mean(0))
                lists[f'H{l}_c'].append(layer[r['context_positions']].mean(0) if l==0 else layer[-1])
            ids=[r['prompt_ids'][p] for p in r['spans']['u']['positions']]
            lists['lexical'].append([len(r['u']),len(r['v']),len(ids),len(r['spans']['v']['positions']),len(r['prompt_ids']),
                float(r['language']=='zh'),float(r['input_order']=='AB'),float(r['negative_query']),float(np.mean(ids))/151936,float(np.std(ids))/151936])
            if i%128==0:print('S2_FEATURES',i,512,flush=True)
        features={k:np.stack(v).astype(np.float32) for k,v in lists.items()};npz(dest,**features)
        save(OUT/'features/index.json',{'sample_ids':[r['sample_id'] for r in rows],'source_sha':sha(Path(__file__)),
            'fields':{k:list(v.shape) for k,v in features.items()}})
    oldrows=read(OLD/'material.json');y=np.eye(8)[[r['family_index'] for r in rows]]
    oldy=np.eye(8)[[r['family_index'] for r in oldrows]];results=[];coefficient_checks=[]
    def record(split,rep,algo,pred,mid):
        r=dict(model_id=mid,split=f'{split}_frozen',target='family',representation=rep,algorithm=algo,**metrics(y,pred,True))
        for attr in ('language','family'):
            r['by_'+attr]={}
            for label in sorted({row[attr] for row in rows}):
                mask=np.array([row[attr]==label for row in rows]);r['by_'+attr][label]=metrics(y[mask],pred[mask],True)
        results.append(r);npz(OUT/f'predictions/{mid}.npz',prediction=pred,target=y,test_indices=np.arange(512))
        return r
    for split in ('word','joint'):
        with np.load(OLD/f'models/{split}__family__lexical__A1_linear.npz') as z:
            raw=features['lexical'].astype(np.float64)/z['raw_scale_vector'];pred=(1+raw@z['z_train'].T)@z['alpha']
            record(split,'lexical','A1_linear',pred,f'{split}__family__lexical__A1_linear')
        for l in (0,12,24,36):
            rep=f'H{l}';x=np.concatenate([features[f'{rep}_{b}'] for b in ('u','v','c')],axis=1).astype(np.float64)
            with np.load(OLD/f'models/{split}__family__{rep}__A1_linear.npz') as z:
                ztrain=z['z_train'];train=z['train'];scale=z['raw_scale_vector'];xn=x/scale
            dot=xn@ztrain.T;dist=(xn*xn).sum(1)[:,None]+(ztrain*ztrain).sum(1)[None]-2*dot
            record(split,rep,'A0_mean',np.repeat(oldy[train].mean(0)[None],512,axis=0),f'{split}__family__{rep}__A0_mean')
            record(split,rep,'A0_distance',oldy[train[np.argmin(dist,axis=1)]],f'{split}__family__{rep}__A0_distance')
            blockdots=[3*xn[:,b*2560:(b+1)*2560]@ztrain[:,b*2560:(b+1)*2560].T for b in range(3)]
            kernels={'A1_linear':1+dot,'A2_quadratic':(1+dot)**2,'A2_cubic':(1+dot)**3,
                'A3_ordered_pair':1+dot+blockdots[0]*blockdots[1],
                'A4_conditional':1+dot+blockdots[0]*blockdots[1]+blockdots[0]*blockdots[1]*blockdots[2]}
            for algo,k in kernels.items():
                mid=f'{split}__family__{rep}__{algo}'
                with np.load(OLD/f'models/{mid}.npz') as z:
                    assert np.array_equal(ztrain,z['z_train']) and np.array_equal(scale,z['raw_scale_vector'])
                    pred=k@z['alpha']
                    if algo=='A1_linear':
                        w=(ztrain.T@z['alpha'])/scale[:,None];bias=z['alpha'].sum(0)
                        error=float(np.max(np.abs(x@w+bias-pred)));assert error<1e-8
                        npz(OUT/f'coordinate_ledgers/{mid}.npz',weights=w,bias=bias)
                        coefficient_checks.append(dict(model_id=mid,shape=list(w.shape),reconstruction_max_error=error))
                record(split,rep,algo,pred,mid)
            event('s2pilot','frozen_readout_group',split=split,representation=rep)
            print('S2_READOUT',split,rep,flush=True)
    passed=[]
    for r in results:
        if r['representation']=='lexical' or r['algorithm'].startswith('A0'):continue
        baseline=min(t['mse'] for t in results if t['split']==r['split'] and ((t['representation']==r['representation'] and t['algorithm'].startswith('A0')) or t['representation']=='lexical'))
        r['gain_over_fixed_A0']=(baseline-r['mse'])/max(baseline,1e-12)
        if r['gain_over_fixed_A0']>=.1:passed.append({k:r[k] for k in ('model_id','gain_over_fixed_A0','mse','accuracy')})
    behavior={}
    records=[dict(r,**read(OUT/f'behavior/{r["sample_id"]}.json')) for r in rows]
    for lang in ('en','zh'):
        for family in [None]+[a['family'] for a in rows[::64]]:
            selected=[r for r in records if r['language']==lang and (family is None or r['family']==family)]
            behavior[lang+'/'+(family or 'all')]={'n':len(selected),'correct':sum(r['correct'] for r in selected),'eos':sum(r['eos'] for r in selected),'unparsed':sum(not r['parsed'] for r in selected)}
    save(OUT/'result.json',{'phase':2694,'timestamp':stamp(),'status':'prospective_science_complete_audit_pending','cases':512,
        'results':results,'accepted_directions':passed,'behavior':behavior,'full_coordinate_linear_ledgers':coefficient_checks,
        'fit_new_parameters':False,'language_mechanism_closed':False,'limits':contract['limits'],
        'next':'Retain confirmed readouts and map native-coordinate contributions. Do not claim high-order gears or expand every relation solely from class readability.'})
    status('s2pilot',state='prospective_science_complete_audit_pending',completed=512,total=512,source_mode='recorded_model_samples')
    event('s2pilot','prospective_science_complete',comparisons=len(results),accepted=len(passed))
    print('S2_DONE',len(results),len(passed),flush=True)

if __name__=='__main__':main()
