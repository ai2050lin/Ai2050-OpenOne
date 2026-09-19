"""Frozen S1 full-coordinate feature contest, lexical controls and prospective H prediction."""
from collections import defaultdict
from rdc_feature_common import *
from rdc_feature_extractors import *

OUT=CAMPAIGN/'s1'; LAYERS=(0,12,24,36)

def partitions(rows):
    return {
        'word':([i for i,r in enumerate(rows) if r['unit']<4],[i for i,r in enumerate(rows) if 4<=r['unit']<6],[i for i,r in enumerate(rows) if r['unit']>=6]),
        'form':([i for i,r in enumerate(rows) if r['form']<2],[i for i,r in enumerate(rows) if r['form']==2],[i for i,r in enumerate(rows) if r['form']==3]),
        'joint':([i for i,r in enumerate(rows) if r['unit']<4 and r['form']<2],
                 [i for i,r in enumerate(rows) if 4<=r['unit']<6 and r['form']==2],
                 [i for i,r in enumerate(rows) if r['unit']>=6 and r['form']==3])}

def features(rows):
    dest=OUT/'features/all_samples.npz'
    if dest.exists():
        with np.load(dest) as z:return {k:z[k] for k in z.files}
    features=defaultdict(list);identities=[]
    for i,r in enumerate(rows):
        c=read(OUT/f'commits/{r["sample_id"]}.json')
        assert c['protocol_sha']==sha(OUT/'protocol.json')
        field=OUT/f'fields/{r["sample_id"]}.npz'
        assert sha(field)==c['files'][f'fields\\{r["sample_id"]}.npz'] if f'fields\\{r["sample_id"]}.npz' in c['files'] else sha(field)==c['files'][f'fields/{r["sample_id"]}.npz']
        with np.load(field,allow_pickle=False) as z:
            for layer in LAYERS:
                h=unbits(z['h'][layer])
                for block in ('u','v'):features[f'H{layer}_{block}'].append(h[r['spans'][block]['positions']].mean(0))
                features[f'H{layer}_c'].append(h[r['context_positions']].mean(0) if layer==0 else h[-1])
            for native_layer in (0,11,23,35):
                positions=z['native_positions'].tolist();parts=[]
                for b in ('u','v','c'):
                    needed=r['spans'][b]['positions'] if b in ('u','v') else [len(r['prompt_ids'])-1]
                    idx=[positions.index(t) for t in needed]
                    vecs=[unbits(z[f'L{native_layer}_{name}'][idx]).reshape(len(idx),-1).mean(0) for name in ('qnorm','knorm','v','a')]
                    features[f'native{native_layer}_{b}'].append(np.concatenate(vecs))
            # Future H target at u anchor: target used only as training labels and test truth.
            features['future_H36'].append(unbits(z['h'][36,r['spans']['u']['positions']]).mean(0))
        ids=[r['prompt_ids'][p] for p in r['spans']['u']['positions']]
        # No latent unit/family/partner IDs in this lexical-control feature vector.
        features['lexical'].append([len(r['u']),len(r['v']),len(ids),len(r['spans']['v']['positions']),len(r['prompt_ids']),
            float(r['language']=='zh'),float(r['input_order']=='AB'),float(r['negative_query']),
            float(np.mean(ids))/151936,float(np.std(ids))/151936])
        identities.append(r['sample_id'])
        if i%64==0:print('FEATURES',i,512,flush=True)
    arrays={k:np.stack(v).astype(np.float32) for k,v in features.items()}
    npz(dest,**arrays);save(OUT/'features/index.json',{'sample_ids':identities,'fields':{k:list(v.shape) for k,v in arrays.items()},
        'capture_protocol_sha':sha(OUT/'protocol.json'),'source_code_sha':sha(Path(__file__)),
        'pooling':'u/v average all native coordinates over actual term span; H0 c only current question tokens, deeper c current last token. Native A5 spans pool qnorm/knorm/v/a; no target labels.'})
    return arrays

def main():
    rows=read(OUT/'material.json');assert len(rows)==512 and len(list((OUT/'commits').glob('*.json')))==512
    contract={'version':1,'capture_protocol_sha':sha(OUT/'protocol.json'),'source_sha':sha(Path(__file__)),
      'math_sha':sha(ROOT/'tests/glm5/rdc_feature_extractors.py'),'representations':['H0','H12','H24','H36'],
      'algorithms':list(ALGORITHMS),'native_A5':'qnorm,knorm,V,a at layers0,11,23,35 pooled at declared spans; different pretrained computation, not pure algorithm-only advantage',
      'targets':['family','external_answer'],'splits':['word','form','joint'],'random_word_label_seeds':[31,47,59],
      'lambdas':list(LAMBDAS),'selection':'validation only; test cannot choose scales/ridge/models',
      'expansion_gate':'At least10% lower testMSE than best A0 lexical/mean/distance on word AND joint for the same input/target, then prospective new-material confirmation; not a mechanism claim.',
      'future':'H12 full u/v/c predicts H36 full u target; no H36 input or future token',
      'limits':['Word-entry holdout is not token-ID disjoint: ordinary function words recur as grammatical scaffolding.',
                'Form3 is an unseen paraphrase but shares negative-query polarity with validation form2.',
                'Contextual last-token H includes content; conditional kernel is not a unique causal factorization.',
                'Eight classes are task-defined and overlap in unrestricted language.']}
    immutable(OUT/'benchmark_protocol.json',contract)
    f=features(rows);splits=partitions(rows); results=[];models=[];predictions=[]
    ys={'family':np.eye(8)[[r['family_index'] for r in rows]],'external_answer':np.eye(2)[[int(r['expected_yes']) for r in rows]]}
    def run_fit(blocks,y,split,target,rep,algo):
        tr,va,te=splits[split]
        score,pred,params=fit_predict(blocks,y,tr,va,te,algo,classification=True)
        model_id=f'{split}__{target}__{rep}__{algo}'
        result=dict(model_id=model_id,split=split,target=target,representation=rep,algorithm=algo,**score)
        results.append(result)
        npz(OUT/f'predictions/{model_id}.npz',prediction=pred,target=y[te],test_indices=np.asarray(te))
        if algo in ('A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional','A5_native'):
            path=f'models/{model_id}.npz'
            npz(OUT/path,**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
            models.append(dict(model_id=model_id,algorithm=algo,path=path,split=split,target=target,representation=rep))
        for lang in ('en','zh'):
            mask=np.asarray([rows[i]['language']==lang for i in te])
            result.setdefault('by_language',{})[lang]=metrics(y[te][mask],pred[mask],True)
        return result
    status('s1',state='analyzing',completed=512,total=512,source_mode='recorded_model_samples')
    for split in splits:
        for target,y in ys.items():
            run_fit([f['lexical']],y,split,target,'lexical','A1_linear')
            for layer in LAYERS:
                rep=f'H{layer}';blocks=[f[f'{rep}_{b}'] for b in ('u','v','c')]
                for algo in ALGORITHMS:run_fit(blocks,y,split,target,rep,algo)
                event('s1','extractor_group_complete',split=split,target=target,representation=rep)
                print('BENCHMARK',split,target,rep,flush=True)
            for native_layer in (0,11,23,35):
                run_fit([f[f'native{native_layer}_{b}'] for b in ('u','v','c')],y,split,target,f'native{native_layer}','A5_native')
            save(OUT/'result.json',{'phase':2693,'status':'analysis_partial','results':results,'models':models})
    # Fixed random labels attached to lexical types; train and test types do not overlap on word/joint.
    controls=[]
    for seed in (31,47,59):
        labels=np.repeat(np.arange(8),8);np.random.default_rng(seed).shuffle(labels)
        y=np.eye(8)[[labels[r['family_index']*8+r['unit']] for r in rows]]
        for split in ('word','form','joint'):
            for layer in (0,36):
                for algo in ('A1_linear','A2_quadratic','A2_cubic','A4_conditional'):
                    score,_,_=fit_predict([f[f'H{layer}_{b}'] for b in ('u','v','c')],y,*splits[split],algo,classification=True)
                    controls.append(dict(seed=seed,split=split,representation=f'H{layer}',algorithm=algo,**score))
    future=[]
    for split in ('word','joint'):
        tr,va,te=splits[split];y=f['future_H36']
        for algo in ('A0_mean','A0_distance','A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'):
            score,pred,_=fit_predict([f[f'H12_{b}'] for b in ('u','v','c')],y,tr,va,te,algo)
            future.append(dict(split=split,algorithm=algo,**score))
            npz(OUT/f'predictions/{split}__future_H36__{algo}.npz',prediction=pred.astype(np.float32),target=y[te],test_indices=np.asarray(te))
        future.append(dict(split=split,algorithm='carry_H12',**metrics(y[te],f['H12_u'][te],False)))
    behavior=[]
    for r in rows:behavior.append(dict(r,**{k:v for k,v in read(OUT/f'behavior/{r["sample_id"]}.json').items() if k not in r}))
    summary={}
    for lang in ('en','zh'):
        for family in [None]+sorted({r['family'] for r in rows}):
            selected=[r for r in behavior if r['language']==lang and (family is None or r['family']==family)]
            summary[lang+'/'+(family or 'all')]={'n':len(selected),'correct':sum(r['correct'] for r in selected),'unparsed':sum(not r['parsed'] for r in selected),'eos':sum(r['eos'] for r in selected)}
    # A rule can remain a candidate in its condition; never require all eight families to pass simultaneously.
    signals=[]
    for rep in [f'H{l}' for l in LAYERS]:
        for target in ys:
            for algo in ('A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'):
                gains={}
                for split in ('word','joint'):
                    base=min(r['mse'] for r in results if r['split']==split and r['target']==target and ((r['representation']==rep and r['algorithm'].startswith('A0')) or r['representation']=='lexical'))
                    value=next(r['mse'] for r in results if (r['split'],r['target'],r['representation'],r['algorithm'])==(split,target,rep,algo))
                    gains[split]=(base-value)/max(base,1e-12)
                if min(gains.values())>=.1:signals.append(dict(representation=rep,target=target,algorithm=algo,gains=gains))
    save(OUT/'model_index.json',models)
    final={'phase':2693,'timestamp':stamp(),'status':'science_complete_client_audit_pending','cases':512,
       'results':results,'models':models,'random_word_controls':controls,'future_field_prediction':future,'behavior':summary,
       'expansion_candidates':signals,'language_mechanism_closed':False,'checks':{'512_committed':True,'no_coordinate_selection':True,
       'word_partner_blocks_disjoint':all(r['unit']//2==r['partner_unit']//2 for r in rows)},
       'limits':contract['limits'],'next':'Review per-domain held-out gains and random-label controls before authorizing a bounded prospective confirmation.'}
    save(OUT/'result.json',final)
    status('s1',state='science_complete_client_audit_pending',completed=512,total=512,source_mode='recorded_model_samples')
    event('s1','science_complete',results=len(results),expansion_candidates=len(signals))
    print('BENCHMARK_DONE',len(results),len(signals),flush=True)

if __name__=='__main__':main()
