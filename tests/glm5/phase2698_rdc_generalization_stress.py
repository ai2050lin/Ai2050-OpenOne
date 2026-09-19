"""Informative same-goal continuation: family/language transfer and numerical decision audit."""
import argparse
from rdc_mechanism_common import *
from rdc_feature_extractors import fit_predict,metrics
OUT=CAMPAIGN/'b_relations';DEST=CAMPAIGN/'d_generalization'

def protocol():
    immutable(DEST/'protocol.json',{'source_sha':sha(Path(__file__)),'phase':2698,'source':sha(OUT/'analysis_protocol.json'),
        'status':'Post-discovery stress on existing512 frozen cases, not fresh independent confirmation.',
        'family_transfer':'8 folds: exclude entire test family from training/validation; other families unit0..3train,4..5val; all64 cases excluded family test.',
        'language_transfer':'train source language unit0..3; validate source4..5; test other language6..7. Two directions.',
        'representations':['H24_UVC','H36_UVC','native_L23'],'targets':['positive_support','requested_answer'],
        'algorithms':['A1_linear','A2_quadratic'],'normalization':'train-only, ridge validation only, no frozen old model overwritten',
        'numerical_decision_audit':'All512 prefill argmax vs natural generated firstID; pair ties explicitly recorded; compare heldout model behavior to heldout reader, not different denominators.',
        'resource':'No new model load or capture; at most120 fits, two BLAS threads; <2GiB extra artifacts expected; no automatic infinite loop.'})

def main():
    protocol();rows=read(OUT/'material.json')
    with np.load(OUT/'features/all_samples.npz') as z:f={k:z[k] for k in z.files}
    representations={f'H{l}_UVC':[f[f'H{l}_{b}'] for b in ('u','v','c')] for l in (24,36)}
    representations['native_L23']=[f[f'L23_{k}'] for k in ('qnorm','knorm','v','a')]
    y={'positive_support':np.eye(2)[[int(r['fact_truth']) for r in rows]],'requested_answer':np.eye(2)[[int(r['expected_yes']) for r in rows]]}
    splits=[]
    for family in sorted({r['family'] for r in rows}):
        tr=[i for i,r in enumerate(rows) if r['family']!=family and r['unit']<4];va=[i for i,r in enumerate(rows) if r['family']!=family and 4<=r['unit']<6];te=[i for i,r in enumerate(rows) if r['family']==family]
        splits.append((f'family_{family}',tr,va,te))
    for language in ('en','zh'):
        tr=[i for i,r in enumerate(rows) if r['language']==language and r['unit']<4];va=[i for i,r in enumerate(rows) if r['language']==language and 4<=r['unit']<6];te=[i for i,r in enumerate(rows) if r['language']!=language and r['unit']>=6]
        splits.append((f'language_from_{language}',tr,va,te))
    results=[]
    for split,tr,va,te in splits:
        for name,blocks in representations.items():
            for target,label in y.items():
                for algo in ('A1_linear','A2_quadratic'):
                    score,pred,param=fit_predict(blocks,label,tr,va,te,algo,True)
                    mid=f'{split}__{name}__{target}__{algo}';results.append(dict(model_id=mid,split=split,representation=name,target=target,algorithm=algo,**score))
                    npz(DEST/f'predictions/{mid}.npz',prediction=pred,target=label[te],test_indices=np.array(te))
        print('TRANSFER',split,len(results),flush=True)
    behavior=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in rows]
    mismatch=[];ties=[]
    for r,b in zip(rows,behavior):
        if b['first_argmax']!=b['generated_ids'][0]:mismatch.append(dict(sample_id=r['sample_id'],prefill_argmax=b['first_argmax'],generated_first=b['generated_ids'][0],pair_logprob=b['yes_no_logprob']))
        if b['yes_no_logprob'][0]==b['yes_no_logprob'][1]:ties.append(dict(sample_id=r['sample_id'],behavior=b))
    scopes={}
    for split in ('train','validation','test'):
        ids=[i for i,r in enumerate(rows) if r['word_split']==split]
        scopes[split]={'n':len(ids),'natural_format_correct':sum(behavior[i]['correct'] for i in ids),'pair_correct_tie_excluded':sum((behavior[i]['yes_no_logprob'][0]>behavior[i]['yes_no_logprob'][1])==rows[i]['expected_yes'] for i in ids if behavior[i]['yes_no_logprob'][0]!=behavior[i]['yes_no_logprob'][1]),'ties':sum(behavior[i]['yes_no_logprob'][0]==behavior[i]['yes_no_logprob'][1] for i in ids)}
    summary=[]
    for name in representations:
        for target in y:
            for algo in ('A1_linear','A2_quadratic'):
                rr=[r for r in results if r['split'].startswith('family_') and (r['representation'],r['target'],r['algorithm'])==(name,target,algo)]
                summary.append(dict(representation=name,target=target,algorithm=algo,n=sum(r['n'] for r in rr),correct=int(round(sum(r['accuracy']*r['n'] for r in rr))),mse=sum(r['mse']*r['n'] for r in rr)/sum(r['n'] for r in rr)))
    result={'timestamp':stamp(),'results':results,'family_transfer_summary':summary,'behavior_same_denominator':scopes,
        'prefill_vs_generated_mismatches':mismatch,'pair_ties':ties,'limits':read(DEST/'protocol.json')['status']}
    save(DEST/'result.json',result);save(OUT/'extension_result.json',result)
    announce('d_generalization',state='complete',completed=120,total=120);print('STRESS_DONE',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();protocol() if a.prepare else main()
