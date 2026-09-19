"""Exploratory all-coordinate four-question relation preservation, not semantics."""
import argparse
from collections import Counter,defaultdict
from itertools import permutations
from rdc_question_common import *
import rdc_question_data as data
from phase2748_rdc_fit_analysis import bootstrap

CENTER=np.eye(4)-np.ones((4,4))/4
PERMUTATIONS=list(permutations(range(4)))


def gram(values):
    assert values.ndim==2 and len(values)==4
    centered=values-values.mean(0)
    return centered @ centered.T/values.shape[1]


def token_gram(token_lists):
    assert len(token_lists)==4 and all(token_lists)
    hist=[{k:v/len(t)for k,v in Counter(t).items()}for t in token_lists]
    raw=np.array([[sum(v*b.get(k,0.)for k,v in a.items())for b in hist]for a in hist])
    return CENTER @ raw @ CENTER


def relations(grams):
    assert grams.ndim==3 and grams.shape[1:]==(4,4)
    flat=grams.reshape(len(grams),16);norm=np.sqrt(np.sum(flat**2,axis=1))
    denominator=np.outer(norm,norm);valid=denominator>0
    traces=np.trace(grams,axis1=1,axis2=2)
    similarity=np.divide(flat @ flat.T,denominator,out=np.zeros_like(denominator),where=valid)
    # Exact mean over all24within-context question permutations: E[P K P^T]
    # is tr(K)/3 times the centered identity for four questions.
    baseline=np.divide(np.outer(traces,traces)/3,denominator,out=np.zeros_like(denominator),where=valid)
    return {'similarity':similarity,'permutation_mean':baseline,'excess':similarity-baseline,'valid':valid,
        'within_response_mean_square':traces/4}


def freeze():
    path=OUT/'relation_geometry/execution.json'
    execution={'source':snapshot(__file__),'data':snapshot(Path(__file__).with_name('rdc_question_data.py')),
        'bootstrap':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py')),
        'material_sha256':sha(OUT/'material/manifest.json')}
    if path.exists():
        old=read(path);assert old['execution']==execution;return old
    unit=read(OUT/'unit/relation_geometry_current.json')
    assert unit['all_passed']and unit['analysis']['sha256']==execution['source']['sha256']
    value={'timestamp':stamp(),'execution':execution,'unit_sha256':sha(OUT/'unit/relation_geometry_current.json'),
        'status':'New exploratory diagnostic after Q4first-prefix and nativehistory analyses; not original preregistration and not a new rule-selection input.',
        'population':'All336nonconfirmation contexts,4questions each; train/validation/diagnostic separate, both cohorts separate/equal. No confirmation access.',
        'objects':'All native residual boundaries plus separate postnorm, all gate/up/product units of declared capturedblocks; query token-frequency kernel over complete token-ID vocabulary and question-token-length kernel are external controls.',
        'algorithm':'Four-question centering within context, Gram contraction over EVERYcoordinate, Frobenius cosine between centeredGram matrices. No coordinate selection or latent projection. Exact average over all24question permutations corrects high similarity from isotropic four-item geometry.',
        'permutation_mean_formula':'tr(K_l)*tr(K_m)/(3*||K_l||F*||K_m||F). Undefined zero-energy pairs stored with explicitvalidFalse and numericplaceholder0, never interpreted as dissimilarity.',
        'primary_display_pairs':'Postnorm with H12,H24,H32,Hlast and lexical-question-frequency and question-token-length. Choices fixed by existing target boundaries, not by this output.',
        'uncertainty':'Whole-context2000paired bootstrap for similarity-minus-exactpermutationmean for fixeddisplaypairs, seeds2748012/2748013; descriptive95percent no multiplicity correction. Full pair matrices are descriptive means with validcounts, not significance maps.',
        'limits':'Only4questions yields rank<=3 Gram by design; this does not show native hiddenfield rank3. Similarity can arise from wording, length, residual carryover or other confounds. No crossmodel/crosslayer coordinate functional correspondence, semantic causality, or generation-step closure. Preserve original full fields as primary data.'}
    immutable(path,value);return value


def main(key):
    start=time.monotonic();spec=freeze();folder=Path('relation_geometry')/key;final=OUT/folder/'result.json'
    if final.exists():
        old=read(final);assert old['execution_sha256']==sha(OUT/'relation_geometry/execution.json')
        print('NATURAL_RELATION_GEOMETRY_ALREADY_COMPLETE',key,flush=True);return
    contract=effective_contract();blocks=contract['capture']['selected_MLP_blocks'][key]
    depth=read(ROOT/'models/hf'/MODELS[key]/'config.json')['num_hidden_layers']
    labels=['H'+str(i)for i in range(depth+1)]+['postnorm']+[f'block{b}_{c}'for b in blocks for c in ['gate','up','product']]+['question_token_frequency','question_token_length']
    ids=[];fields=[];all_grams=[];all_values={k:[]for k in ['similarity','permutation_mean','excess','valid','within_response_mean_square']}
    for split in ['train','validation','diagnostic']:
        rows,groups,questions=data.index(key,{split});by_group=defaultdict(list)
        for row in rows:by_group[row['group_id']].append(row)
        for gid,rr in by_group.items():
            assert len(rr)==4
            loaded=[data.field(questions[r['question_id']]['field'],['hidden_BF16','postnorm_BF16']+[f'block{b}_{c}_BF16'for b in blocks for c in ['gate','up','product']])for r in rr]
            nodes=[np.stack([q['hidden_BF16'][i]for q in loaded])for i in range(depth+1)]
            nodes+=[np.stack([q['postnorm_BF16']for q in loaded])]
            nodes+=[np.stack([q[f'block{b}_{c}_BF16']for q in loaded])for b in blocks for c in ['gate','up','product']]
            tokens=[[r['tokens']['input_ids'][p]for p in r['tokens']['question_token_positions']]for r in rr]
            kernels=np.stack([gram(node)for node in nodes]+[token_gram(tokens),gram(np.array([[len(t)]for t in tokens],float))])
            values=relations(kernels);assert len(kernels)==len(labels)
            ids.append({'group_id':gid,'cohort':rr[0]['cohort'],'split':split,'question_ids':[r['question_id']for r in rr]})
            fields.extend([questions[r['question_id']]['field']for r in rr]);all_grams.append(kernels)
            for k,v in values.items():all_values[k].append(v)
        print('NATURAL_RELATION_GEOMETRY',key,split,len(rows),round(time.monotonic()-start,1),flush=True)
    arrays={k:np.stack(v)for k,v in all_values.items()};arrays['full_coordinate_Gram_by_context']=np.stack(all_grams)
    summaries=[];pair_records=[];post=labels.index('postnorm')
    for split in ['train','validation','diagnostic']:
        for cohort in ['drop','quoref']:
            take=np.array([r['split']==split and r['cohort']==cohort for r in ids]);valid=arrays['valid'][take]
            counts=valid.sum(0);stem=split+'__'+cohort
            arrays[stem+'__valid_contexts']=counts
            for metric in ['similarity','permutation_mean','excess']:
                arrays[stem+'__'+metric+'_mean']=np.divide((arrays[metric][take]*valid).sum(0),counts,out=np.zeros(counts.shape),where=counts>0)
            summaries.append({'split':split,'cohort':cohort,'contexts':int(take.sum()),'field_prefix':stem})
        for label in ['H12','H24','H32','H'+str(depth),'question_token_frequency','question_token_length']:
            left=labels.index(label);take=np.array([r['split']==split for r in ids])&arrays['valid'][:,left,post]
            rr=[r for r,t in zip(ids,take)if t]
            # Existing paired bootstrap validates4question rows per context.
            # Repeat the single context statistic on its4actual questions:
            # the helper first averages those four, then resamples contexts.
            # This is exactly context bootstrap, not four independent draws.
            a=arrays['similarity'][take,left,post];b=arrays['permutation_mean'][take,left,post]
            bootstrap_rows=[{**r,'question_id':qid}for r in rr for qid in r['question_ids']]
            pair_records.append({'split':split,'left':label,'right':'postnorm','valid_contexts':len(rr),
                'by_cohort':{c:{'contexts':sum(r['cohort']==c for r in rr),
                    'similarity_mean':float(a[[r['cohort']==c for r in rr]].mean()),
                    'permutation_mean':float(b[[r['cohort']==c for r in rr]].mean())}for c in ['drop','quoref']},
                'paired_excess':bootstrap(bootstrap_rows,np.repeat(a,4),np.repeat(b,4),2748012)})
    ref=commit_arrays(folder,'all_context_relations',arrays)
    result={'timestamp':stamp(),'all_passed':True,'model':key,'contexts':len(ids),'questions':4*len(ids),
        'execution_sha256':sha(OUT/'relation_geometry/execution.json'),'native_result_sha256':sha(OUT/f'native/{key}/nonconfirmation/result.json'),
        'labels':labels,'field':ref,'identities':ids,'source_first_fields':fields,'summaries':summaries,'fixed_display_pairs':pair_records,
        'limits':spec['limits'],'seconds':time.monotonic()-start,'scope':'Retrospective first-prefix response relation atlas; no language prediction or causal intervention.'}
    immutable(final,result);print('NATURAL_RELATION_GEOMETRY_COMPLETE',key,len(labels),round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4']);parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:freeze()
    else:
        assert args.model;main(args.model)
