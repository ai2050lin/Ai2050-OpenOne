"""Strength relative to question variation and actual frozen query kernels.

Added after raw attention-mass observations; no fit or selection is changed.
"""
import argparse
from rdc_question_common import *
import rdc_question_data as data

METRICS = ['native_mean_square','permuted_mean_square','permutation_MSE','common_context_change_MSE',
    'within_native_mean_square','within_permuted_mean_square','within_permutation_MSE','within_native_permuted_inner_product']


def group_statistics(native, permuted):
    assert native.shape == permuted.shape and native.ndim == 2 and len(native) == 4
    delta = permuted-native; rn = native-native.mean(0); rs = permuted-permuted.mean(0)
    values = np.array([np.mean(native**2),np.mean(permuted**2),np.mean(delta**2),np.mean(delta.mean(0)**2),
        np.mean(rn**2),np.mean(rs**2),np.mean((rs-rn)**2),np.mean(rn*rs)])
    error = max(abs(values[2]-values[3]-values[6]),abs(values[6]-values[4]-values[5]+2*values[7]))
    assert np.isfinite(values).all() and error < 1e-10*max(1.,float(np.max(np.abs(values))))
    return values


def unit():
    r=np.array([[1.,2.],[2.,4.],[4.,1.],[3.,5.]])
    s=r+np.array([[.2,.1],[-.2,-.1],[.1,.2],[-.1,-.2]])
    x=group_statistics(r,s); y=group_statistics(r+1000,s+1000)
    assert y[2]/y[0] < x[2]/x[0]/10000
    np.testing.assert_allclose(x[4:],y[4:],rtol=1e-10,atol=1e-10)
    constant=group_statistics(r,r+3)
    assert constant[6] == 0 and constant[2] == constant[3] == 9
    zero=group_statistics(np.ones((4,2)),np.ones((4,2)))
    assert zero[4] == zero[6] == 0
    native=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    shifted=native[[1,0,3,2]]; v=group_statistics(native,shifted)
    assert v[4] == v[5] == .25 and v[6] == .5
    result={'timestamp':stamp(),'all_passed':True,'analysis_sha256':sha(__file__),'checks':4,
        'tests':['large_common_background_can_hide_question_relative_perturbation','common_context_shift_not_question_change',
                 'zero_question_variation_explicit_undefined_ratio','known_four_question_permutation'],
        'scope':'Synthetic all-coordinate arithmetic only.'}
    immutable(OUT/'unit'/('source_coupling_relative_'+str(time.time_ns())+'.json'),result)
    save(OUT/'unit/source_coupling_relative_current.json',result)
    print('NATURAL_SOURCE_RELATIVE_UNIT_PASS',4,flush=True)


def freeze():
    path=OUT/'source_coupling/relative_execution.json'
    revision={'source':snapshot(__file__),'data':snapshot(Path(__file__).with_name('rdc_question_data.py')),
              'source_coupling_execution_sha256':sha(OUT/'source_coupling/execution.json')}
    if path.exists():
        result=read(path);assert result['execution']==revision;return result
    result=read(OUT/'unit/source_coupling_relative_current.json');assert result['all_passed'] and result['analysis_sha256']==sha(__file__)
    value={'timestamp':stamp(),'execution':revision,'unit_sha256':sha(OUT/'unit/source_coupling_relative_current.json'),
        'status':'New diagnostic after Q4/Q14 raw source coupling results, before reading these relative statistics. No primary/source/seed/checkpoint choice changes.',
        'spaces':['raw_read','common_native_training_scale','respective_frozen_route_scales','full_query_respective_scales','frozen_query_kernel'],
        'normalization':'Use actual saved selected native_source_read and source_value_pair_shuffle operator query means/scales. These were fixed by original training before this diagnostic. No new standardization fit.',
        'kernel':'Each route normalized full [H12;read] query inner product against all768original training queries divided by full query width. Retain every query-to-training pair. This is query kernel only, not full question/context interaction kernel or a new predictor.',
        'estimands':'Four-question centering within unchanged context. Exact decomposition of total permutation energy into common-context shift and within-context change. Ratio of mean within-change energy to mean native within-energy is separate from ratio to total native energy.',
        'zero':'Ratios and cosine are null when denominator is zero. All raw energies retained. No post-outcome filtering of contexts.',
        'limits':'Small relative to total read energy need not be small relative to question variation. Unchanged kernel or fit cannot uniquely identify redundancy or absence of a source mechanism. Norms/cosines are not semantic/casual attribution. Nonconfirmation only.'}
    immutable(path,value);return value


def main(key):
    start=time.monotonic();spec=freeze();folder=Path('source_coupling')/key;final=OUT/folder/'relative_strength.json'
    if final.exists():
        assert read(final)['execution_sha256']==sha(OUT/'source_coupling/relative_execution.json');return
    source=read(OUT/folder/'result.json');assert source['all_passed']
    rows,groups,questions=data.index(key,{'train','validation','diagnostic'})
    means={};refs=[]
    for variant in ['native_source_read','source_value_pair_shuffle']:
        record=read(OUT/'fit'/key/variant/'evaluation.json');ref=record['operator_fields']['selected'];refs.append(ref)
        assert sha(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path']) as z:means[variant]={k:z[k].copy() for k in ['query_mean','query_scale']}
    hs=[];reads=[];shuffled=[]
    for row in rows:
        fields=data.field(questions[row['question_id']]['field'],['H12_last_BF16','native_source_read_BF16','source_value_pair_shuffle_BF16'])
        hs.append(fields['H12_last_BF16']);reads.append(fields['native_source_read_BF16']);shuffled.append(fields['source_value_pair_shuffle_BF16'])
    h=np.stack(hs);r=np.stack(reads);s=np.stack(shuffled);d=r.shape[1]
    a=means['native_source_read'];b=means['source_value_pair_shuffle']
    np.testing.assert_array_equal(a['query_mean'][:d],b['query_mean'][:d]);np.testing.assert_array_equal(a['query_scale'][:d],b['query_scale'][:d])
    rnorm=(r-a['query_mean'][d:])/a['query_scale'][d:]
    snorm=(s-b['query_mean'][d:])/b['query_scale'][d:]
    hnorm=(h-a['query_mean'][:d])/a['query_scale'][:d]
    qnative=np.concatenate([hnorm,rnorm],axis=1);qpermuted=np.concatenate([hnorm,snorm],axis=1)
    train=np.array([row['split']=='train' for row in rows]);assert train.sum()==768
    kr=qnative@qnative[train].T/qnative.shape[1];ks=qpermuted@qpermuted[train].T/qpermuted.shape[1]
    spaces={'raw_read':(r,s),'common_native_training_scale':(rnorm,(s-a['query_mean'][d:])/a['query_scale'][d:]),
        'respective_frozen_route_scales':(rnorm,snorm),'full_query_respective_scales':(qnative,qpermuted),'frozen_query_kernel':(kr,ks)}
    ids=[];locations={}
    for i,row in enumerate(rows):locations.setdefault(row['group_id'],[]).append(i)
    values=[]
    for gid,take in locations.items():
        assert len(take)==4;row=rows[take[0]]
        ids.append({'group_id':gid,'split':row['split'],'cohort':row['cohort'],'question_ids':[rows[i]['question_id'] for i in take]})
        values.append(np.stack([group_statistics(left[take],right[take]) for left,right in spaces.values()]))
    arrays={'all_context_space_metrics':np.stack(values),'native_query_kernel':kr,'permuted_query_kernel':ks}
    summaries=[]
    for split in ['train','validation','diagnostic']:
        for cohort in ['drop','quoref']:
            take=np.array([row['split']==split and row['cohort']==cohort for row in ids])
            for j,space in enumerate(spaces):
                average=arrays['all_context_space_metrics'][take,j].mean(0);m=dict(zip(METRICS,map(float,average)))
                m['relative_total_permutation_energy']=m['permutation_MSE']/m['native_mean_square'] if m['native_mean_square']>0 else None
                m['relative_question_permutation_energy']=m['within_permutation_MSE']/m['within_native_mean_square'] if m['within_native_mean_square']>0 else None
                denom=np.sqrt(m['within_native_mean_square']*m['within_permuted_mean_square'])
                m['pooled_question_response_cosine']=m['within_native_permuted_inner_product']/denom if denom>0 else None
                summaries.append({'split':split,'cohort':cohort,'space':space,'contexts':int(take.sum()),'values':m})
    reference=commit_arrays(folder,'question_relative_source_strength',arrays)
    value={'timestamp':stamp(),'all_passed':True,'model':key,'source_coupling_result_sha256':sha(OUT/folder/'result.json'),
        'execution_sha256':sha(OUT/'source_coupling/relative_execution.json'),'frozen_normalization_sources':refs,
        'spaces':list(spaces),'metrics':METRICS,'identities':ids,'query_kernel_row_question_ids':[r['question_id'] for r in rows],
        'query_kernel_column_training_question_ids':[r['question_id'] for r,t in zip(rows,train) if t],
        'field':reference,'summaries':summaries,'seconds':time.monotonic()-start,'limits':spec['limits']}
    immutable(final,value);print('NATURAL_SOURCE_RELATIVE_COMPLETE',key,round(value['seconds'],1),flush=True)
    for item in summaries:
        if item['split']=='diagnostic':print('NATURAL_SOURCE_RELATIVE_DIAGNOSTIC',key,item['cohort'],item['space'],item['values'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen4','qwen14','glm4']);p.add_argument('--unit',action='store_true');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.unit:unit()
    elif a.freeze_only:freeze()
    else:
        assert a.model;main(a.model)
