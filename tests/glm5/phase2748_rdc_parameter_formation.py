"""Every actual learned parameter word: exact inversion, full-vector moments."""
from rdc_question_common import *
from rdc_question_checkpoint_codec import decode_FP32,decode_BF16
from rdc_relation_native_parameters import parameter as native_parameter


def pair_moments(original,left,right,chunk=1048576):
    """Bounded FP64 summation over every scalar; no sparsity/coordinate sampling."""
    o=np.asarray(original).reshape(-1)
    la,lb=(np.asarray(x).reshape(-1)for x in left)
    ra,rb=(np.asarray(x).reshape(-1)for x in right)
    assert len(o)==len(la)==len(lb)==len(ra)==len(rb)
    result={k:0. for k in ['FP32_delta_dot','BF16_delta_dot','left_FP32_BF16_delta_dot','left_cast_error_squared','original_norm_squared']}
    result.update({'left_FP32_changed_words':0,'left_BF16_changed_words':0,'left_FP32_maximum_change':0.,'left_BF16_maximum_change':0.})
    for begin in range(0,len(o),chunk):
        end=min(begin+chunk,len(o));oldbits=o[begin:end]
        old=unbits(oldbits).astype(np.float64)
        a=la[begin:end].astype(np.float64);b=unbits(lb[begin:end]).astype(np.float64)
        c=ra[begin:end].astype(np.float64);d=unbits(rb[begin:end]).astype(np.float64)
        da,db,dc,dd=a-old,b-old,c-old,d-old
        result['FP32_delta_dot']+=float(np.dot(da,dc))
        result['BF16_delta_dot']+=float(np.dot(db,dd))
        result['left_FP32_BF16_delta_dot']+=float(np.dot(da,db))
        result['left_cast_error_squared']+=float(np.sum((a-b)**2))
        result['original_norm_squared']+=float(np.dot(old,old))
        result['left_FP32_changed_words']+=int(np.count_nonzero(la[begin:end].view(np.uint32)!=(oldbits.astype(np.uint32)<<16)))
        result['left_BF16_changed_words']+=int(np.count_nonzero(lb[begin:end]!=oldbits))
        result['left_FP32_maximum_change']=max(result['left_FP32_maximum_change'],float(np.max(np.abs(da))))
        result['left_BF16_maximum_change']=max(result['left_BF16_maximum_change'],float(np.max(np.abs(db))))
    return result


def freeze():
    path=OUT/'parameter_formation/execution.json'
    rev={'source':snapshot(__file__),'codec':snapshot(Path(__file__).with_name('rdc_question_checkpoint_codec.py')),
        'reader':snapshot(Path(__file__).with_name('rdc_relation_native_parameters.py')),
        'training_material_sha256':sha(OUT/'training/material_manifest.json')}
    if path.exists():
        old=read(path);assert old['execution']==rev;return old
    runs=[r['run']for r in read(OUT/'training/material_manifest.json')['run_inventory']]
    assert not any(list((OUT/'training'/run/'steps').glob('*.json'))for run in runs)
    unit=read(OUT/'unit/parameter_formation_current.json')
    assert unit['all_passed']and unit['analysis']['sha256']==rev['source']['sha256']
    value={'timestamp':stamp(),'execution':rev,'before_any_formal2748_optimizer_step':True,
        'unit_sha256':sha(OUT/'unit/parameter_formation_current.json'),'runs':runs,
        'parameters':'All74711040original Q4block16gate/up/down scalar parameters, both actualFP32finaltraining and actualGPUcastBF16deployment. No parameter selected by change/amplitude.',
        'verification':'Original training checkpoint manifest shardSHA, each learned archiveSHA and every decoded wordSHA against save-time actualtensor hashes.',
        'moments':'All6x6pairwise update innerproducts/norms/cosines separatelyFP32andBF16. Every coordinate accumulatedinFP64in1,048,576scalar chunks; changedword counts/maxchanges andBF16casting error separately.',
        'memory':'At mosttwo run-specific parameter pairs decodedatonce; no six-model stack or GPU load. Original immutablecheckpoint accessedread-only.',
        'limits':'A parameter update direction is actual fine-tuning evidence, not a semantic basis or reconstruction ofpretraining. Cosine undefined for zero-normdirection isnull; two seeds not a seed-population estimate.'}
    immutable(path,value);return value


def decoded(run,record,original):
    reference=record['field'];assert sha(ROOT/reference['path'])==reference['sha256']
    with np.load(ROOT/reference['path'])as z:
        a=decode_FP32(z['FP32_XOR_words'],original)
        b=decode_BF16(z['BF16_XOR_words'],original)
    assert hashlib.sha256(memoryview(a).cast('B')).hexdigest()==record['FP32_actual_word_sha256']
    assert hashlib.sha256(memoryview(b).cast('B')).hexdigest()==record['BF16_actual_CUDA_cast_word_sha256']
    assert a.dtype==np.float32 and b.dtype==np.uint16 and np.isfinite(a).all()and np.isfinite(unbits(b)).all()
    return a,b


def cosine(gram):
    return [[float(gram[i,j]/np.sqrt(gram[i,i]*gram[j,j]))if gram[i,i]>0 and gram[j,j]>0 else None
             for j in range(len(gram))]for i in range(len(gram))]


def main():
    start=time.monotonic();spec=freeze();final=OUT/'parameter_formation/result.json'
    if final.exists():
        assert read(final)['execution_sha256']==sha(OUT/'parameter_formation/execution.json')
        print('NATURAL_PARAMETER_FORMATION_ALREADY_COMPLETE',flush=True);return
    training=read(OUT/'training/result.json');assert training['all_passed']
    original_manifest=read(OUT/'training/original_checkpoint_manifest.json')
    checkpoint_folder=Path(original_manifest['folder'])
    for name,record in original_manifest['shards'].items():assert sha(checkpoint_folder/name)==record['sha256']
    names=spec['runs'];assert len(names)==6
    checkpoints={run:read(OUT/'training'/run/'checkpoint96.json')for run in names}
    for run,record in checkpoints.items():
        assert record['all_passed']and record['total_parameters']==74711040
        assert record['original_manifest_sha256']==sha(OUT/'training/original_checkpoint_manifest.json')
    gram32=np.zeros((6,6));gram16=np.zeros((6,6));details=[];selfstats={run:{}for run in names}
    count=0
    for name in original_manifest['trainable_names']:
        original=native_parameter(ROOT,name,'qwen3-4b');count+=original.size
        records={run:next(p for p in checkpoints[run]['parameters']if p['parameter']==name)for run in names}
        for i,run in enumerate(names):
            left=decoded(run,records[run],original)
            for j in range(i,6):
                other=names[j];right=left if i==j else decoded(other,records[other],original)
                moments=pair_moments(original,left,right)
                gram32[i,j]+=moments['FP32_delta_dot'];gram16[i,j]+=moments['BF16_delta_dot']
                if i==j:selfstats[run][name]=moments
                details.append({'parameter':name,'left':run,'right':other,'moments':moments})
                if i!=j:del right
            del left
            print('NATURAL_PARAMETER_FORMATION',name,run,round(time.monotonic()-start,1),flush=True)
        del original
    assert count==74711040
    gram32=gram32+gram32.T-np.diag(np.diag(gram32));gram16=gram16+gram16.T-np.diag(np.diag(gram16))
    summaries=[]
    for i,run in enumerate(names):
        parts=list(selfstats[run].values());o=sum(v['original_norm_squared']for v in parts)
        fp=float(np.sqrt(gram32[i,i]));bf=float(np.sqrt(gram16[i,i]));cross=sum(v['left_FP32_BF16_delta_dot']for v in parts)
        summaries.append({'run':run,'FP32_update_L2':fp,'BF16_update_L2':bf,'original_parameter_L2':float(np.sqrt(o)),
            'FP32_relative_update_L2':fp/np.sqrt(o),'BF16_relative_update_L2':bf/np.sqrt(o),
            'FP32_changed_words':sum(v['left_FP32_changed_words']for v in parts),
            'BF16_changed_words':sum(v['left_BF16_changed_words']for v in parts),
            'FP32_maximum_parameter_change':max(v['left_FP32_maximum_change']for v in parts),
            'BF16_maximum_parameter_change':max(v['left_BF16_maximum_change']for v in parts),
            'actual_FP32_to_actual_BF16_rounding_L2':float(np.sqrt(sum(v['left_cast_error_squared']for v in parts))),
            'FP32_BF16_direction_cosine':float(cross/(fp*bf))if fp>0 and bf>0 else None})
    reference=commit_arrays(Path('parameter_formation'),'full_update_grams',{'FP32_delta_gram':gram32,'BF16_delta_gram':gram16})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'runs':names,'parameters':count,
        'execution_sha256':sha(OUT/'parameter_formation/execution.json'),
        'original_manifest_sha256':sha(OUT/'training/original_checkpoint_manifest.json'),
        'checkpoint_receipts':{run:sha(OUT/'training'/run/'checkpoint96.json')for run in names},
        'all_decoded_FP32_and_BF16_words_match_actual_saved_word_hashes':True,'summaries':summaries,
        'field':reference,'FP32_direction_cosines':cosine(gram32),'BF16_direction_cosines':cosine(gram16),
        'per_parameter_pair_details':details,'seconds':time.monotonic()-start,'limits':spec['limits']}
    immutable(final,result)
    print('NATURAL_PARAMETER_FORMATION_COMPLETE',count,round(result['seconds'],1),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    freeze()if args.freeze_only else main()
