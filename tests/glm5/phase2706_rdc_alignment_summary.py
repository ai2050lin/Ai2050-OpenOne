"""Paired summaries and within-family/language pairing null controls for full-state model maps."""
import argparse
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel,group_ids
from phase2704_rdc_predictive_gates import errors
OUT=CAMPAIGN/'l_aligned'


def prepare():
    immutable(OUT/'pairing_control_protocol.json',{'source_sha':sha(Path(__file__)),'timestamp_frozen_before_results':'See file metadata and source journal; frozen before serial all-model analysis',
      'seeds':[611,612,613],'algorithm':'Keep same family/language and64training cases, permute target rows only within each of16 four-case family/language traininggroups; validation/test remain correctly paired. Fit samefullnativecoordinate linear/quadratic kernel andridgegrid. Compare true pairing vs thisfamilyprototype-preserving null.',
      'stages':'prefill-prefill andcontent-content only; Q4H24->Q14/GLMH27, allnativeinput/outputcoordinates. Not an independently deployable targettimestep selector for content stage.',
      'limits':['Three permutations are sensitivity controls, not a calibrated significance test.','Only64training and2heldoutentitygroups; groupedtemplate signatures can dominate rawstate MSE.','A paired prediction advantage does not establish a shared native program or identifiable tensor basis.']})


def main():
    prepare();result=read(OUT/'result.json');assert result
    rows=read(OUT/'qwen4/prefixes.json');tr,va,te=splits(rows);groups=group_ids(rows,'family_language')
    with np.load(OUT/'qwen4/features.npz') as z:q4={stage:unbits(z[stage+'_h'][:,24]) for stage in ('prefill','content')}
    nulls=[];phase_models={};behaviors={}
    for key in ('qwen4','qwen14','glm4'):
        rr=read(OUT/key/'prefixes.json');bb=[read(OUT/key/f'behavior/{r["sample_id"]}.json') for r in rr];behaviors[key]=bb
        with np.load(OUT/key/'features.npz') as z:identical=np.all(z['prefill_h']==z['content_h'],axis=(1,2))
        hlist=(0,12,24,36) if key=='qwen4' else (0,13,27,40)
        phase_models[key]={'prefixes':256,'first_correct':result['models'][key]['first_token_correct'],'content_correct':result['models'][key]['first_content_correct'],
          'content_pair_correct':result['models'][key]['content_candidate_pair_correct'],'content_step_counts':result['models'][key]['content_step_counts'],
          'identical_prefill_content_all_H':int(identical.sum()),'fixed_checkpoint_readers':[r for r in result['models'][key]['readers'] if r['H'] in hlist]}
    paired=[]
    for left,right in (('qwen4','qwen14'),('qwen4','glm4'),('qwen14','glm4')):
        a,b=behaviors[left],behaviors[right]
        counts={'both_correct':0,'left_only':0,'right_only':0,'both_wrong':0}
        for x,y in zip(a,b):
            xc,yc=x['content_token_correct'],y['content_token_correct'];counts['both_correct' if xc and yc else 'left_only' if xc else 'right_only' if yc else 'both_wrong']+=1
        paired.append({'left':left,'right':right,'n':256,**counts})
    for key in ('qwen14','glm4'):
        with np.load(OUT/key/'features.npz') as z:target={stage:unbits(z[stage+'_h'][:,27]) for stage in ('prefill','content')}
        for stage in ('prefill','content'):
          for kind in ('linear','quadratic'):
            f=FullKernel(q4[stage],tr,va,te,kind);y=target[stage]
            for seed in (611,612,613):
                rng=np.random.default_rng(seed);yn=y.copy();permutation=np.arange(len(rows))
                for group in range(16):
                    ii=tr[groups[tr]==group];assert len(ii)==4
                    permutation[ii]=rng.permutation(ii)
                yn[tr]=y[permutation[tr]];mid=f'null_{key}_{stage}_{kind}_{seed}'
                p,m=f.fit(yn,OUT/f'models/{mid}.npz');metrics,arr=errors(y[te],p,np.mean(y[tr].astype(np.float64)**2,0))
                nulls.append({'target_model':key,'stage':stage,'algorithm':kind,'seed':seed,**m,**metrics,'training_fixed_points':int(np.sum(permutation[tr]==tr))})
                npz(OUT/f'predictions/{mid}.npz',prediction=p,target=y[te],test=te,training_target_permutation=permutation,**arr)
            print('PAIRING_NULLS',key,stage,kind,flush=True)
    comparisons=[]
    for key in ('qwen14','glm4'):
     for stage in ('prefill','content'):
      for kind in ('linear','quadratic'):
        true=next(r for r in result['crossmodel_maps'] if r['target_model']==key and r['fit_stage']==r['test_stage']==stage and r['algorithm']==kind)
        nr=[r for r in nulls if (r['target_model'],r['stage'],r['algorithm'])==(key,stage,kind)]
        comparisons.append({'target_model':key,'stage':stage,'algorithm':kind,'true_pair_mse':true['mse'],'true_pair_energy_ratio':true['total_error_energy_ratio'],
          'null_mse':[r['mse'] for r in nr],'null_energy_ratio':[r['total_error_energy_ratio'] for r in nr],
          'true_better_than_all_three':all(true['mse']<r['mse'] for r in nr)})
    save(OUT/'summary_audit.json',{'timestamp':stamp(),'phase':2706,'models':phase_models,'paired_native_content_correctness':paired,
      'crossmodel_maps':result['crossmodel_maps'],'pairing_null_comparisons':comparisons,'pairing_nulls':nulls,'limits':read(OUT/'pairing_control_protocol.json')['limits']})
    print('ALIGNED_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else main()
