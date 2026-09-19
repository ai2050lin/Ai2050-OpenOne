"""Evaluate unchanged first-prefix operators; no confirmation-time fitting."""
import argparse
from rdc_question_common import *
from rdc_question_fit import FeatureKernel, denominators
from phase2748_rdc_fit import freeze as original_fit_freeze, score, identity_rows
from phase2748_rdc_fit_analysis import bootstrap
import rdc_question_data as data


def freeze():
    original_fit_freeze()
    path=OUT/'confirmation/evaluation_execution.json'
    execution={'source':snapshot(__file__),'fit_implementation_sha256':sha(OUT/'fit/implementation_contract.json'),
        'paired_analysis':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py'))}
    if path.exists():
        old=read(path);assert old['execution']==execution;return old
    value={'timestamp':stamp(),'execution':execution,'before_any_confirmation_target':not(OUT/'confirmation/freeze_certificate.json').exists(),
        'algorithm':'Reuse saved complete768x768solution operators, exact training normalization and unchanged training targets. No solve, grid, hyperparameter, target-selection or correction on confirmation.',
        'evaluations':'All9routes x selected/alpha0 x all5registeredtargets; primary and control identities unchanged. All-coordinate/per-question scores and whole-context paired2000bootstrap retained.',
        'qualification':'Before confirmation, validation-replay audit independently rebuilds all90predictions/scores from the retained operators and verifies every saved question/coordinate score within1e-8.',
        'seal':'data.index confirmation requires final freeze_certificate and complete native confirmation. Missing gate raises; never substitutes validation/diagnostic for confirmation.',
        'paired_intervals':'Same2cohort context bootstrap convention/seeds2748005/2748006 as diagnostic. Descriptive95percent, not multiplicity adjusted.',
        'limits':'Unchanged empirical first-prefix prediction only, not an autonomous output mechanism.'}
    immutable(path,value);return value


def main(key,audit=False):
    start=time.monotonic();freeze()
    split='validation'if audit else'confirmation'
    folder=Path('confirmation/validation_replay')/key if audit else Path('fit')/key/'confirmation'
    final=OUT/folder/'result.json'
    if final.exists():
        assert read(final)['execution_sha256']==sha(OUT/'confirmation/evaluation_execution.json')
        print('NATURAL_CONFIRMATION_EVAL_ALREADY_COMPLETE',key,split,flush=True);return
    if not audit:
        cert=read(OUT/'confirmation/freeze_certificate.json');assert cert['all_passed']
        unit=read(OUT/'confirmation/validation_replay'/key/'result.json');assert unit['all_passed']
    selection=read(OUT/'fit'/key/'validation_selection.json')
    fit=read(OUT/'fit'/key/'result.json');assert fit['all_passed']
    assert fit['selection_sha256']==sha(OUT/'fit'/key/'validation_selection.json')
    train,tg,tq=data.index(key,{'train'})
    rows,gg,qq=data.index(key,{split})
    assert len(train)==768 and len(rows)==(192 if audit else 256)
    train_features=data.native_features(key,train,tg,tq)
    eval_features=data.native_features(key,rows,gg,qq)
    groups=np.array([r['group_id']for r in train]);cohorts=np.array([r['cohort']for r in train])
    permutation=data.target_permutation(train)
    checks=[];evaluations=[];matrices={}
    for variant in fit['variants']:
        record=read(OUT/'fit'/key/variant/'evaluation.json')
        for kind,choice in selection['variants'][variant].items():
            reference=record['operator_fields'][kind];assert sha(ROOT/reference['path'])==reference['sha256']
            with np.load(ROOT/reference['path'])as z:
                retained={k:z[k].copy()for k in z.files}
            weights=retained['weights'].copy();op=retained['solution_operator']
            feature=FeatureKernel(*data.variant_features(key,train,train_features,variant),weights,variant=='lexical_position')
            for name,value in feature.state.items():assert np.array_equal(value,retained[name]),(variant,kind,name)
            part=feature.parts(*data.variant_features(key,rows,eval_features,variant))
            cross,column,grand=feature.kernel(part,choice['rho'])
            assert np.array_equal(column,retained['kernel_column_mean'])and grand==float(retained['kernel_grand_mean'])
            for target in fit['full_targets']:
                native=data.targets(train,tq,target);actual=data.targets(rows,qq,target)
                yy=native[permutation]if variant=='within_context_target_pair_shuffle'else native
                mean=weights @ yy;prediction=cross @ (op @ (yy-mean))+mean
                den=denominators(native,groups,cohorts)
                summary,arrays=score(prediction,actual,rows,den)
                if audit:
                    ref=next(r['field']for r in record['evaluations']if r['kind']==kind and r['split']==split and r['target']==target)
                    assert sha(ROOT/ref['path'])==ref['sha256']
                    with np.load(ROOT/ref['path'])as z:
                        assert set(z.files)==set(arrays)
                        errors={name:float(np.max(np.abs(z[name]-value)))for name,value in arrays.items()}
                    assert all(v<1e-8 for v in errors.values()),(variant,kind,target,errors)
                    checks.append({'variant':variant,'kind':kind,'target':target,'all_question_coordinate_maximum_error':errors})
                else:
                    ref=commit_arrays(folder/variant,kind+'_'+target,arrays)
                    evaluations.append({'variant':variant,'kind':kind,'target':target,'split':split,'summary':summary,'field':ref})
                    matrices[(variant,kind,target)]={k:v for k,v in arrays.items()if k.endswith('_by_question')}
                del native,actual,yy,prediction,arrays
            print('NATURAL_CONFIRMATION_EVAL',key,split,variant,kind,round(time.monotonic()-start,1),flush=True)
    comparisons=[];primary=selection['primary_rule']
    if not audit:
        for target in fit['full_targets']:
            for variant in fit['variants']:
                for metric in ['absolute_MSE_by_question','within_MSE_by_question']:
                    comparisons.append({'left':variant+':selected','right':variant+':alpha0','target':target,'metric':metric,
                        'paired':bootstrap(rows,matrices[(variant,'selected',target)][metric],matrices[(variant,'alpha0',target)][metric],2748005)})
            for control in [v for v in fit['variants']if v!=primary]+['zero_response_change']:
                for metric in ['absolute_MSE_by_question','within_MSE_by_question']:
                    if control=='zero_response_change'and metric.startswith('absolute'):continue
                    left=matrices[(primary,'selected',target)]
                    right=left['zero_change_MSE_by_question']if control=='zero_response_change'else matrices[(control,'selected',target)][metric]
                    comparisons.append({'left':primary+':selected','right':control,'target':target,'metric':metric,'paired':bootstrap(rows,left[metric],right,2748005)})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,'split':split,
        'execution_sha256':sha(OUT/'confirmation/evaluation_execution.json'),
        'selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'no_new_fitting_or_selection':True,'validation_replay_only':audit,'questions':len(rows),
        'training_questions':len(train),'identities':identity_rows(rows),'checks':checks,'evaluations':evaluations,
        'paired_comparisons':comparisons,'seconds':time.monotonic()-start}
    if not audit:result['freeze_certificate_sha256']=sha(OUT/'confirmation/freeze_certificate.json')
    immutable(final,result)
    print('NATURAL_CONFIRMATION_EVAL_COMPLETE',key,split,len(checks)if audit else len(evaluations),round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    parser.add_argument('--validation-audit',action='store_true');args=parser.parse_args();main(args.model,args.validation_audit)
