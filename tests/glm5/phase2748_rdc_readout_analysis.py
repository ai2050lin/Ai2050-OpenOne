"""All fixed full-vocabulary first-prefix readouts; paired context summaries."""
import argparse
from rdc_question_common import *
from phase2748_rdc_fit_analysis import bootstrap
import rdc_question_data as data

METRICS=['native_to_predicted_KL','native_top1_agreement','predicted_first_teacher_NLL']
NATIVE=['native_choice','native_entropy','native_first_teacher_NLL','native_chosen_probability']


def freeze():
    path=OUT/'readout_analysis/execution.json'
    rev={'source':snapshot(__file__),'bootstrap':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py')),
        'readout':snapshot(Path(__file__).with_name('phase2748_rdc_readout.py')),
        'material_manifest_sha256':sha(OUT/'material/manifest.json')}
    if path.exists():
        old=read(path);assert old['execution']==rev;return old
    assert not any((OUT/'fit').glob('*/readout/*/result.json')),'Freeze before formal full vocabulary readout outcomes'
    value={'timestamp':stamp(),'execution':rev,'metrics':METRICS,'before_any_formal_readout_result':True,
        'comparisons':'All9routes selected-versus-ownalpha0; fixed primary selected versus each otherselectedroute. No selection by readout or diagnostic/confirmation.',
        'estimator':'Allquestions,4percontext, allcontexts. Bothcohorts separately andequal;2000whole-context pairedbootstrapseeds2748014/2748015; descriptive95percent no multiplicity correction.',
        'native_baseline':'Same original source logit summaries must be identical in every route andkind, allquestions; original unembeddingCUDAreplay is separately required by readoutstage.',
        'limits':'Only first-prefix vocabulary distribution; often commonJSONformat. KL/top1/firstteacherNLL improvement not semantic answer generation or subsequent-historyclosure. Castingerror and predictedentropy shown without assuming monotonic scientific quality.',
        'confirmation':'Same fixed implementation only after gate; no re-fitting, alpha or checkpoint choice.'}
    immutable(path,value);return value


def main(key,confirmation=False):
    start=time.monotonic();spec=freeze();scope='confirmation'if confirmation else'nonconfirmation'
    folder=Path('readout_analysis')/key/scope;path=OUT/folder/'result.json'
    if path.exists():
        old=read(path);assert old['execution_sha256']==sha(OUT/'readout_analysis/execution.json')
        print('NATURAL_READOUT_ANALYSIS_ALREADY_COMPLETE',key,scope,flush=True);return
    source=read(OUT/'fit'/key/'readout'/scope/'result.json');assert source['all_passed']
    selection=read(OUT/'fit'/key/'validation_selection.json');primary=selection['primary_rule']
    variants=read(OUT/'fit'/key/'result.json')['variants'];splits=['confirmation']if confirmation else['validation','diagnostic']
    records={};matrices={};summaries=[];comparisons=[]
    for split in splits:
        rows,_,_=data.index(key,{split});ids=[r['question_id']for r in rows];columns=None;native=None
        for variant in variants:
            for kind in ['selected','alpha0']:
                item=next(r for r in source['records']if r['split']==split and r['variant']==variant and r['kind']==kind)
                assert item['question_ids']==ids and item['selection_sha256']==sha(OUT/'fit'/key/'validation_selection.json')
                if columns is None:columns=item['columns']
                assert columns==item['columns'];ref=item['field'];assert sha(ROOT/ref['path'])==ref['sha256']
                with np.load(ROOT/ref['path'])as z:array=z['statistics'].copy()
                assert array.shape==(len(rows),len(columns))and np.isfinite(array).all()
                nn=array[:,[columns.index(c)for c in NATIVE]]
                if native is None:native=nn
                else:assert np.array_equal(native,nn)
                matrices[(variant,kind,split)]=array;records[(variant,kind,split)]=item
                summary={}
                for cohort in ['drop','quoref']:
                    take=np.array([r['cohort']==cohort for r in rows])
                    summary[cohort]={'questions':int(take.sum()),'contexts':int(take.sum()/4),
                        **{c:float(array[take,i].mean())for i,c in enumerate(columns)if c not in ['native_choice','predicted_choice']}}
                summary['equal_cohort']={c:float(np.mean([summary[h][c]for h in ['drop','quoref']]))for c in summary['drop']if c not in ['questions','contexts']}
                summaries.append({'split':split,'variant':variant,'kind':kind,'summary':summary,'source_field':ref})
        pairs=[(v,'selected',v,'alpha0')for v in variants]+[(primary,'selected',v,'selected')for v in variants if v!=primary]
        for lv,lk,rv,rk in pairs:
            for metric in METRICS:
                i=columns.index(metric)
                comparisons.append({'split':split,'left':lv+':'+lk,'right':rv+':'+rk,'metric':metric,
                    'paired':bootstrap(rows,matrices[(lv,lk,split)][:,i],matrices[(rv,rk,split)][:,i],2748014)})
    result={'timestamp':stamp(),'all_passed':True,'model':key,'scope':scope,
        'execution_sha256':sha(OUT/'readout_analysis/execution.json'),
        'readout_result_sha256':sha(OUT/'fit'/key/'readout'/scope/'result.json'),
        'selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'native_baselines_identical_across_every_route_and_kind':True,'summaries':summaries,'paired_comparisons':comparisons,
        'native_first_choice_counts':source['native_first_choice_counts'],'limits':spec['limits'],'seconds':time.monotonic()-start}
    immutable(path,result);print('NATURAL_READOUT_ANALYSIS_COMPLETE',key,scope,len(comparisons),round(result['seconds'],1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen4','qwen14','glm4']);p.add_argument('--confirmation',action='store_true');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze()
    else:
        assert a.model;main(a.model,a.confirmation)
