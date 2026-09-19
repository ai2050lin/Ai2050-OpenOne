"""Open confirmation only after all frozen prerequisites genuinely complete."""
import argparse
import ast
from rdc_question_common import *

MODEL_KEYS=['qwen4','qwen14','glm4']
CODE_FILES=['phase2748_rdc_acquire.py','rdc_question_native.py','rdc_question_history.py',
    'phase2748_rdc_fit.py','rdc_question_fit.py','rdc_question_data.py',
    'phase2748_rdc_readout.py','phase2748_rdc_prospective.py','rdc_question_predictor.py',
    'phase2748_rdc_training.py','rdc_question_learning.py','rdc_question_checkpoint.py',
    'phase2748_rdc_learned.py','phase2748_rdc_confirmation_fit.py','phase2748_rdc_learning_analysis.py',
    'phase2748_rdc_parameter_formation.py','phase2748_rdc_prospective_analysis.py','phase2748_rdc_readout_analysis.py',
    'phase2748_rdc_identity_audit.py','phase2748_rdc_generated_identity_audit.py',
    'phase2748_rdc_training_bridge_baseline.py']


def local_import_names(source):
    """Static direct local imports, including imports inside execution functions."""
    result=set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):
            result.update(alias.name.split('.')[0]for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module and not node.level:
            result.add(node.module.split('.')[0])
    return result


def source_closure():
    directory=Path(__file__).parent
    pending=[directory/name for name in CODE_FILES]+[Path(__file__)]
    found={}
    while pending:
        path=pending.pop().resolve()
        if str(path)in found:continue
        assert path.parent==directory.resolve() and path.suffix=='.py' and path.is_file()
        found[str(path)]=path
        for module in local_import_names(path.read_text(encoding='utf-8-sig')):
            candidate=directory/(module+'.py')
            if candidate.is_file():pending.append(candidate)
    return sorted(found.values())


def requirements():
    files=[]
    for key in MODEL_KEYS:
        files.extend([f'native/{key}/nonconfirmation/result.json',f'fit/{key}/result.json',
            f'fit/{key}/readout/nonconfirmation/result.json',f'confirmation/validation_replay/{key}/result.json',
            f'prospective_analysis/{key}/diagnostic/result.json',f'readout_analysis/{key}/nonconfirmation/result.json',
            f'verification/native_{key}_nonconfirmation.json',
            f'verification/generated_prospective_{key}_diagnostic.json'])
        selection=read(OUT/'fit'/key/'validation_selection.json')if(OUT/'fit'/key/'validation_selection.json').exists()else None
        if selection:
            for variant in [selection['primary_rule'],selection['control_rule']]:
                files.append(f'prospective/{key}/{variant}/diagnostic/result.json')
        else:files.append(f'fit/{key}/validation_selection.json')
    files+=['training/pilot/result.json','training/result.json','parameter_formation/result.json',
        'learning_analysis/nonconfirmation/result.json','training_bridge_baseline/result.json']
    inventory=read(OUT/'training/material_manifest.json')['run_inventory']
    for item in inventory:
        run=item['run'];files.extend([f'training/{run}/result.json',f'learned/{run}/nonconfirmation/result.json',
            f'verification/generated_learned_{run}_nonconfirmation.json'])
    return files


def status():
    entries=[]
    for name in requirements():
        path=OUT/name
        record=read(path)if path.exists()else{}
        complete=bool(record.get('all_passed',False))
        entries.append({'path':name,'exists':path.exists(),'complete':complete})
    return {'timestamp':stamp(),'all_prerequisites_complete':all(r['complete']for r in entries),'requirements':entries,
        'note':'Diagnostic readout, autonomous histories, BF16learned behaviors and validation-replay checks are conservatively completed in addition to the original minimum gate. They never select a new rule/checkpoint.'}


def reference(path):
    path=Path(path)
    assert path.is_file() and path.resolve().is_relative_to(ROOT.resolve()) or path.is_file() and path.resolve().is_relative_to(PHYSICAL.resolve())
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}


def check_ref(ref):
    path=ROOT/ref['path']
    assert path.is_file() and sha(path)==ref['sha256'],ref['path']


def verify():
    path=OUT/'confirmation/freeze_certificate.json'
    value=read(path);assert value['all_passed']
    assert value['material_manifest_sha256']==sha(OUT/'material/manifest.json')
    assert {str(p.resolve())for p in source_closure()}=={
        str((ROOT/ref['path']).resolve())for ref in value['fixed_source_files']},'Local import dependency set changed after freeze'
    for ref in value['fixed_artifacts']+value['fixed_source_files']:check_ref(ref)
    for item in value['native']:
        for ref in item['groups']:check_ref(ref)
    for item in value['trained_checkpoints']:
        check_ref(item['checkpoint'])
        record=read(ROOT/item['checkpoint']['path'])
        for parameter in record['parameters']:check_ref(parameter['field'])
    print('NATURAL_CONFIRMATION_CERTIFICATE_VERIFIED',len(value['fixed_artifacts']),flush=True)
    return value


def freeze():
    path=OUT/'confirmation/freeze_certificate.json'
    if path.exists():return verify()
    available=status()
    assert available['all_prerequisites_complete'],[r for r in available['requirements']if not r['complete']]
    # Nothing in this function opens the frozen confirmation material or fields.
    for family in ['native','prospective','learned']:
        assert not any((OUT/family).glob('**/confirmation/groups/*.json')),'Confirmation observed before final freeze'
    artifacts=[];native=[]
    artifacts.append(reference(OUT/'material/manifest.json'))
    for r in available['requirements']:artifacts.append(reference(OUT/r['path']))
    for name in ['effective_experiment_contract.json','retention_contract.json','training/material_manifest.json',
        'training/formal_protocol.json','training/execution_contract.json','training/original_checkpoint_manifest.json',
        'fit/implementation_contract.json','confirmation/evaluation_execution.json','learned/execution.json',
        'learning_analysis/execution.json','parameter_formation/execution.json','prospective_analysis/execution.json',
        'readout_analysis/execution.json','unit/generated_identity_current.json',
        'training_bridge_baseline/execution.json','training_bridge_baseline/acquisition.json',
        'training_bridge_baseline/metadata_qualification.json',
        'unit/training_bridge_current.json']:
        artifacts.append(reference(OUT/name))
    for key in MODEL_KEYS:
        result=read(OUT/f'native/{key}/nonconfirmation/result.json')
        assert result['all_passed']and result['contexts']==336 and result['questions']==1344 and result['native_free_histories']==576
        assert len(result['group_receipts'])==336
        commits=[];all_ids=set();all_questions=set();field_refs={}
        for name in result['group_receipts']:
            group=read(OUT/name)
            assert group['execution']==result['execution']and group['split']!='confirmation'
            assert len(group['questions'])==4 and group['group_id']not in all_ids
            all_ids.add(group['group_id']);commits.append(reference(OUT/name))
            for ref in [group['context_field']]+[q['field']for q in group['questions']]:field_refs[ref['path']]=ref
            for q in group['questions']:
                assert q['question_id']not in all_questions;all_questions.add(q['question_id'])
                for kind in ['teacher','history']:
                    if kind in q:field_refs[q[kind]['field']['path']]=q[kind]['field']
        assert len(all_questions)==1344
        for ref in field_refs.values():check_ref(ref)
        native.append({'model':key,'groups':commits,'all_native_archive_files_checked':len(field_refs)})
        selection=read(OUT/'fit'/key/'validation_selection.json')
        for variant in [selection['primary_rule'],selection['control_rule']]:
            recordpath=OUT/'fit'/key/'deployed'/(variant+'.json')
            record=read(recordpath);check_ref(record['field'])
            artifacts.extend([reference(recordpath),record['field']])
        for name in [f'fit/{key}/validation_selection.json',f'prospective/{key}/execution.json',f'prospective/{key}/pilot/result.json']:
            artifacts.append(reference(OUT/name))
        for folder in ['pilot','history_cost']:
            pointer=read(OUT/folder/key/'current.json');check_ref(pointer)
            record=read(ROOT/pointer['path']);assert record['all_passed']
            artifacts.extend([reference(OUT/folder/key/'current.json'),reference(ROOT/pointer['path'])])
    checkpoints=[]
    training=read(OUT/'training/result.json')
    assert training['actual_optimizer_steps']==576 and training['actual_training_question_draws']==4608 and len(training['runs'])==6
    for run in training['runs']:
        name=run['run'];path=OUT/'training'/name/'checkpoint96.json'
        assert run['steps']==96 and run['questions']==768 and run['all_other_parameters_word_identical']
        assert run['checkpoint96_sha256']==sha(path)
        checkpoint=read(path)
        assert checkpoint['all_passed']and checkpoint['step']==96 and checkpoint['total_parameters']==74711040
        for parameter in checkpoint['parameters']:
            assert parameter['persisted_inverse_FP32_and_BF16_all_words_equal'];check_ref(parameter['field'])
        checkpoints.append({'run':name,'checkpoint':reference(path)})
    closure_unit=read(OUT/'unit/confirmation_source_closure_current.json')
    assert closure_unit['all_passed'] and closure_unit['gate_sha256']==sha(__file__)
    artifacts.append(reference(OUT/'unit/confirmation_source_closure_current.json'))
    sources=[reference(path)for path in source_closure()]
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'material_manifest_sha256':sha(OUT/'material/manifest.json'),
        'fixed_artifacts':list({r['path']:r for r in artifacts}.values()),'fixed_source_files':sources,
        'native':native,'trained_checkpoints':checkpoints,
        'source_freeze_scope':'All directly named numeric runners plus the transitive static local-Python import closure, including shared loaders, scoring, bootstrap, cache handling and checkpoint codec. External installed package versions/architecture identities remain in original numerical qualifications; not an operating-system snapshot.',
        'rules_and_checkpoints_fixed_without_confirmation':True,'native_confirmation_targets_read_before_certificate':False,
        'scope':'All3original nonconfirmation collections, fixed9route fits, full vocabulary readout, two nativeearly ownhistory diagnostics/model,6complete training endpoints and6nativeBF16learned nonconfirmation evaluations finished. All source and deployment bytes fixed. No confirmation outcome has selected any choice.',
        'parent_requirement':'Run this certificate verification before every confirmation launch; existing frozen collector also enforces material-SHA/all_passed gate.'}
    immutable(OUT/'confirmation/freeze_certificate.json',value)
    print('NATURAL_CONFIRMATION_OPENED',len(value['fixed_artifacts']),flush=True)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.verify:verify()
    else:print(status(),flush=True)
