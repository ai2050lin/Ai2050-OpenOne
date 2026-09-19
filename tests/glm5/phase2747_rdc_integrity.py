"""Fresh checkpoint hashes, complete streaming arrays and frozen formation identities."""
import argparse
from collections import Counter
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays
import phase2745_rdc_construction_integrity as inherited

VERIFY=OUT/'verification'


def start_cache():
    VERIFY.mkdir(parents=True,exist_ok=True)
    cache=VERIFY/'archive_cache.json'
    if not cache.exists():
        prior=BASE/'phase2746/verification/archive_cache.json';shutil.copyfile(prior,cache)
        immutable(VERIFY/'inherited_archive_source.json',{'timestamp':stamp(),'source':snapshot(__file__),
            'prior_cache_sha256':sha(prior),'rule':'Reuse old finite/CRC checks only after fresh completeSHA; fully stream every new archive. Old receipts unmodified.'})
    inherited.OUT=VERIFY


def identities():
    start=time.monotonic();checks=[]
    contract=read(BASE/'protocol.json')
    assert hashlib.sha256(MEMO.read_bytes()[:contract['memo_original_bytes']]).hexdigest()==contract['memo_original_sha256']
    for r in contract['required_prior_artifacts']:assert sha(OLD/r['path'])==r['sha256']
    for path in (BASE/'sources').iterdir():assert sha(path).startswith(path.stem.rsplit('_',1)[-1]),path
    checks.append('Entire pre2745MEMOprefix, all14priorartifacts and allimmutable source snapshots preserved')
    protocol=read(OUT/'material/protocol.json');material=OUT/'material/rows.json.gz'
    assert sha(material)==protocol['material_sha256']
    rows=gzread(material);training=read(OUT/'training/result.json');assert training['all_passed']
    trace_audit=read(VERIFY/'material_trace_identities.json')
    assert trace_audit['all_passed'] and trace_audit['material_sha256']==sha(material)
    assert trace_audit['training_result_sha256']==sha(OUT/'training/result.json')
    assert len(rows['train'])==2048 and len(rows['validation'])==192 and len(rows['fresh'])==192
    assert len(rows['diagnostic'])==512 and len(training['runs'])==6
    assert training['actual_backward_training_examples']==12288
    for r in training['runs']:
        assert r['all_passed'] and r['distinct_drawn_examples']==2048 and r['complete_training_pool_consumed_once']
        assert [c['step'] for c in r['checkpoints']]==[1,8,32,128]
        assert len(r['trace'])==128
        checked_arrays(r['deployment']);checked_arrays(r['deployment_fields'])
    checks.append('Frozen2048training,192validation,512diagnostic,192fresh material and all6actual128step traces independently reconstructed, including everyactual sample draw and label histogram')
    follow=read(OUT/'followup/protocol.json');assert sha(OUT/'followup/material.json.gz')==follow['material_sha256']
    radius=read(OUT/'radius_analysis/result.json');assert radius['all_passed'] and len(radius['records'])==40
    calibration=read(OUT/'calibration_analysis/result.json');assert calibration['all_passed'] and len(calibration['checks'])==54
    assert all(r['argmax_unchanged'] for r in calibration['checks'])
    prop=read(OUT/'parameter_propagation/smooth/result.json');assert prop['all_passed'] and prop['rows']==64
    assert prop['finite_endpoints']==1152 and len(prop['adjoint_checks'])==768 and len(prop['complete_field_receipts'])==20
    assert all(r['absolute_error']<=r['tolerance'] for r in prop['adjoint_checks'])
    checks.append('All40radius,54calibration and1152complete-prefix endpoints/768adjointchecks retain exact frozen scope')
    own=read(OUT/'own_history/analysis/result.json');assert own['all_passed'] and own['complete_runs']==9 and not own['partial']
    assert all(r['scoring_audit']['all_frozen_scoring_objects_recomputed_exactly']
               and r['scoring_audit']['controlled_expressions']==320 for r in own['reports'])
    language_review=read(OUT/'own_history/terminal_review/result.json');assert language_review['all_passed']
    assert language_review['original_analysis_sha256']==sha(OUT/'own_history/analysis/result.json')
    assert language_review['original_GLM_result_sha256']==sha(OUT/'own_history/glm4/native/result.json')
    assert len(language_review['reviews'])==5
    for review in language_review['reviews']:assert sha(ROOT/review['record_path'])==review['record_sha256']
    assert read(OUT/'program_own_history/result.json')['trajectories']==192
    assert read(OUT/'program_own_history/result.json')['all_passed']
    program=read(OUT/'program_own_history/analysis/result.json')
    assert program['all_passed'] and len(program['checks'])==192 and program['pilot_replay_count']==12
    terminal=read(OUT/'program_own_history/terminal_review/result.json');assert terminal['all_passed']
    assert terminal['original_program_result_sha256']==sha(OUT/'program_own_history/result.json')
    assert terminal['original_program_analysis_sha256']==sha(OUT/'program_own_history/analysis/result.json')
    transfer=read(OUT/'transfer/readout_result.json');assert transfer['all_passed'] and transfer['records']==256 and transfer['readouts']==128000
    checks.append('Nine complete native/trained512own-history runs,192programownbranches and all128000fullV mapped readouts verified')
    qualification=read(OUT/'engineering/microbatch/qwen4_qualification/result.json')
    assert qualification['all_passed'] and len(qualification['pilot_replays'])==16
    current_qualification=read(OUT/'engineering/microbatch/current_qualification.json')
    assert sha(ROOT/current_qualification['path'])==current_qualification['sha256']
    revised=read(ROOT/current_qualification['path'])
    assert revised['all_passed'] and len(revised['pilot_replays'])==16
    engine_sha=sha(ROOT/'tests/glm5/rdc_formation_microbatch.py')
    assert revised['engine']['sha256']==engine_sha
    sliced=read(OUT/'engineering/microbatch/slice_reader_current.json')
    assert sha(ROOT/sliced['path'])==sliced['sha256'] and read(ROOT/sliced['path'])['all_passed']
    assert read(ROOT/sliced['path'])['engine']['sha256']==engine_sha
    unit_pointer=read(OUT/'engineering/microbatch/unit_current.json')
    assert sha(ROOT/unit_pointer['path'])==unit_pointer['sha256']
    unit=read(ROOT/unit_pointer['path'])
    assert unit['all_passed'] and unit['engine']['sha256']==engine_sha
    glm_adapter_sha=sha(ROOT/'tests/glm5/rdc_formation_glm_wave.py')
    assert unit['glm_architecture_adapter']['sha256']==glm_adapter_sha
    assert sum(r['model_type']=='glm' for r in unit['checks'])==12
    for model in ['qwen14','glm4']:
        replay=read(OUT/'own_history'/model/'native/wave_pilot_replay.json')
        assert replay['all_passed'] and len(replay['checks'])==16
        assert replay['whole_first_wave_completed_before_main_commit']
        assert read(OUT/'own_history'/model/'native/result.json')['engine']['sha256']==engine_sha
    assert read(OUT/'own_history/glm4/native/result.json')['architecture_adapter']['sha256']==glm_adapter_sha
    checks.append('Qualified unchanged native microbatch arithmetic: Q4 and both offloaded models each replayed all16complete original pilot packets')
    for directory in ['training_v2','propagation_v1','probability_v2','history_v1']:
        review=read(OUT/'figures'/directory/'visual_review.json');assert review['all_passed']
    for pointer in ['current_regression.json','current_evidence_regression.json','current_history_regression.json']:
        current=read(OUT/'client'/pointer);assert sha(ROOT/current['result'])==current['result_sha256']
        assert read(ROOT/current['result'])['all_passed']
        assert read(ROOT/current['image_directory']/'visual_review.json')['all_passed']
    code=read(OUT/'client/code_checks.json');assert code['all_passed']
    for source in code['sources']:assert sha(source['path'])==source['sha256'],('Code changed after build audit',source['path'])
    relocation=read(OUT/'engineering/metadata_relocation_preflight.json')
    for entry in relocation['files']:assert sha(OUT/entry['path'])==entry['sha256'],entry['path']
    checks.append('Actual reviewed images, current build and every3146relocated original metadata byte verified; later appended transfer files do not replace originals')
    archive=read(VERIFY/'archives.json');assert archive['all_passed']
    inventory={r['path']:r for r in read(VERIFY/'archive_cache.json')}
    current_files={p.relative_to(BASE).as_posix() for p in BASE.rglob('*.npz') if '.tmp' not in p.name}
    assert current_files==set(inventory),'Later commits require fresh archive audit'
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,'archive_coverage':archive,
        'checkpoint_audit':read(VERIFY/'model_checkpoint_fingerprints.json'),'seconds':time.monotonic()-start,
        'scope':'Evidence integrity and declared boundaries; not universal semantic closure.'}
    save(VERIFY/'result.json',value);ledger('phase2747_integrity_final',value['seconds'])
    print('FORMATION_INTEGRITY_FINAL',len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['checkpoints','archives','final']);args=parser.parse_args()
    start_cache()
    if args.mode=='checkpoints':inherited.checkpoints()
    elif args.mode=='archives':inherited.archives()
    else:identities()
