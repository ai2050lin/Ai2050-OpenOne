"""Explicitly version a failed pre-checkpoint persistence assertion, not the science."""
from rdc_formation_common import *


def main():
    path = OUT / 'training/protocol.json'
    prior = read(path)
    oldhash = prior['engine']['sha256']
    original_hash = 'd3d703a7882b7b5e8b8548e5fc4427c34f868d798baaeace0749189808fcad9f'
    engine = ROOT / 'tests/glm5/phase2747_rdc_training.py'
    if (OUT / 'training/persistence_recovery.json').exists():
        assert prior['engine']['sha256'] == sha(engine)
        print('PERSISTENCE_RECOVERY_ALREADY_REGISTERED', flush=True)
        return
    assert oldhash == original_hash
    failures = sorted((OUT / 'training').glob('failure_*.json'))
    assert failures and any('original[n].numpy()+v' in read(f)['traceback'] for f in failures)
    # No actual learning checkpoint had been committed before the failing
    # assertion. Native/bridge baselines and initial gradients remain unchanged.
    assert not list((OUT / 'training').glob('*_274[78]/commits/*.json'))
    revision = OUT / 'training/protocol_revisions' / (sha(path)+'.json')
    revision.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, revision)
    new = dict(prior)
    new['engine'] = snapshot(engine)
    new['parameter_persistence'] = (
        'Store everyFP32delta coordinate and a completeFP32reconstruction_residual array. '
        'Reconstruct as FP32(FP32(original+delta)+residual), verify allUINT32bits. '
        'Norms/gradient products use exactFP64 differences of actualFP32parameter endpoints. '
        'No tiny-coordinate threshold or sparse semantic selection.')
    assert {k: v for k, v in new.items() if k not in ['engine', 'parameter_persistence']} == {
        k: v for k, v in prior.items() if k != 'engine'}
    save(OUT / 'training/persistence_recovery.json', {'timestamp': stamp(), 'source': snapshot(__file__),
        'old_protocol_snapshot': str(revision.relative_to(BASE)), 'old_engine_sha256': oldhash,
        'new_engine_sha256': new['engine']['sha256'], 'failures_preserved': [str(f.relative_to(BASE)) for f in failures],
        'change': 'Strengthen exact checkpoint reconstruction using full-coordinate lossless residual; do not waive equality.',
        'unchanged': 'Material, splits, all supervision, all update mathematics, step norm, draw order, checkpoints, evaluation and selections.',
        'state_at_failure': 'Native/bridge baselines and initial gradients committed; step1update computed but assertion prevented any learning checkpoint commit.',
        'original_model_or_old_results_modified': False})
    save(path, new)
    print('PERSISTENCE_RECOVERY_REGISTERED', flush=True)


if __name__ == '__main__':
    main()
