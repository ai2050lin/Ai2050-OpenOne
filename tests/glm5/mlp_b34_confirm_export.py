"""Serialize frozen analysis results and describe the failed selectivity control."""
import numpy as np

from mlp_b34_confirm_v3 import configure
import mlp_b34_confirm_design as D
from shared.dist_protocol.mechanism_trace import unbits


def normalize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {normalize(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def main():
    previous = configure()
    import mlp_b34_confirm_analyze as analysis

    plan = D.checked_plan()
    train = {34: [], 35: []}
    for path in sorted(previous.glob('discovery-*.npz')):
        with np.load(path) as archive:
            for b in train:
                norms = np.linalg.norm(unbits(archive[f'A{b}']), axis=-1)
                train[b].extend((norms[:,1]-norms[:,0]).tolist())
    hold = {34: [], 35: []}
    groups = {}
    for samples, rows, path in analysis.chunks(plan, 'holdout'):
        with np.load(path) as archive:
            for b in hold:
                norms = np.linalg.norm(unbits(archive[f'A{b}']), axis=-1)
                hold[b].extend((norms[:,1]-norms[:,0]).tolist())
        for sample, row in zip(samples, rows):
            correct = np.argmax(row['base_scores'], axis=1) == sample['correct']
            stats = groups.setdefault(sample['frame'], {'n': 0, 'false_correct': 0, 'true_correct': 0, 'both_correct': 0})
            stats['n'] += 1
            stats['false_correct'] += int(correct[0])
            stats['true_correct'] += int(correct[1])
            stats['both_correct'] += int(correct.all())
    norm_result = {}
    for b, values in hold.items():
        direction = float(np.sign(np.mean(train[b])))
        norm_result[f'B{b}'] = {'discovery_norm_direction': direction,
            'holdout_rate': float(np.mean(direction*np.asarray(values) > 0)), 'n': len(values)}
    D.SAVE(D.OUT/'selectivity-diagnosis.json', {'schema': 'mlp_b34_selectivity_diagnosis.v1',
        'provenance': plan['provenance'], 'plan_sha256': D.HASH(D.OUT/'prereg.json'), 'source_sha256': D.HASH(__file__),
        'scope': 'Post hoc diagnostic after the random-group comparison failed; no evidence upgrades or threshold changes.',
        'baseline_by_frame': groups, 'scalar_norm_control': norm_result})
    print({'baseline_by_frame': groups, 'scalar_norm_control': norm_result}, flush=True)
    # The frozen analysis uses NumPy boolean scalars in some gate results.
    # Converting at the JSON boundary preserves every score and decision.
    analysis.SAVE = lambda path, payload: D.SAVE(path, normalize(payload))
    analysis.report(plan)


if __name__ == '__main__':
    main()
