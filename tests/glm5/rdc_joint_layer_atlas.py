"""Native-order all-layer moments and held-out common-update controls."""
from collections import Counter
from rdc_joint_common import *


def layer_atlas(store):
    out = BASE / 'layer_atlas'
    snapshot(Path(__file__))
    groups = [(s, l) for s in ('train', 'validation', 'test') for l in ('en', 'zh')]
    # First, second and four ordinary anchor positions; ordinary samples have equal weight.
    sums = np.zeros((6, 3, 37, 2560), float)
    squares = np.zeros_like(sums)
    count = np.zeros((6, 3), int)
    for r in store.material:
        gi = groups.index((r['split'], r['language']))
        h = unbits(store[r]['layers']).astype(float)
        for role, positions in enumerate(([0], [1], [2, 3, 4, 5])):
            a = h[:, positions]
            sums[gi, role] += a.sum(1)
            squares[gi, role] += (a*a).sum(1)
            count[gi, role] += len(positions)
    mean = sums/count[:, :, None, None]
    second = squares/count[:, :, None, None]
    train_count = count[:2].sum(0)
    mu = sums[:2].sum(0)/train_count[:, None, None]
    sd = np.sqrt(np.maximum(squares[:2].sum(0)/train_count[:, None, None]-mu**2, 0))
    common = np.diff(mu, axis=1).astype(np.float32)
    error = np.zeros((2, 3, 36, 2560), float)
    identity = np.zeros_like(error)
    evalcount = np.zeros((2, 3), int)
    per_source = []
    for r in store.material:
        if r['split'] == 'train':
            continue
        si = ('validation', 'test').index(r['split'])
        h = unbits(store[r]['layers']).astype(float)
        rr = {k: r[k] for k in ('sample_id', 'source_group', 'split', 'language', 'genre')}
        rr['roles'] = {}
        for role, positions in enumerate(([0], [1], [2, 3, 4, 5])):
            delta = np.diff(h[:, positions], axis=0)
            ee = (delta-common[role, :, None])**2
            error[si, role] += ee.sum(1)
            identity[si, role] += (delta**2).sum(1)
            evalcount[si, role] += len(positions)
            rr['roles'][('first', 'second', 'ordinary')[role]] = {
                'common_update_layer_MSE': ee.mean((1, 2)).tolist(),
                'identity_layer_MSE': (delta**2).mean((1, 2)).tolist()}
        per_source.append(rr)
    error /= evalcount[:, :, None, None]
    identity /= evalcount[:, :, None, None]
    energy = (squares[:2].sum(0)/train_count[:, None, None]).mean(-1)
    # Locate numeric amplification on TRAIN only, without calling it attention sink or semantics.
    log_growth = np.diff(np.log(np.maximum(energy[0], 1e-20)))
    ordinary_growth = np.diff(np.log(np.maximum(energy[2], 1e-20)))
    selected = int(np.argmax(log_growth-ordinary_growth))
    guard(13*1024**2)
    npz(out/'full_coordinate_moments.npz', mean=mean.astype(np.float32), second_moment=second.astype(np.float32),
        counts=count, train_mean=mu.astype(np.float32), train_std=sd.astype(np.float32),
        train_common_increment=common, common_update_MSE=error.astype(np.float32), identity_MSE=identity.astype(np.float32))
    compressed_json(out/'source_layer_errors.json.gz', per_source)
    report = {'timestamp': stamp(), 'groups': groups, 'roles': ['first', 'second', 'ordinary'],
        'coordinates': 2560, 'boundaries': 37, 'raw_native_order': True,
        'standardized_view_recipe': '(H-train_mean[role,layer])/max(train_std[role,layer],1e-6); no per-split fitting or coordinate sorting.',
        'aggregation': 'Mean and second moment over all sources/positions in stratum; one-step errors averaged per source for uncertainty.',
        'training_energy_by_role_layer': energy.tolist(),
        'training_selected_amplification_block_zero_index': selected,
        'selection': 'argmax_l log(E_first[l+1]/E_first[l])-log(E_ordinary[l+1]/E_ordinary[l]); training only, not semantic identification.',
        'heldout_common_increment_MSE_by_role_layer': error.mean(-1).tolist(),
        'heldout_identity_MSE_by_role_layer': identity.mean(-1).tolist(),
        'limits': 'Only six positions have all-layer coverage; H12/H23/H36 cover all tokens. Same coordinate across layers is an index, not assumed function. Common update is an empirical baseline, not transported semantic difference.'}
    save(out/'result.json', report)
    print('JOINT_LAYER_ATLAS', 'selected block', selected, 'first H12/H23 energy', energy[0, [12,23]].tolist(), flush=True)
    return report
