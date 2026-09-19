"""Receipt after the main agent actually viewed all seven propagation PNGs."""
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays


def main():
    folder = OUT/'figures/propagation_v1'; index = read(folder/'index.json')
    assert len(index['figures']) == 7
    for image in index['figures']: assert sha(BASE/image['path']) == image['sha256']
    arrays = checked_arrays(index['numeric_fields'])
    assert len(arrays) == 7
    for a in arrays.values():
        assert a.shape[:3] == (2,6,20) and np.isfinite(a).all() and np.min(a) >= 0
        assert np.count_nonzero(a[:,:,0]) == 0, 'Both routes have identical current-position initial derivative'
    immutable(folder/'visual_review.json', {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'reviewer': 'Main agent, actual seven images viewed in the conversation before writing this receipt.',
        'checked': ['All12endpoint panels legible; separate native and smooth targets, own log axes.',
                    'Both family heatmaps retain signed exceptions and all18family-scale columns.',
                    'All4complete coordinate maps retain20blocks and original 2560/9728indices; zero block16 expected.',
                    'No clipped titles, axis labels or legends; no semantic significance inferred from colors.'],
        'figures': index['figures'], 'numeric_source_verified': True})
    print('FORMATION_PROPAGATION_VISUAL_REVIEW', 7, flush=True)


if __name__ == '__main__': main()
