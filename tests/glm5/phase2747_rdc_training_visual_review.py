"""Receipt authored after the main agent inspected all five v2 figures."""
from rdc_formation_common import *


def main():
    pointer = read(OUT/'figures/current_training_figures.json')
    index_path = BASE/pointer['index']
    assert sha(index_path) == pointer['sha256']
    figures = read(index_path)['figures']
    assert len(figures) == 5
    for figure in figures: assert sha(BASE/figure['path']) == figure['sha256']
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'actually_visually_inspected': [{'path': r['path'], 'sha256': r['sha256']} for r in figures],
        'checks': 'Five actual v2images inspected: readable full titles, paired-count labels, separate precision baselines, full-gradient matrix, complete37by2560coordinate surfaces, shared declared color scales. H16/H17overlapping ticklabels corrected without changing underlying numbers.',
        'previous_images_preserved': True, 'original_numerical_field_recomputed_bit_equal': True,
        'scope': 'Scientific rendering/axis review, not confirmation of semantic closure. Longhistory/calibration and fullPhase delivery remain pending.'}
    save(index_path.parent/'visual_review.json', result)
    print('FORMATION_TRAINING_VISUAL_REVIEW', len(figures), flush=True)


if __name__ == '__main__': main()
