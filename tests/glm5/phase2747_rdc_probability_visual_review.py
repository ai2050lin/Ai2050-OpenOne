"""Actual image inspection receipts, including the preserved first-layout issue."""
from rdc_formation_common import *


def main():
    old=OUT/'figures/probability_v1';folder=OUT/'figures/probability_v2'
    first,index=read(old/'index.json'),read(folder/'index.json')
    assert len(first['figures'])==len(index['figures'])==3
    for group in [first,index]:
        for entry in group['figures']:
            assert sha(BASE/entry['path'])==entry['sha256']
            for source in entry['sources']:assert sha(BASE/source['path'])==source['sha256']
    assert [r['sources'] for r in first['figures']]==[r['sources'] for r in index['figures']]
    immutable(old/'visual_review.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':False,
        'reviewer':'Main agent actually inspected all three images before the revision.',
        'problem':'Third figure left-panel legend obscured the last-row no-bias NLL point. Numerical inputs correct and unchanged.',
        'original_images_retained':True,'figures':first['figures']})
    immutable(folder/'visual_review.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'reviewer':'Main agent actually inspected all three revised images before authoring this receipt.',
        'checked':['All5matched-radius directions and both seeds visible, with explicit independent-axis scales.',
            'All8direction/heldout-split paired mapping comparisons and sign-changing exceptions visible.',
            'First-literal-digit NLL and conditional ordering clearly separate; Markdown prefix limitation explicit.',
            'Shared legend is in unused right-panel space; no points, intervals, labels or titles obscured.'],
        'unchanged_scientific_inputs':True,'figures':index['figures'],
        'boundary':'Visual verification is not evidence of semantic closure.'})
    save(OUT/'figures/current_probability_figures.json',{'timestamp':stamp(),'index':(folder/'index.json').relative_to(BASE).as_posix(),
        'sha256':sha(folder/'index.json'),'revision':'Remove redundant left-panel legend that covered a point; preserve all original images and numerical inputs.'})
    print('FORMATION_PROBABILITY_VISUAL_REVIEW',3,flush=True)


if __name__=='__main__':main()
