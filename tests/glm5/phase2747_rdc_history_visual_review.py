"""Write exact image receipts only after the main agent has viewed those images."""
import argparse
from rdc_formation_common import *


def main(mode):
    if mode=='figures':
        folder=OUT/'figures/history_v1';index=read(folder/'index.json');assert len(index['figures'])==2
        images=[]
        for r in index['figures']:
            assert sha(BASE/r['path'])==r['sha256']
            for s in r['sources']:assert sha(BASE/s['path'])==s['sha256']
            images.append({'path':r['path'],'sha256':r['sha256']})
        checks=['All9native/trained runs, bothcontrolled denominators, naturalEOS/cap and B1/B8shape counts are legible.',
            'All6program paths retain original primary counts; paired intervals and unchanged sequences are visible.',
            'Supplemental unblinded terminal review is explicitly separate from the original frozen parser; no marks are obscured by legends.']
    else:
        pointer='current_program_regression.json' if mode=='program' else 'current_history_regression.json'
        current=read(OUT/'client'/pointer);result=read(ROOT/current['result']);folder=ROOT/current['image_directory']
        assert result['all_passed'] and sha(ROOT/current['result'])==current['result_sha256']
        assert result['program_only']==(mode=='program')
        assert len(result['images']) in ([4,5] if mode=='program' else [6,7,8,9])
        images=[{'path':str((folder/name).relative_to(BASE)),'sha256':sha(folder/name)} for name in result['images']]
        checks=['Every actual rendered native coordinate remains in original index order; row/column boundaries and scale are visible.',
            'Program ownstep postnorm and initial actual readout are distinct axes; limits and generated output are not presented as identical numerical states.',
            'Mobile program controls and complete branch summaries wrap within the panel; no horizontal overflow or clipped labels.',
            'These are actually viewed captured states; program-only checks do not stand in for pending larger-model results.']
        if 'program_terminal_review.png' in result['images']:
            checks.append('Separate post-outcome unblinded terminal review and exact requested-variable quote are visible without rewriting the frozen score.')
        if 'native_controlled_scoring.png' in result['images']:
            checks.append('Native language frozen answer, target, strict format and correct-plus-stop fields remain separate and legible; an unparsed answer is not labelled a semantic error.')
        if 'native_language_terminal_review.png' in result['images']:
            checks.append('GLM supplemental terminal conclusion, exact quote and unblinded/non-chain-grading caveat are visible alongside the unchanged primary score.')
    immutable(folder/'visual_review.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'mode':mode,'images':images,'reviewer':'Main research agent, after actually viewing every listed image before executing this receipt writer.',
        'checks':checks,'scope':'Actual presentation inspection, not new science, an independent reviewer or semantic closure.'})
    if mode=='figures':
        save(OUT/'figures/current_history_figures.json',{'timestamp':stamp(),'index':(folder/'index.json').relative_to(BASE).as_posix(),
            'sha256':sha(folder/'index.json'),'review':'Two complete-own-history summary figures actually inspected; exact receipt in their directory.'})
    print('FORMATION_HISTORY_VISUAL_REVIEW',mode,len(images),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['program','history','figures']);main(parser.parse_args().mode)
