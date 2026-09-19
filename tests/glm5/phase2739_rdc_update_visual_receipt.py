"""Record a completed human-readable PNG review, not automatic visual grading."""
import argparse
from rdc_update_common import *


def main(acknowledged=False):
    assert acknowledged,'Run only after the main agent has inspected every indexed final PNG and all listed browser captures.'
    index=read(BASE/'figures/index.json');assert index['final']
    assert read(BASE/'client/browser_final.json')['all_passed']
    scientific=[]
    for r in index['figures']:
        digest=sha(BASE/'figures'/r['path']);assert digest==r['sha256']
        scientific.append({'file':r['path'],'sha256':digest,'review':'Rendered PNG visually inspected: title/axes/colorbar/legend legible and not cropped; numerical and inferential validity is audited separately.'})
    names=['headless_desktop.png','headless_full_coordinate_field.png','headless_history.png','headless_native_path.png','headless_manual_terminal.png','headless_mobile.png']
    screenshots=[{'file':name,'sha256':sha(BASE/'client'/name),'review':'Actual isolated-headless application screenshot visually inspected; not the user browser.'} for name in names]
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
      'reviewer':'Main research agent, after individual image inspection; this script only records acknowledged review and current hashes.',
      'scientific_figures':scientific,'browser_screenshots':screenshots,
      'synthetic_state_figures_explicitly_labeled':True,'full_coordinates_preserved_in_data':True,
      'color_scale_scope':'Raw/asinh/RMS and coordinate/MLP-unit/token-position axes distinguished; display resizing is not feature selection. Missing identity controls are not drawn as zero estimates; undefined unit correlations are gray.',
      'live_user_browser_claimed_inspected':False,'CUA_failure_receipt_sha256':sha(BASE/'client/cua_initialization_failure.json'),
      'scope':'Layout and labeling review, not a new scientific experiment or confirmation of semantic mechanism.'}
    path=BASE/'client/visual_review.json'
    if path.exists():save(BASE/'client/visual_review_history'/f'{time.time_ns()}.json',read(path))
    save(path,result);print('UPDATE_VISUAL_REVIEW_RECORDED',len(scientific),len(screenshots),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--acknowledge-inspected',action='store_true');main(p.parse_args().acknowledge_inspected)
