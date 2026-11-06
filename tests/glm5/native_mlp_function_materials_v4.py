"""Freeze only calibration-eligible families; no confirm/lockbox outputs seen."""
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v3 as V3
from shared.dist_protocol.mechanism_trace import digest,file_sha
from tools.power_analysis import check_phase
ROOT=V3.ROOT;OUT=ROOT/'tests/glm5/result/native_mlp_function_v4';PUBLIC=ROOT/'research/native_mlp_function/v4';RUN='NMLP-FUNCTION-20261104-v4';MODEL=V3.MODEL;SEED=V3.SEED
read=V3.read;save=V3.save;FAMILIES={k:v for k,v in V3.FAMILIES.items() if k!='enumeration'}
def main():
    OUT.mkdir(parents=True,exist_ok=True);base=read(V3.OUT/'corpus.json');pilot=read(V3.OUT/'pilot-result.json');eligible=[f for f in FAMILIES if all(c['passed'] for c in pilot['cells'] if c['family']==f)];assert len(eligible)==3
    c={**base,'run_id':RUN,'items':[r for r in base['items'] if r['family'] in eligible],'hypotheses':[r for r in base['hypotheses'] if r['family'] in eligible],
        'calibration_excluded_families':['enumeration'],'scope_note':'selected-entry conditions, universal coverage andlocation definitions; enumeration failed calibration andis excluded before heldouts'}
    pr={**read(V3.PUBLIC/'prereg.research.json'),'run_id':RUN,'corpus_sha256':digest(c),'n_inputs':len(c['items']),'hypotheses':c['hypotheses'],'criteria':copy.deepcopy(read(V3.PUBLIC/'prereg.research.json')['criteria']),
        'selection_note':'Eligibility selected on discovery behavior only. Original alpha .05/4 retained conservatively for3families. No heldout response used.',
        'dose':{'units':'fixed8 discovery-selected units; alpha0,.5,1 atfinal assistant query only; random8 strictdiscovery A_RMS/downcolumn/product ratios[.8,1.25]',
            'pairs':'first96 lockbox pairs perfamily; bothsemantic levels;288 independentbackground-pair observations per family-specific contrast, sharedbackgrounds acrossfamilies clustered in inference',
            'primary':'mean pair-both-correct loss>=.1, target-minus-random>=.05 andpositive95 lower paired-bootstrap bounds; exploratory joint operation not individual necessity',
            'power':check_phase(.1,288,p_d=.2)},
        'capability':{'n_formal':2200,'n_reference':140,'NI_margin':.02,'pool':'same previously exposed keyed monitor, new8-unit operation','power_planning':check_phase(.02,2200,p_d=.1)}}
    assert pr['dose']['power']['pass'] and pr['capability']['power_planning']['pass']
    save(OUT/'corpus.json',c);save(OUT/'discovery.json',read(V3.OUT/'discovery.json'));save(PUBLIC/'prereg.research.json',pr)
    save(OUT/'pilot-result.json',{**pilot,'run_id':RUN,'rows':[r for r in pilot['rows'] if r['family'] in eligible],'cells':[r for r in pilot['cells'] if r['family'] in eligible],
        'passed':True,'reused_identical_inputs_no_new_inference':True,'source_sha256':file_sha(V3.OUT/'pilot-result.json')})
    print({'eligible':eligible,'inputs':len(c['items'])})
if __name__=='__main__':main()
