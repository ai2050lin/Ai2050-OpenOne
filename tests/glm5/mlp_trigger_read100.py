"""Prospective fresh-prefix confirmation of coordinate-100 read-in, after R4 audit."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import mlp_trigger_design as D
import mlp_trigger_run as R
from tools.power_analysis import check_phase

PRIMARY=D.OUT
OUT=PRIMARY.with_name('mlp_trigger_read100_v1')
RUN='MLP-TRIGGER-READ100-v1'
SEED=202611092883

def configure():
    D.OUT=OUT;D.RUN=RUN;D.SEED=SEED;D.N_HOLD=128

def freeze():
    from transformers import AutoTokenizer
    assert not (OUT/'prereg.json').exists()
    original=D.checked();rows,rr,y,nuis,meta=R.arrays('discovery',original)
    sel=copy.deepcopy(D.READ(PRIMARY/'selection.json'))
    idx=[100];sel['coordinates']['rare']=idx
    sel['predictors']['rare']['common']=R.fit(np.column_stack([nuis[:,1],rr[:,1,idx]]),y[:,1])
    tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True)
    configure();OUT.mkdir(exist_ok=True)
    # New permutations, identifiers, entity codes and nouns. Same 16 role frames;
    # confirmation concerns unseen prefixes, not additional unseen frame families.
    new=D.material(tok,'holdout',128)
    replace={'window':'closet','doorway':'balcony','cushion':'stool','curtain':'screen','railing':'fence','counter':'bench','cupboard':'locker','blanket':'rug'}
    for row in new:
        row['id']='read100-'+row['id'];s=row['prompt'].replace('NUholdout','NVholdout').replace('NUh','NVh')
        for a,b in replace.items():s=s.replace(' '+a+' ',' '+b+' ')
        row['prompt']=s;row['tokens']=tok.encode(s,add_special_tokens=False)
    # Accept only fully matched families; the new noun vocabulary was chosen for
    # the same one-token slots. Failure stops before any new forward pass.
    for f in range(128):
        for long in (0,1):assert len({len(r['tokens']) for r in new[f*8:f*8+8] if r['long']==long})==1
    old_prompts={r['prompt'] for r in D.READ(PRIMARY/'holdout-materials.json')['rows']}
    assert all(r['prompt'] not in old_prompts for r in new)
    p=copy.deepcopy(original);p['run_id']=RUN;p['provenance']={**p['provenance'],'seed':SEED,'tool':RUN,'created_at':datetime.now().astimezone().isoformat()}
    p['discovery_reused_from']=str(PRIMARY.relative_to(D.ROOT))
    p['supplement_reason']='R4 dominates residual energy but its direct gate/up read-in is negligible. Discovery-only exact weight contributions nominate coordinate100. Fresh prefix combinations test the revised read-in candidate; prior R4 results are retained.'
    p['hypotheses']['rare']='Coordinate100 supplies the dominant G/V read-in during the coordinate4-dominated state. Removing100 and clamping the nominated source unit distinguishes common input from a serial source-unit path.'
    p['selection']='Broad 8 coordinates unchanged; rare coordinate100 selected by discovery-only mean absolute gate/up weight contributions. No candidate choice uses the new prefixes.'
    p['limits'].append('This supplement reuses discovered background syntax and 16 role frames with new nouns/permutations. It is new-prefix confirmation, not confirmation on unseen frame families or natural corpora.')
    p['power']={**p['power'],'rare_paired_change':check_phase(.25,128,.5,.005),'prediction_family_win':check_phase(.25,128,.5,.005),'causal_event_change':check_phase(.25,128,.5,.005)}
    assert all(v['pass'] for v in p['power'].values())
    p['gates']['minimum_eligible_families']=128;p['causal']['families']=128
    p['sources']['tests/glm5/mlp_trigger_read100.py']=D.HASH(Path(__file__))
    p['prior_registration_sha256']=D.HASH(PRIMARY/'prereg.json')
    p['adaptation_after_primary_data_collection']=True
    D.SAVE(OUT/'discovery-materials.json',D.READ(PRIMARY/'discovery-materials.json'))
    D.SAVE(OUT/'holdout-materials.json',{'provenance':p['provenance'],'rows':new})
    p['materials']['holdout']={'sha256':D.HASH(OUT/'holdout-materials.json'),'n':len(new),'families':128,'tokens':[min(len(r['tokens']) for r in new),max(len(r['tokens']) for r in new)]}
    D.SAVE(OUT/'prereg.json',p);D.save('selection.json',sel,p)
    D.save('candidate-origin.json',{'source_discovery_plan_sha256':D.HASH(PRIMARY/'prereg.json'),'coordinate':100,'rule':'discovery mean |gamma_j Wg_j R_j/RMS(R)| and analogous value contribution both maximal at100','no_new_prefix_output_exists':True},p)
    print({'run':RUN,'materials':p['materials'],'power':p['power']},flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');stage=parser.parse_args().stage
    if stage=='freeze':freeze();return
    configure()
    if stage=='analyze':
        import mlp_trigger_analyze as A
        A.main();return
    if stage=='verify':
        import mlp_trigger_verify as V
        V.local();return
    if stage=='publish':
        import mlp_trigger_publish as P
        P.main();return
    # R.main parses the same positional stage and uses the configured immutable plan.
    R.main()
if __name__=='__main__':main()
