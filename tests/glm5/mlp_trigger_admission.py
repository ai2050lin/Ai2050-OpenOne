"""Explicitly veto language-mechanism promotion when a frozen behavior cell fails."""
import argparse
import time
from collections import defaultdict
import numpy as np
import mlp_trigger_design as D

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--read100',action='store_true');parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.read100:
        import mlp_trigger_read100 as S
        S.configure()
    p=D.checked();started=time.monotonic()
    while len(list(D.OUT.glob('holdout-[0-9]*.json')))<D.N_HOLD:
        assert args.wait and time.monotonic()-started<600
        print({'awaiting_complete_holdout':len(list(D.OUT.glob('holdout-[0-9]*.json'))),'required':D.N_HOLD},flush=True);time.sleep(15)
    rows=D.materials(p,'holdout');cells=defaultdict(list)
    for f in range(D.N_HOLD):
        m=D.READ(D.OUT/f'holdout-{f:04d}.json')
        for row,obs in zip(rows[f*8:f*8+8],m['rows']):cells[row['frame']].append(np.argmax(obs['scores'][:2])==row['gold'])
    scores={str(k):{'correct':int(sum(v)),'n':len(v),'accuracy':float(np.mean(v)),'passed':bool(np.mean(v)>=.85)} for k,v in cells.items()}
    failed=[k for k,v in scores.items() if not v['passed']]
    D.save('semantic-admission.json',{'schema':'mlp_behavior_admission.v1','per_frame':scores,'failed_frames':failed,
       'before_first_intervention_output':not list(D.OUT.glob('causal-[0-9]*.json')),
       'language_mechanism_promotion_allowed':False,
       'action':'Failed frames are not positive evidence for role-binding ability. Retain all numerical response/intervention observations and wrong-answer counterexamples. No semantic label or behavioral mediation confirmation is granted.',
       'task_scope':'Current primary hypotheses concern native numerical triggering and common input, not a successful attribute-binding language mechanism.'},p)
    print({'failed_frames':failed,'no_language_mechanism_promotion':True},flush=True)
if __name__=='__main__':main()
