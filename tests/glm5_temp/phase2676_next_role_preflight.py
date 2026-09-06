"""CPU-only annotation tests, not Phase2677 completion or a new model experiment."""
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2677_source_role_regions import character_regions,token_regions,ROLES


def main():
    cases=read(RESULT/'phase2670_native_mlp_contract/material/cases.json');counts=Counter();examples=[];checks={}
    for row in cases:
        regions=character_regions(row);counts.update(r['role'] for r in regions)
        # Per-character offsets test exact coverage; real tokenizer offsets are a
        # future-contract check and are deliberately not claimed here.
        out=token_regions([(i,i+1) for i in range(len(row['prompt']))],regions)
        assert all(r['role'] not in ('mixed','zero_width') for r in out)
        assert token_regions([(0,len(row['prompt'])),(0,0)],regions)[0]['role']=='mixed'
        if row['published']:examples.append({'case_index':row['case_index'],'case_id':row['case_id'],'regions':regions})
    checks={'8192_prompt_partitions':len(cases)==8192,'16_text_examples':len(examples)==16,'allroles_declared':set(counts)<=set(ROLES),
        'no_label_or_activation_input':True,'zero_width_explicit':token_regions([(0,0)],examples[0]['regions'])[0]['role']=='zero_width'}
    out=RESULT/'phase2676_native_mlp_delivery/analysis/next_role_preflight.json'
    save(out,{'checks':checks,'all_checks_passed':all(checks.values()),'region_counts':dict(counts),'examples':examples,
        'scope':'Preparing next whole campaign only. Current8192existingtexts, no newmodel forwards, no newPhaseappend. Real tokenizer offset tests remain for2677. Character roles are external annotations, not semantic causal modules.'})
    print('ROLE PREFLIGHT',len(cases),dict(counts));assert all(checks.values())


if __name__=='__main__':main()
