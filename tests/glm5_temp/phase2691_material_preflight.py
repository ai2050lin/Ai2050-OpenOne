"""Read-only crossmodel material and old candidate orientation audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
import torch
import phase2691_crossmodel_role_confirmation as c


def main():
    plan=c.read(c.OUT/'protocol/frozen.json');reports={}
    for key,cfg in plan['models'].items():
        rows=c.read(c.OUT/key/'material/cases.json');cal=c.read(c.OUT/key/'material/calibration.json')
        assert c.sha(c.OUT/key/'material/cases.json')==cfg['material_sha256']
        assert len(rows)==cfg['cases'] and len(cal)==64
        for start in range(0,len(rows),8):
            rr=rows[start:start+8]
            assert {(r['target_index'],r['output_function']) for r in rr}=={(v,f) for v in (0,1) for f in c.FUNCTIONS}
            assert len({tuple(r[k] for k in ('family','language','unit','content_instance','form','roster_order','mention_order')) for r in rr})==1
        opens=sum(r['prompt'].rfind('<think>')>r['prompt'].rfind('</think>') for r in rows)
        reports[key]={'conditions':len(rows),'pair_groups':len(rows)//8,'open_thinking':opens,'no_model_loaded':True}
    native=c.read(c.OUT/'ds7/material/cases.json');direct=c.read(c.OUT/'ds7_answer/material/cases.json')
    assert reports['ds7']['open_thinking']==512 and reports['ds7_answer']['open_thinking']==0
    assert all(a['source_case_index']==b['source_case_index'] and a['prompt_ids'][:a['body_end_token']+1]==b['prompt_ids'][:b['body_end_token']+1] for a,b in zip(native,direct))
    previous=c.RESULT/'phase2683_crossmodel_function_atlas/qwen14/maps/global_counts.npz'
    found=[]
    with np.load(previous) as z:
        for metric in ('h','a'):
            for sign in ('positive','negative'):
                for l,j in np.argwhere(z[metric+'__all4_'+sign][:,1]==64):found.append((metric,int(l),int(j),sign))
    assert sorted(found)==sorted(c.CANDIDATES)
    assert not torch.cuda.is_initialized()
    report={'all_checks_passed':True,'preview_only':True,'formal_model_outputs':0,'model_loaded':False,'cuda_initialized':False,
        'models':reports,'DS_same_body_token_prefixes':512,'old_Q14_all_passing_addresses_exact':found,
        'orientation':'v0-v1 as2683; no sign borrowed from2687v1-v0','code_sha256':c.sha(c.TESTS/'phase2691_crossmodel_role_confirmation.py')}
    c.save(c.OUT/'analysis/material_preflight.json',report);print(report,flush=True)


if __name__=='__main__':main()
