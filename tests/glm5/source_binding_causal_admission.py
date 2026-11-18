"""Apply the predeclared invalid-control stop before any held-out intervention."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def prepare():
    old=D.ROOT/'tests/glm5/source_binding_causal.py';new=D.ROOT/'tests/glm5/source_binding_causal_v2.py';assert not new.exists()
    text=old.read_text(encoding='utf-8')
    needle="    verify_code();proof=D.admission(model,split);root=D.OUT/model;c=D.read(root/'candidates.json')\n"
    addition=needle+'''    amendment=D.read(D.OUT/'causal-admission-v2.json')
    for path,sha in amendment['code_sha256'].items():assert D.sha(D.ROOT/path)==sha
    skipped=[]
    if not all(x['matched_control_valid'] for x in c['mlps']):
        skipped=['mlp_selected','mlp_matched']
        ARMS[:]=[arm for arm in ARMS if arm not in skipped]
'''
    assert text.count(needle)==1;text=text.replace(needle,addition)
    needle="'arms':ARMS,'candidates_sha256'";assert text.count(needle)==1;text=text.replace(needle,"'arms':ARMS,'skipped_invalid_controls':skipped,'candidates_sha256'")
    needle="'mlp_controls_valid':all(x['matched_control_valid'] for x in c['mlps'])";assert text.count(needle)==1
    text=text.replace(needle,"'mlp_controls_valid':all(x['matched_control_valid'] for x in c['mlps']),'skipped_invalid_controls':skipped")
    new.write_text(text,encoding='utf-8')
    compile(text,str(new),'exec')
    D.save('causal-admission-v2.json',{'original_internal_freeze_sha256':D.sha(D.OUT/'internal-code-freeze.json'),
        'code_sha256':{str(p.relative_to(D.ROOT)):D.sha(p) for p in [new,Path(__file__)]},
        'reason':'Discovery matching failed the already-frozen10% RMS tolerance. Stop MLP causal arms before heldout intervention; preserve component prediction and original candidates. No rematching or threshold change.',
        'observed_qwen_discovery_control_ratios':[x['matched_rms_ratio'] for x in D.read(D.OUT/'qwen14/candidates.json')['mlps']],
        'applicability':'same rule for each model; no MLP group intervention unless both selected groups controls are valid'})

if __name__=='__main__':prepare()
