"""Read-only independent pilot artifact audit; does not open confirmation materials."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch,transformers
import source_binding_design as D
from deploy.storage_policy import check_write
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
pre=D.read(OUT/'prereg.json');res=D.read(OUT/'pilot-result.json');rows=D.read(OUT/'pilot-materials.json')
assert D.sha(OUT/'prereg.json')==res['prereg_sha256']
assert D.sha(OUT/'pilot-native.npz')==res['native_sha256']
assert D.sha(OUT/'pilot-materials.json')==pre['materials']['pilot']['sha256']
for name,sha in pre['code'].items():assert D.sha(Path(__file__).parent/name)==sha
for row,score in zip(rows,res['scores']):
    assert row['id']==score['id']
    assert row['target_record']==2*row['b']+(row['r0'],row['r1'])[row['b']]
    vals=score['scores'];gold=row['target_record']
    assert score['gold_margin']==vals[gold]-max(v for j,v in enumerate(vals) if j!=gold)
data=np.load(OUT/'pilot-native.npz');checks={}
for b in (20,21,22):
    g=torch.from_numpy(data[f'g{b}'].copy()).view(torch.bfloat16)
    u=torch.from_numpy(data[f'u{b}'].copy()).view(torch.bfloat16)
    a=torch.from_numpy(data[f'a{b}'].copy()).view(torch.bfloat16)
    rebuilt=torch.nn.functional.silu(g)*u
    checks[str(b)]={'cpu_bf16_max_error':float((rebuilt.float()-a.float()).abs().max()),'bitwise_fraction':float((rebuilt==a).float().mean())}
index={(r['family'],r['r0'],r['r1'],r['b'],r['late']):i for i,r in enumerate(rows)}
prefix={}
for k in range(41):
    h=data[f'hs{k:02}'];count=0
    for f in range(16):
        for a in (0,1):
            for c in (0,1):count+=int(np.array_equal(h[index[f,a,c,0,1],0],h[index[f,a,c,1,1],0]))
    prefix[str(k)]=count
report={'hashes_valid':True,'formula_audit':checks,'prefix_equal_pairs_by_boundary':prefix,'strict_correct':sum(s['gold_margin']>0 for s in res['scores']),'top1_correct':sum(s['top1_correct'] for s in res['scores']),'n':len(rows),'behavior_pass':res['behavior_pass'],'technical_pass':res['technical_pass'],'torch':torch.__version__,'transformers':transformers.__version__,'native_bytes':(OUT/'pilot-native.npz').stat().st_size,'prereg_sha256':res['prereg_sha256'],'provenance':pre['provenance'],'limitations':['pilot only','GUA check CPU BF16 vs GPU, not mechanism extraction','no confirmation exposure or causal experiment']}
check_write(OUT/'audit.json',2**20,operation='CLT audit')
assert not (OUT/'audit.json').exists()
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
