"""Isolated real-data center for browser QA; no production state or GPU access."""
import json
import os
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
DATA=ROOT/'tests/glm5/result/mlp_tail_acceptance'
DATA.mkdir(exist_ok=True);(DATA/'empty-research').mkdir(exist_ok=True)
os.environ.update(AI2050_DIST_DIR=str(DATA/'center'),AI2050_MLP_TAIL_DIR=str(ROOT/'tests/glm5/result/mlp_tail_goodfire_v1'),
    AI2050_RESEARCH_DIR=str(DATA/'empty-research'),AI2050_SNAPSHOT_DISABLE='1',AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',
    AI2050_DIST_ADMIN_TOKEN='local-tail-acceptance-only',AI2050_PUBLIC_URL='http://127.0.0.1:5032')
from deploy.research_library import Library
from deploy.research_ingest_bundle import verified_objects,read_member
lib=Library(DATA/'center/research_library')
with tarfile.open(ROOT/'release/ai2050-research-mlp-tail-20261108.tar.gz','r:gz') as tar:
    manifest=json.loads(read_member(tar,'MANIFEST.json',2**20))
    for i,(sha,raw) in enumerate(verified_objects(tar,manifest)):
        assert lib.ingest(json.loads(raw),f'acceptance-upload:{sha}',kind='local_report')['sha256']==sha
        if i%16==0:print(f'acceptance seed {i+1}/{len(manifest["objects"])}',flush=True)
print('CPU-only center ready; real data seeded, no production changes',flush=True)
from deploy.center_app import create_app
import uvicorn
uvicorn.run(create_app(),host='127.0.0.1',port=5032)
