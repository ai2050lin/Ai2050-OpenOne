"""Read real registered artifacts, CPU-only local browser acceptance."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
OUT=(ROOT/'tests/glm5/result/qwen14_mlp_atlas_v1').resolve()
EMPTY=ROOT/'tests/glm5_temp/qwen14-empty-research';EMPTY.mkdir(exist_ok=True)
os.environ.update(AI2050_DIST_DIR=str(ROOT/'distributed_data'),AI2050_QWEN14_MLP_DIR=str(OUT),
    AI2050_RESEARCH_DIR=str(EMPTY),AI2050_SNAPSHOT_DISABLE='1',AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',
    AI2050_DIST_ADMIN_TOKEN='qwen14-loopback-acceptance-only',AI2050_PUBLIC_URL='http://127.0.0.1:5033')
from deploy.center_app import create_app
import uvicorn
print('Local CPU-only server; existing research archive; no remote deployment or GPU load',flush=True)
uvicorn.run(create_app(),host='127.0.0.1',port=5033)
