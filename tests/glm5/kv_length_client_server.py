"""Serve the actual client and read-only Phase 2952 catalog for loopback acceptance."""

import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
from deploy.storage_policy import check_write

WORK=check_write(ROOT/'tests/glm5_temp/kv_length_client_20261117_v1',20*2**20)
WORK.mkdir(exist_ok=True)
DATA=check_write(WORK/'center')
os.environ.update(AI2050_DIST_DIR=str(DATA),AI2050_FRONTEND_DIST=str(WORK/'dist'),
                  AI2050_RESEARCH_DIR=str(ROOT/'research/kv_length/v1'),AI2050_SNAPSHOT_DISABLE='1',
                  AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',AI2050_PUBLIC_URL='http://127.0.0.1:5041',
                  TEMP=str(WORK),TMP=str(WORK))
from deploy.center_app import create_app
from deploy.research_library import library
import uvicorn

if __name__=='__main__':
    assert (WORK/'dist/index.html').exists()
    cache=check_write(DATA/'snapshots'); cache.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'distributed_data/snapshots/lang_atlas.json',check_write(cache/'lang_atlas.json'))
    scan=library().scan(ROOT/'research/kv_length/v1')
    assert not scan['errors'],scan
    uvicorn.run(create_app(),host='127.0.0.1',port=5041)
