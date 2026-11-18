"""Loopback client review of real archives; no GPU tools enabled."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
from deploy.storage_policy import check_write
base=check_write(ROOT/'tests/glm5/result/goodfire_middle_20261113_v2')
scratch=check_write(ROOT/'tests/glm5_temp/goodfire_middle_20261113_v2/server');scratch.mkdir(parents=True,exist_ok=True)
os.environ.update(AI2050_DIST_DIR=str(ROOT/'distributed_data'),AI2050_NATIVE_MLP_ROOT=str(base),
    AI2050_QWEN14_MLP_DIR=str(ROOT/'tests/glm5/result/qwen14_client_v2/native'),
    AI2050_RESEARCH_DIR=str(scratch),AI2050_SNAPSHOT_DISABLE='1',AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',
    AI2050_DIST_ADMIN_TOKEN='native-middle-loopback-engineering-only',AI2050_PUBLIC_URL='http://127.0.0.1:5035',
    TEMP=str(scratch),TMP=str(scratch))
from deploy.center_app import create_app
import uvicorn
print('Real atlas loopback review on http://127.0.0.1:5035; GPU tools disabled',flush=True)
uvicorn.run(create_app(),host='127.0.0.1',port=5035)
