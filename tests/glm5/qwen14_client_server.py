"""Local center for real-data client acceptance, with GPU tools disabled."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
from deploy.storage_policy import check_write
out=check_write(ROOT/'tests/glm5/result/qwen14_client_v2')
scratch=check_write(ROOT/'tests/glm5_temp/qwen14-client');scratch.mkdir(exist_ok=True)
os.environ.update(AI2050_DIST_DIR=str(ROOT/'distributed_data'),AI2050_QWEN14_MLP_DIR=str(out/'native'),
    AI2050_RESEARCH_DIR=str(scratch),AI2050_SNAPSHOT_DISABLE='1',AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',
    AI2050_DIST_ADMIN_TOKEN='qwen14-v2-loopback-engineering-only',AI2050_PUBLIC_URL='http://127.0.0.1:5034',
    TEMP=str(scratch),TMP=str(scratch))
from deploy.center_app import create_app
import uvicorn
print('Local CPU-only client acceptance at http://127.0.0.1:5034; GPU tools disabled',flush=True)
uvicorn.run(create_app(),host='127.0.0.1',port=5034)
