"""最小实时前向服务（live-only）——只挂 /api/live/*，不做数据包预热、不挂普查/图谱。

用途：Electron 客户端等场景下，「空间 · 实时分析」需要一个能跑真实前向的本地进程，
但完整中心（server/server.py）启动要做数据包预热（数分钟）。本入口秒级就绪：
  .venv/Scripts/python.exe server/live_only_app.py   （AI2050_PORT 缺省 5002）

与完整中心的关系：完整中心也挂了同一 router（:5001）；本服务是「客户端连线上中心、
但实时分析要用本机权重」时的独立兜底，端口错开避免抢占完整中心。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 直接以文件运行时保证 `import server.*` 可用

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.live_analysis_service import router

app = FastAPI(title='AI2050 Live Forward', docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],          # 本机分析服务；Electron file:// 与 vite dev 都要能跨端口访问
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router)


@app.get('/api/v1/health')
def health():
    return {'status': 'ok', 'live_only': True}


if __name__ == '__main__':
    uvicorn.run(
        app,
        host=os.environ.get('AI2050_HOST', '127.0.0.1'),
        port=int(os.environ.get('AI2050_PORT', '5002')),
        log_level='warning',
    )
