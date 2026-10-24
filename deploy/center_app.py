# -*- coding: utf-8 -*-
"""AI2050 开放研究平台 · 中心服务独立入口（单进程：API + 客户端静态页）。

    python deploy/center_app.py --host 0.0.0.0 --port 5010

挂载：distributed_service（模板、节点、结果、谱图、账号）、census_service（神经元普查分片）、platform_service（公开配置、
进展快照、贡献榜、参与者一键包）、server/neuron_census_service（神经元单元卡）；frontend/dist 存在时同源托管客户端。

环境变量（生产部署见 deploy/README_DEPLOY.md）：
  AI2050_DIST_DIR            中心数据目录（SQLite、结果对象、普查作业）
  AI2050_NATIVE_DIR          普查语料与合并后的普查结果
  AI2050_DIST_ADMIN_TOKEN    中心管理令牌（生产必填；未设置时只有 AI2050_DEV=1 才允许用开发默认值）
  AI2050_PUBLIC_URL          对外地址（写进参与者一键包）
  AI2050_REGISTRATION        open / closed
  AI2050_FRONTEND_DIST       客户端构建目录（缺省 <repo>/frontend/dist）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    import census_service
    import distributed_service
    import platform_service
    from server import neuron_census_service

    app = FastAPI(title='AI2050 Open LLM Reverse-Engineering Platform', version=platform_service.PLATFORM_VERSION)
    origins = [o.strip() for o in os.environ.get('AI2050_CORS_ORIGINS', '*').split(',') if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['*'], allow_headers=['*'])
    app.include_router(distributed_service.router)
    app.include_router(census_service.router)
    app.include_router(platform_service.router)
    app.include_router(platform_service.auth_router)
    app.include_router(neuron_census_service.router)

    @app.get('/api/health')
    def health() -> dict:
        return {'service': 'ai2050-center', 'version': platform_service.PLATFORM_VERSION,
                'protocol': distributed_service.ROUTER_VERSION}

    dist = Path(os.environ.get('AI2050_FRONTEND_DIST', str(REPO / 'frontend' / 'dist'))).resolve()
    if (dist / 'index.html').is_file():
        @app.get('/{path:path}', include_in_schema=False)
        def spa(path: str):
            if path.startswith('api/'):
                return JSONResponse({'detail': 'Not Found'}, status_code=404)
            f = (dist / path).resolve()
            if path and f.is_file() and f.is_relative_to(dist):
                return FileResponse(f)
            return FileResponse(dist / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.on_event('startup')
    def _warm() -> None:
        platform_service.start_refresher()

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description='AI2050 center (API + client)')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=int(os.environ.get('AI2050_DIST_PORT', '5010')))
    ap.add_argument('--workers', type=int, default=1, help='保持 1：SQLite 写锁与快照缓存都在进程内')
    args = ap.parse_args()
    if not os.environ.get('AI2050_DIST_ADMIN_TOKEN') and os.environ.get('AI2050_DEV') != '1':
        raise SystemExit('生产部署必须设置 AI2050_DIST_ADMIN_TOKEN（开发环境可设 AI2050_DEV=1 使用默认令牌）')
    os.environ.setdefault('AI2050_AUTOTEST_CENTER', f'http://127.0.0.1:{args.port}')
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips='*')


if __name__ == '__main__':
    main()
