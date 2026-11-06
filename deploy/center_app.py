# -*- coding: utf-8 -*-
"""AI2050 开放研究平台 · 中心服务独立入口（单进程：API + 客户端静态页）。

    python deploy/center_app.py --host 0.0.0.0 --port 5010

挂载：distributed_service（模板、节点、结果、谱图、账号）、census_service（神经元普查分片）、data_admin_service（管理员数据管理，只读）、platform_service（公开配置、
进展快照、贡献榜、参与者一键包）、server/neuron_census_service（神经元单元卡）；frontend/dist 存在时同源托管客户端。

环境变量（生产部署见 deploy/README_DEPLOY.md）：
  AI2050_DIST_DIR            中心数据目录（SQLite、结果对象、普查作业）
  AI2050_NATIVE_DIR          普查语料与合并后的普查结果
  AI2050_DIST_ADMIN_TOKEN    中心管理令牌（生产必填；未设置时只有 AI2050_DEV=1 才允许用开发默认值）
  AI2050_PUBLIC_URL          对外地址（写进参与者一键包）
  AI2050_REGISTRATION        open / closed
  AI2050_FRONTEND_DIST       客户端构建目录（缺省 <repo>/frontend/dist）
  AI2050_LOCAL_TOOLS         1 = 本机中心，开放读取本机模型权重与下载模型的工具端点（公开中心不要设置）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 只适合在研究者自己机器上用的端点：读取本机模型权重、下载模型
LOCAL_TOOL_PREFIXES = ('/api/live/', '/api/models/', '/api/parameters/', '/api/traces/')
# 会真的在本机启动进程、加载权重、占显存的端点。中心（公开部署）一律不执行：
# 自动测试要在有 GPU 的机器上跑，中心只负责保存任务记录、分发命令与回收结果。
# 这些路径不给「管理员令牌」这个后门——给管理员留后门，等于给「中心偷偷跑 GPU」留了路。
GPU_ONLY_PREFIXES = ('/api/autotest/',)
GPU_ONLY_REASON = ('中心不做 GPU 计算：自动测试在本机执行（AI2050_LOCAL_TOOLS=1 的本机中心，'
                   '或节点从中心领取任务后在本机跑）。中心只保存任务记录、转发命令、回收结果。')
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    import census_service
    import data_admin_service
    import distributed_service
    import platform_service
    from server import neuron_census_service

    app = FastAPI(title='AI2050 Open LLM Reverse-Engineering Platform', version=platform_service.PLATFORM_VERSION)
    origins = [o.strip() for o in os.environ.get('AI2050_CORS_ORIGINS', '*').split(',') if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['*'], allow_headers=['*'])
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=2048)      # 谱图、3D 神经元空间等大 JSON
    local_tools = os.environ.get('AI2050_LOCAL_TOOLS') == '1'

    @app.middleware('http')
    async def guard_local_tools(request, call_next):
        """两级守卫，区别是「能不能靠令牌放行」：

        GPU_ONLY_PREFIXES  一律拒绝（公开中心没有 GPU，也不该替别人跑）：只回 403 与一句去哪里跑。
        LOCAL_TOOL_PREFIXES 公开中心要管理令牌——这些是读目录、触发下载这类本机工具，
                           研发机上的本机中心才开放；公开中心上给管理员留一条受控通道。
        """
        path = request.url.path
        if request.method != 'OPTIONS':
            if not local_tools and path.startswith(GPU_ONLY_PREFIXES):
                return JSONResponse({'detail': GPU_ONLY_REASON, 'local_only': True}, status_code=403)
            archived_index = request.method in ('GET', 'HEAD') and path == '/api/traces/research-space'
            if not local_tools and path.startswith(LOCAL_TOOL_PREFIXES) and not archived_index:
                try:
                    distributed_service._auth_admin(request.headers.get('x-admin-token', ''))
                except Exception:
                    return JSONResponse({'detail': '本机工具：公开中心未开放（需要管理令牌）'}, status_code=403)
        return await call_next(request)

    app.include_router(distributed_service.router)
    app.include_router(census_service.router)
    app.include_router(data_admin_service.router)
    app.include_router(platform_service.router)
    app.include_router(platform_service.auth_router)
    app.include_router(platform_service.lang_router)
    app.include_router(neuron_census_service.router)
    from deploy.research_library import router as library_router
    from deploy.parameter_service import router as parameter_router
    from server.rule_ledger_service import overview as rule_overview
    if local_tools and (REPO / 'server' / 'mechanism_trace_service.py').is_file():
        from server.mechanism_trace_service import router as mechanism_trace_router
        app.include_router(mechanism_trace_router)
    else:
        # The public index depends on the archive only. Native file and model tools stay local.
        from server.research_space_service import router as research_space_router
        app.include_router(research_space_router, prefix='/api/traces')
    app.include_router(library_router)
    app.include_router(parameter_router)
    app.add_api_route('/api/rules/overview', rule_overview, methods=['GET'])
    from deploy.node_control import router as node_control_router
    app.include_router(node_control_router)

    data_dir = Path(distributed_service._DATA_DIR).resolve()
    in_release = REPO.resolve() in data_dir.parents          # 数据目录在带版本号的程序目录里：换新版本目录时会「丢」数据
    if in_release:
        print(f'[center] 警告：数据目录 {data_dir} 位于程序目录 {REPO} 之内。升级到新版本目录前，'
              '请设置 AI2050_DIST_DIR 指向程序目录之外（如 /data/ai2050-data/center），并把现有数据移过去。', flush=True)

    @app.get('/api/health')
    def health() -> dict:
        return {'service': 'ai2050-center', 'version': platform_service.PLATFORM_VERSION,
                'protocol': distributed_service.ROUTER_VERSION, 'data_dir_in_release': in_release}

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
        platform_service.start_refresher()          # 后台计算子进程（快照、规律分析、普查合并）

    @app.on_event('shutdown')
    def _stop() -> None:
        platform_service.stop_refresher()

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
