# -*- coding: utf-8 -*-
"""打包开放平台中心的部署包：release/ai2050-center-<版本>.tar.gz（含已构建的客户端与公开研究数据）。

包含：中心服务与节点程序（deploy/）、协议代码（shared/dist_protocol/）、神经元单元卡端点（server/neuron_census_service.py）、
客户端构建（frontend/dist/，需先 vite build）、公开研究数据（research/interventions/、research/neurons/：预注册、干预与解释检验结果）、
许可证与说明。不包含：研究记录（research/claude/docs 等）、模型权重、普查语料与原生全场（体积大，按 README_DEPLOY.md 单独上传）。

用法：python scripts/build_center_release.py [--skip-frontend-check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))

INCLUDE = [
    'LICENSE', 'NOTICE', 'README.md', 'CONTRIBUTING.md',
    'deploy/center_app.py', 'deploy/distributed_service.py', 'deploy/census_service.py', 'deploy/platform_service.py',
    'deploy/node_agent.py', 'deploy/admin_tool.py', 'deploy/dist_templates_seed.py', 'deploy/dist_runner_tm.py',
    'deploy/install_center.sh', 'deploy/ai2050-center.service', 'deploy/nginx-ai2050-center.conf',
    'deploy/requirements-center.txt', 'deploy/platform_config.example.json', 'deploy/README_DEPLOY.md',
    'server/__init__.py', 'server/neuron_census_service.py',
    'scripts/run_language_family.py', 'scripts/prepare_census_corpus.py',
]
INCLUDE_DIRS = ['shared/dist_protocol', 'research/interventions', 'research/neurons', 'frontend/dist']
EXCLUDE_PARTS = {'__pycache__', '.pytest_cache'}


def files() -> list:
    out = [ROOT / p for p in INCLUDE]
    for d in INCLUDE_DIRS:
        out += [p for p in sorted((ROOT / d).rglob('*')) if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDE_PARTS)]
    missing = [str(p.relative_to(ROOT)) for p in out if not p.is_file()]
    if missing:
        raise SystemExit(f'缺少文件：{missing}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'release'))
    ap.add_argument('--skip-frontend-check', action='store_true')
    args = ap.parse_args()
    from platform_service import PLATFORM_VERSION
    dist = ROOT / 'frontend' / 'dist'
    if not (dist / 'index.html').is_file():
        raise SystemExit('frontend/dist 不存在：先在 frontend/ 运行 vite build')
    if not args.skip_frontend_check:
        newest_src = max(p.stat().st_mtime for p in (ROOT / 'frontend' / 'src').rglob('*') if p.is_file())
        if (dist / 'index.html').stat().st_mtime < newest_src:
            raise SystemExit('frontend/dist 比源码旧：先重新 vite build（或 --skip-frontend-check）')
    name = f'ai2050-center-{PLATFORM_VERSION}'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = files()
    manifest = {'name': name, 'version': PLATFORM_VERSION, 'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'files': {str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest() for p in fl}}
    tar_path = out / f'{name}.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        for p in fl:
            info = tar.gettarinfo(str(p), arcname=f'{name}/{p.relative_to(ROOT).as_posix()}')
            info.mode = 0o755 if p.suffix == '.sh' else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = 'root'
            with p.open('rb') as fh:
                tar.addfile(info, fh)
        data = json.dumps(manifest, ensure_ascii=False, indent=1).encode('utf-8')
        info = tarfile.TarInfo(f'{name}/RELEASE_MANIFEST.json')
        info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
        import io
        tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / f'{name}.tar.gz.sha256').write_text(f'{digest}  {tar_path.name}\n', encoding='utf-8')
    print(f'{tar_path}  {tar_path.stat().st_size / 2**20:.1f} MB  {len(fl)} 个文件  sha256 {digest[:16]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
