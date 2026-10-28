# -*- coding: utf-8 -*-
"""打包开放平台中心的部署包：release/ai2050-center-<版本>.tar.gz（含已构建的客户端与公开研究数据）。

包含：中心服务与节点程序（deploy/）、协议代码（shared/dist_protocol/）、神经元单元卡端点（server/neuron_census_service.py）、
客户端构建（frontend/dist/，需先 vite build）、公开研究数据（research/interventions/、research/neurons/：预注册、干预与解释检验结果）、
许可证与说明。不包含：研究记录（research/claude/docs 等）、模型权重、普查语料与原生全场（体积大，按 README_DEPLOY.md 单独上传）。

每次打包都递增版本号（VERSION 文件，缺省递增补丁号 x.y.Z），发布包与解压目录名都带版本号；
release/ 下已有同版本的包时拒绝覆盖。

用法：python scripts/build_center_release.py [--bump patch|minor|major|none] [--skip-frontend-check]
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
    'VERSION', 'LICENSE', 'NOTICE', 'README.md', 'CONTRIBUTING.md',
    'deploy/center_app.py', 'deploy/distributed_service.py', 'deploy/census_service.py', 'deploy/data_admin_service.py', 'deploy/platform_service.py',
    'deploy/node_agent.py', 'deploy/node_service.py', 'deploy/admin_tool.py', 'deploy/reset_password.py', 'deploy/import_center_data.py', 'deploy/snapshot_worker.py', 'deploy/datapack.py', 'deploy/dist_templates_seed.py', 'deploy/dist_runner_tm.py',
    'deploy/install_center.sh', 'deploy/upgrade_center.sh', 'deploy/ai2050-center.service', 'deploy/nginx-ai2050-center.conf',
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
    ap.add_argument('--bump', default='patch', choices=['patch', 'minor', 'major', 'none'],
                    help='打包前递增 VERSION（缺省补丁号）；none = 沿用当前版本（该版本的包已存在时拒绝）')
    args = ap.parse_args()
    vfile = ROOT / 'VERSION'
    cur = vfile.read_text(encoding='utf-8').strip()
    major, minor, patch = (int(x) for x in cur.split('.'))
    new = {'patch': (major, minor, patch + 1), 'minor': (major, minor + 1, 0), 'major': (major + 1, 0, 0),
           'none': (major, minor, patch)}[args.bump]
    PLATFORM_VERSION = '.'.join(map(str, new))
    if (Path(args.out) / f'ai2050-center-{PLATFORM_VERSION}.tar.gz').exists():
        raise SystemExit(f'release/ai2050-center-{PLATFORM_VERSION}.tar.gz 已存在：换一个版本号（--bump）')
    dist = ROOT / 'frontend' / 'dist'
    if not (dist / 'index.html').is_file():
        raise SystemExit('frontend/dist 不存在：先在 frontend/ 运行 vite build')
    if not args.skip_frontend_check:
        newest_src = max(p.stat().st_mtime for p in (ROOT / 'frontend' / 'src').rglob('*') if p.is_file())
        if (dist / 'index.html').stat().st_mtime < newest_src:
            raise SystemExit('frontend/dist 比源码旧：先重新 vite build（或 --skip-frontend-check）')
    if PLATFORM_VERSION != cur:                            # 只在递增时改写；固定 LF，避免 Windows 换行让工作区显示为已修改
        vfile.write_bytes((PLATFORM_VERSION + chr(10)).encode('utf-8'))
    import subprocess
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True,
                                    text=True, timeout=120).stdout.strip())
    except Exception:
        commit, dirty = '', None
    name = f'ai2050-center-{PLATFORM_VERSION}'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = files()
    # Windows 上 core.autocrlf=true 的检出会把文本变成 CRLF：服务器上的脚本、unit、VERSION 与预注册哈希都会坏
    crlf = [str(p.relative_to(ROOT).as_posix()) for p in fl
            if (p.name == 'VERSION' or p.suffix in ('.sh', '.service', '.conf', '.sha256', '.tsv') or p.name.endswith('.prereg.json'))
            and b'\r' in p.read_bytes()]
    if crlf:
        raise SystemExit(f'这些文件含 CR（Windows 换行），在服务器上会出错：{crlf[:8]}…  用 git -c core.autocrlf=false 检出后再打包')
    manifest = {'name': name, 'version': PLATFORM_VERSION, 'previous_version': cur, 'git_commit': commit,
                'git_dirty': dirty, 'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
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
    (out / f'{name}.tar.gz.sha256').write_bytes(f'{digest}  {tar_path.name}\n'.encode('utf-8'))
    (out / 'upgrade_center.sh').write_bytes((ROOT / 'deploy' / 'upgrade_center.sh').read_bytes())   # 与发布包一起上传，升级时直接运行
    print(f'版本 {cur} → {PLATFORM_VERSION}')
    print(f'{tar_path}  {tar_path.stat().st_size / 2**20:.1f} MB  {len(fl)} 个文件  sha256 {digest[:16]}')
    print(f'升级：把 {tar_path.name}、{tar_path.name}.sha256 与 upgrade_center.sh 传到服务器同一目录，运行 sudo bash upgrade_center.sh')
    print('提醒：VERSION 已改，请连同代码一起提交，保证仓库版本与发布包一致')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
