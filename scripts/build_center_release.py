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
import io
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
from storage_policy import StorageBlocked, check_write, pack_target, staging_file, stop_with_notice

INCLUDE = [
    'deploy/storage_policy.py', 'deploy/storage_policy.json',
    'deploy/README_QWEN14_MLP.md',
    'deploy/README_NATIVE_MIDDLE.md', 'deploy/native_mlp_pack.py',
    'deploy/research_library.py', 'deploy/mainline_state_service.py', 'deploy/README_MAINLINE_SYNC.md', 'deploy/node_control.py', 'deploy/parameter_service.py', 'deploy/parameter_reader.py',
    'design/platform_spec_v1.md',
    'VERSION', 'LICENSE', 'NOTICE', 'README.md', '.github/CONTRIBUTING.md',
    'deploy/center_app.py', 'deploy/lang_map_service.py', 'deploy/distributed_service.py', 'deploy/census_service.py', 'deploy/data_admin_service.py', 'deploy/platform_service.py',
    'deploy/node_agent.py', 'deploy/node_service.py', 'deploy/admin_tool.py', 'deploy/reset_password.py', 'deploy/import_center_data.py', 'deploy/snapshot_worker.py', 'deploy/datapack.py', 'deploy/dist_templates_seed.py', 'deploy/dist_runner_tm.py',
    'deploy/install_center.sh', 'deploy/upgrade_center.sh', 'deploy/ai2050-center.service', 'deploy/nginx-ai2050-center.conf',
    'deploy/recover_center_lang_map.sh', 'deploy/README_CENTER_LANG_MAP_RECOVERY.md',
    'ai2050_research_os/registry/contracts.json', 'ai2050_research_os/registry/puzzles.json',
    'ai2050_research_os/registry/campaigns.json', 'ai2050_research_os/registry/language_graph_targets.json',
    'deploy/requirements-center.txt', 'deploy/platform_config.example.json', 'deploy/README_DEPLOY.md', 'deploy/README_RESEARCH_UPLOAD.md',
    'deploy/research_pack_stream.py', 'deploy/research_ingest_bundle.py', 'deploy/mlp_tail_pack.py', 'deploy/qwen14_mlp_pack.py', 'deploy/README_MLP_TAIL.md',
    'server/__init__.py', 'server/neuron_census_service.py', 'server/rule_ledger_service.py', 'data/rules/ledger.json',
    'server/research_space_service.py', 'server/trace_catalog.py', 'server/mainline_sample_service.py', 'server/mlp_tail_service.py', 'server/qwen14_mlp_service.py', 'server/native_mlp_service.py',
    'data/lang_bench/meta.json', 'data/lang_bench/results.json',
    # 数据集清单靠这份摘要：题量、科目、三类分布、许可、源地址都在 by_dataset 里。
    # 题目正文不进包（见下面 QUESTIONS_ONLY 的说明）——中心没有正文也能回答「有哪些数据集、各多少题」。
    'data/lang_bench/meta.v3.json',
    # 本机取题库的入口：随包放一份，拿到中心代码的人能在自己机器上按同一口径重建正文。
    'scripts/download_lang_bench_v3.py',
    'scripts/run_language_family.py', 'scripts/prepare_census_corpus.py', 'scripts/build_circuit_pack.py',
]
INCLUDE_DIRS = ['shared/dist_protocol', 'research/interventions', 'research/neurons', 'frontend/dist']
INCLUDE_OPTIONAL = ['data/lang_bench/results.v2.json']
# 题库正文（questions.v3.jsonl，3.3 MB，14,893 题）默认不进发布包：
#   ① 它不是中心的运行必需品——中心只用 meta.v3.json 的 by_dataset 回答数据集清单；
#   ② C-Eval 与 M3Exam 是 CC BY-NC-SA 4.0，随包分发等于再发一份副本，而题目本来就有稳定源地址
#      （见 shared/dist_protocol/bench_sources.py），需要的人在自己机器上一条命令就能取回来。
# 需要一份「题目齐全」的离线包时加 --with-questions。
QUESTIONS_ONLY = ['data/lang_bench/questions.v3.jsonl']
EXCLUDE_PARTS = {'__pycache__', '.pytest_cache'}


def files(with_questions: bool = False, frontend_dir: Path | None = None) -> list[Path]:
    out = [ROOT / p for p in INCLUDE]
    for d in INCLUDE_DIRS:
        directory = frontend_dir if d == 'frontend/dist' and frontend_dir is not None else ROOT / d
        out += [p for p in sorted(directory.rglob('*'))
                if p.is_file() and not (set(p.relative_to(directory).parts) & EXCLUDE_PARTS)]
    for p in INCLUDE_OPTIONAL + (QUESTIONS_ONLY if with_questions else []):
        if (ROOT / p).is_file():
            out.append(ROOT / p)
        else:
            print(f'[跳过] {p} 不存在：包里不带这份数据（中心端数据集清单会为空）')
    if not with_questions and (ROOT / QUESTIONS_ONLY[0]).is_file():
        print('[不含] data/lang_bench/questions.v3.jsonl：题目正文按源地址在本机取（--with-questions 才打进包）')
    missing = [str(p.relative_to(ROOT)) for p in out if not p.is_file()]
    if missing:
        raise SystemExit(f'缺少文件：{missing}')
    return out


class BoundedArchive(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError('压缩包超过内存暂存上限，请使用空间充足的项目盘或调整内存上限。')
        return super().write(data)


def write_archive(tar, members, name, manifest, version_bytes, partial=None):
    for path, relative in members:
        if partial is not None:
            check_write(partial, path.stat().st_size, operation='继续打包中心')
        info = tar.gettarinfo(str(path), arcname=f'{name}/{relative}')
        info.mode = 0o755 if path.suffix == '.sh' else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = 'root'
        if relative == 'VERSION':
            info.size = len(version_bytes)
            tar.addfile(info, io.BytesIO(version_bytes))
        else:
            with path.open('rb') as stream:
                tar.addfile(info, stream)
    data = json.dumps(manifest, ensure_ascii=False, indent=1).encode('utf-8')
    info = tarfile.TarInfo(f'{name}/RELEASE_MANIFEST.json')
    info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'release'))
    ap.add_argument('--skip-frontend-check', action='store_true')
    ap.add_argument('--frontend-dir', type=Path, default=ROOT / 'frontend/dist',
                    help='独立构建目录，包内仍安装到 frontend/dist')
    ap.add_argument('--compress-in-memory-mib', type=int, default=0, metavar='LIMIT',
                    help='在上限内先压缩，再按实测压缩大小检查落盘空间')
    ap.add_argument('--with-questions', action='store_true',
                    help='把题库正文 questions.v3.jsonl 也打进包（默认不打：中心只用 meta 摘要，题目按源地址本机取）')
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
    dist = args.frontend_dir.resolve()
    if not (dist / 'index.html').is_file():
        raise SystemExit('frontend/dist 不存在：先在 frontend/ 运行 vite build')
    if not args.skip_frontend_check:
        newest_src = max(p.stat().st_mtime for p in (ROOT / 'frontend' / 'src').rglob('*') if p.is_file())
        if (dist / 'index.html').stat().st_mtime < newest_src:
            raise SystemExit('frontend/dist 比源码旧：先重新 vite build（或 --skip-frontend-check）')
    fl = files(with_questions=args.with_questions, frontend_dir=dist)
    name = f'ai2050-center-{PLATFORM_VERSION}'
    requested = Path(args.out) / f'{name}.tar.gz'
    if args.compress_in_memory_mib < 0:
        ap.error('--compress-in-memory-mib must be nonnegative')
    tar_path = (check_write(requested, operation='中心发布目录') if args.compress_in_memory_mib
                else pack_target(requested, sum(p.stat().st_size for p in fl), len(fl)))
    out = tar_path.parent
    check_write(out / 'upgrade_center.sh', operation='发布脚本')
    check_write(vfile, operation='发布版本号')
    version_bytes = (PLATFORM_VERSION + '\n').encode('utf-8')
    members = [(p, ('frontend/dist/' + p.relative_to(dist).as_posix()
                    if p.is_relative_to(dist) else p.relative_to(ROOT).as_posix())) for p in fl]
    if len({relative for _, relative in members}) != len(members):
        raise ValueError('中心包包含重复的安装路径，请使用独立的客户端构建目录')
    import subprocess
    try:
        git = ['git', '-c', 'safe.directory=' + ROOT.as_posix()]
        commit = subprocess.run([*git, 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        dirty = bool(subprocess.run([*git, 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True,
                                    text=True, timeout=120, check=True).stdout.strip())
    except Exception:
        commit, dirty = '', None
    out.mkdir(parents=True, exist_ok=True)
    # Windows 上 core.autocrlf=true 的检出会把文本变成 CRLF：服务器上的脚本、unit、VERSION 与预注册哈希都会坏
    crlf = [relative for p, relative in members
            if (p.suffix in ('.sh', '.service', '.conf', '.sha256', '.tsv') or p.name.endswith('.prereg.json'))
            and b'\r' in p.read_bytes()]
    if crlf:
        raise SystemExit(f'这些文件含 CR（Windows 换行），在服务器上会出错：{crlf[:8]}…  用 git -c core.autocrlf=false 检出后再打包')
    manifest = {'name': name, 'version': PLATFORM_VERSION, 'previous_version': cur, 'git_commit': commit,
                'git_dirty': dirty, 'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                # 包里有没有题目正文：中心不需要它；记录在案，便于核对「这一版是按源地址引用还是自带题目」。
                'questions_included': args.with_questions,
                'files': {relative: hashlib.sha256(version_bytes if relative == 'VERSION' else p.read_bytes()).hexdigest()
                          for p, relative in members}}
    if args.compress_in_memory_mib:
        with BoundedArchive(args.compress_in_memory_mib * 2**20) as compressed:
            with tarfile.open(fileobj=compressed, mode='w:gz') as tar:
                write_archive(tar, members, name, manifest, version_bytes)
            with compressed.getbuffer() as payload:
                check_write(tar_path, len(payload) + 2**16, operation='写入已压缩中心包')
                with staging_file(tar_path) as partial:
                    with partial.open('wb') as stream:
                        stream.write(payload)
    else:
        with staging_file(tar_path) as partial, tarfile.open(partial, 'w:gz') as tar:
            write_archive(tar, members, name, manifest, version_bytes, partial)
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / f'{name}.tar.gz.sha256').write_bytes(f'{digest}  {tar_path.name}\n'.encode('utf-8'))
    (out / 'upgrade_center.sh').write_bytes((ROOT / 'deploy' / 'upgrade_center.sh').read_bytes())   # 与发布包一起上传，升级时直接运行
    if PLATFORM_VERSION != cur:
        vfile.write_bytes(version_bytes)
    print(f'版本 {cur} → {PLATFORM_VERSION}')
    print(f'{tar_path}  {tar_path.stat().st_size / 2**20:.1f} MB  {len(fl)} 个文件  sha256 {digest[:16]}')
    print(f'升级：把 {tar_path.name}、{tar_path.name}.sha256 与 upgrade_center.sh 传到服务器同一目录，运行 sudo bash upgrade_center.sh')
    print('提醒：VERSION 已改，请连同代码一起提交，保证仓库版本与发布包一致')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except StorageBlocked as error:
        stop_with_notice(error)
