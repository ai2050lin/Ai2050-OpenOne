# -*- coding: utf-8 -*-
"""把「题库数据集 + 读题库的接口」打成一个小包，单独上传到中心服务器覆盖升级。

为什么不复用 build_center_release：整包（约上百 MB）要重建客户端、重启全部服务，改一次题库
不值得走那条路。这个包只装「数据集 + 读它的代码 + 已构建的客户端」，解压后按 APPLY.md 覆盖到
中心程序目录即可，不动数据目录（/var/lib/ai2050）与配置文件。

包内容（相对路径即服务器上的落点，根目录 = 中心程序目录 /opt/ai2050/app）：
  data/lang_bench/questions.v3.jsonl    题库正文（C-Eval val / MMLU test / M3Exam zh test）
  data/lang_bench/meta.v3.json          题库清单（题量、三类分布、科目、许可）
  data/lang_bench/results.v2.json       已有的 (数据集, 模型) 实测结果（有才带）
  shared/dist_protocol/bench_index.py   题库索引（三方共用的唯一解析实现）
  shared/dist_protocol/bench_categories.py  三类 ↔ 科目的唯一映射表
  deploy/platform_service.py            中心侧两个只读接口 + /api/model/arch
  deploy/datapack.py                    离线数据包也带上这两个条目
  deploy/distributed_service.py         平台路由的依赖（同版本覆盖，内容不变）
  server/lang_bench_service.py          本机后端的实测端点（同源解析）
  server/lang_atlas_service.py          本机后端的数据集实测端点
  frontend/dist/**                      已构建的客户端（页顶数据集方块只在语言图谱出现）
  scripts/download_lang_bench_v3.py     重建题库的脚本（可追溯到原始 parquet）

用法：python scripts/build_data_pack.py [--out release] [--no-frontend]
输出：release/ai2050-datasets-<日期>-<git短码>.tar.gz 与 .sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

# 必带：缺一个就不打包（宁可报错，也不出一个「少半份题库」的包）
REQUIRED = [
    'data/lang_bench/questions.v3.jsonl',
    'data/lang_bench/meta.v3.json',
    'shared/dist_protocol/bench_index.py',
    'shared/dist_protocol/bench_categories.py',
    'deploy/platform_service.py',
    'deploy/datapack.py',
    'deploy/distributed_service.py',
    'server/lang_bench_service.py',
    'server/lang_atlas_service.py',
    'scripts/download_lang_bench_v3.py',
]
# 可选：有就带上，没有不报错
OPTIONAL = ['data/lang_bench/results.v2.json', 'data/lang_bench/results.json',
            'scripts/build_data_pack.py']
FRONTEND_DIR = 'frontend/dist'

# 许可随包声明：M3Exam 是 CC BY-NC-SA 4.0（非商业 + 相同方式共享），
# 上传到公开站点等于再分发，必须把这条写进包里的 NOTICE 与接口的 license 字段。
LICENSES = {
    'C-Eval val': 'CC BY-NC-SA 4.0',
    'MMLU test': 'MIT',
    'M3Exam zh test': 'CC BY-NC-SA 4.0',
}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ''
    except Exception:
        return ''


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect(with_frontend: bool) -> List[Path]:
    files: List[Path] = []
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit(f'缺少必需文件：{missing}')
    files += [ROOT / rel for rel in REQUIRED]
    files += [ROOT / rel for rel in OPTIONAL if (ROOT / rel).is_file()]
    if with_frontend:
        dist = ROOT / FRONTEND_DIR
        if not (dist / 'index.html').is_file():
            raise SystemExit('frontend/dist 里没有 index.html：先在 frontend 目录跑 npx vite build')
        files += [p for p in sorted(dist.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    return files


def _meta_summary() -> Dict[str, Any]:
    p = ROOT / 'data' / 'lang_bench' / 'meta.v3.json'
    meta = json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
    return {'schema': meta.get('schema'), 'by_source': meta.get('by_source') or {},
            'by_category': meta.get('by_category') or {}, 'unmapped': meta.get('unmapped') or {},
            'licenses': meta.get('licenses') or LICENSES}


APPLY_MD = """# 上传说明：题库数据集 + 读题库的接口

这个包只做一件事：让中心的「空间 · 语言图谱」与「进展 · 语言图谱」两页能读到题库清单与历史实测，
顺带把页顶的数据集方块收窄到这两页（神经元页与规则账本不再显示数据集选择）。

## 一、包里是什么

| 路径 | 作用 |
| --- | --- |
| `data/lang_bench/questions.v3.jsonl` | 题库正文：C-Eval val 169 / MMLU test 14,042 / M3Exam zh test 682，共 14,893 题 |
| `data/lang_bench/meta.v3.json` | 题库清单：题量、三类分布、科目、**许可** |
| `shared/dist_protocol/bench_index.py` | 题库索引的唯一实现（中心 / 离线包 / 本机后端共用） |
| `shared/dist_protocol/bench_categories.py` | 三类 ↔ 科目的映射表（唯一真源） |
| `deploy/platform_service.py` | 新增 `GET /api/lang/atlas/datasets`、`GET /api/lang/bench/overview`（另含 `/api/model/arch`） |
| `deploy/datapack.py` | 离线数据包也带这两个条目 |
| `server/lang_bench_service.py`、`server/lang_atlas_service.py` | 本机后端的实测端点，改用同一份解析；结果记录补 `dataset_key` / `dataset_id` / `dataset_license` |
| `frontend/dist/**` | 已构建的客户端（数据集方块只在语言图谱出现） |

不带：模型权重、普查语料、`/var/lib/ai2050` 下的数据目录、`platform_config.json`、`admin_token`。

## 二、装法（在服务器上，root）

```bash
cd /root
sha256sum -c ai2050-datasets-<版本>.tar.gz.sha256
mkdir -p /tmp/ai2050-data-pack && tar xzf ai2050-datasets-<版本>.tar.gz -C /tmp/ai2050-data-pack
cd /tmp/ai2050-data-pack/ai2050-datasets-*        # 目录名带版本

# 1) 先备份要覆盖的文件（出问题能退回去）
APP=/opt/ai2050/app
BK=/root/ai2050-app-backup-$(date +%Y%m%d-%H%M%S)
mkdir -p "$BK"
for f in deploy/platform_service.py deploy/datapack.py deploy/distributed_service.py \\
         server/lang_bench_service.py server/lang_atlas_service.py \\
         shared/dist_protocol/bench_index.py shared/dist_protocol/bench_categories.py; do
  mkdir -p "$BK/$(dirname "$f")"; [ -f "$APP/$f" ] && cp -a "$APP/$f" "$BK/$f"
done
[ -d "$APP/data/lang_bench" ] && cp -a "$APP/data/lang_bench" "$BK/" || true
echo "备份在 $BK"

# 2) 覆盖（保留属主，rsync 更稳）
rsync -a --chown=ai2050:ai2050 ./ "$APP"/

# 3) 重启中心
systemctl restart ai2050-center
sleep 3 && systemctl --no-pager status ai2050-center | head -5
```

没有 rsync 的话用 `cp -a` 逐个拷，注意 `data/lang_bench` 目录要一起建：
`install -d -o ai2050 -g ai2050 /opt/ai2050/app/data/lang_bench`。

## 三、验收（不用登录）

```bash
curl -s https://<域名>/api/lang/atlas/datasets | python3 -c \\
  'import json,sys;d=json.load(sys.stdin);print(d["available"]);[print(x["prefix"],x["n_questions"],x["graph_n"],x["license"]) for x in d["datasets"]]'
```

期望三行：`MMLU 14042 2504 MIT` / `M3Exam 682 565 CC BY-NC-SA 4.0` / `C-Eval 169 169 CC BY-NC-SA 4.0`。

```bash
curl -s https://<域名>/api/lang/bench/overview | python3 -c \\
  'import json,sys;d=json.load(sys.stdin);print(d["schema"], d["model_status"], len(d["datasets"]), len(d["results"]))'
```

期望 `lang_bench_overview.v3 unavailable 3 <历史结果条数>`。`model_status` 恒为 `unavailable`：
中心没有 GPU，`/api/lang/bench/run` 只在装模型的研究机上可用，网页只展示历史读数。

浏览器打开站点 → 空间 · 语言图谱：页顶应有数据集方块（C-Eval / MMLU / M3Exam / 本地数据集）；
切到 空间 · 神经元 或 进展 · 神经元逆向分析：数据集方块应消失，只剩模型。

## 四、许可（再分发前必看）

| 数据集 | 许可 | 说明 |
| --- | --- | --- |
| MMLU | MIT | 可再分发，保留出处 |
| C-Eval | CC BY-NC-SA 4.0 | 非商业 + 相同方式共享 |
| M3Exam | CC BY-NC-SA 4.0 | 非商业 + 相同方式共享 |

这两个 NC 数据集随包分发，服务本身必须是非商业用途且保留署名；若要转商业部署，
先删掉 M3Exam（`meta.v3.json` 的 `by_source` 与 `questions.v3.jsonl` 里 `source` 以 `M3Exam` 开头的行），
或改成只放题号索引、运行时按需下载。`data/lang_bench/` 在 .gitignore 里，不进公开仓库。

## 五、回滚

```bash
systemctl stop ai2050-center
cp -a /root/ai2050-app-backup-<时间戳>/. /opt/ai2050/app/
systemctl start ai2050-center
```

`data/lang_bench` 里旧题库（`questions.v2.jsonl`）不会被删；新接口按 v3 → v2 → v1 顺序取，
删掉 v3 文件即回到旧口径。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'release'))
    ap.add_argument('--no-frontend', action='store_true', help='不打客户端构建（只更新数据与接口）')
    args = ap.parse_args()

    files = collect(with_frontend=not args.no_frontend)
    date = time.strftime('%Y%m%d')
    commit = _git('rev-parse', '--short', 'HEAD') or 'nogit'
    dirty = bool(_git('status', '--porcelain'))
    version = f'{date}-{commit}'
    name = f'ai2050-datasets-{version}'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f'{name}.tar.gz'

    manifest: Dict[str, Any] = {
        'schema': 'ai2050_data_pack.v1',
        'name': name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': commit,
        'git_dirty': dirty,
        'version_file': (ROOT / 'VERSION').read_text(encoding='utf-8').strip() if (ROOT / 'VERSION').is_file() else '',
        'datasets': _meta_summary(),
        'licenses': LICENSES,
        'applies_to': '中心程序目录（/opt/ai2050/app）；不改数据目录与配置',
        'files': [],
    }

    tmp_files: List[Path] = []
    # 说明文件在打包时临时生成，直接写进临时目录再一起 tar
    staging = out_dir / f'.{name}.staging'
    staging.mkdir(parents=True, exist_ok=True)
    (staging / 'APPLY.md').write_text(APPLY_MD, encoding='utf-8', newline='')

    def add(tf: tarfile.TarFile, src: Path, arc: str) -> None:
        info = tf.gettarinfo(str(src), arcname=f'{name}/{arc}')
        info.uid = info.gid = 0
        info.uname = info.gname = 'root'
        info.mtime = int(time.time())
        with open(src, 'rb') as fh:
            tf.addfile(info, fh)

    total = 0
    with tarfile.open(tar_path, 'w:gz', compresslevel=6) as tf:
        for src in files:
            rel = src.relative_to(ROOT).as_posix()
            manifest['files'].append({'path': rel, 'bytes': src.stat().st_size, 'sha256': _sha256(src)})
            add(tf, src, rel)
            total += src.stat().st_size
        add(tf, staging / 'APPLY.md', 'APPLY.md')
        manifest['files'].append({'path': 'APPLY.md', 'bytes': (staging / 'APPLY.md').stat().st_size,
                                  'sha256': _sha256(staging / 'APPLY.md')})
        # MANIFEST 最后写：它要含上面全部条目的摘要。先写到磁盘再入包，便于用户单独核对。
        man_path = staging / 'MANIFEST.json'
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
        add(tf, man_path, 'MANIFEST.json')

    digest = _sha256(tar_path)
    # newline='' 关掉 Windows 的 \n → \r\n 转换：校验文件在 Linux 上要能被 `sha256sum -c` 直接读，
    # 带 \r 会被当成文件名的一部分而报「No such file or directory」。
    (out_dir / f'{name}.tar.gz.sha256').write_text(f'{digest}  {name}.tar.gz\n', encoding='utf-8', newline='')
    # 同一份说明在包外也放一份：上传前先看怎么装、许可能不能满足，不用先解包
    (out_dir / f'{name}-APPLY.md').write_text(APPLY_MD, encoding='utf-8', newline='')
    for p in staging.iterdir():
        p.unlink()
    staging.rmdir()

    size = tar_path.stat().st_size
    print(f'包：{tar_path}')
    print(f'大小：{size / 1048576:.2f} MB（{len(manifest["files"])} 个文件，原始 {total / 1048576:.2f} MB）')
    print(f'sha256：{digest}')
    print('数据集：', ', '.join(f'{k} {v}' for k, v in (manifest["datasets"]["by_source"] or {}).items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
