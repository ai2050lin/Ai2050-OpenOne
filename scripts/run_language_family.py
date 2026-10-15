# -*- coding: utf-8 -*-
"""一个语言模式族的完整测试流程（dist_protocol v3）：

    文法预览 → 冻结试跑版 → 节点跑试跑分片（行为门）→ 冻结正式模板 → 逐分片采集
    → 中心合并 → 规律分析（analysis.v1）→ 写研究报告（Markdown + JSON）

真实模型（GPU 节点，Windows / Linux 通用）：
    python scripts/run_language_family.py --family KN-ISA ^
        --center http://127.0.0.1:5001 --model-path D:\\AI2050\\Ai2050-OpenOne\\models\\hf\\qwen3-4b ^
        --model-id qwen3-4b --dtype bf16

流程验证（无 GPU；必须连接沙盒中心：AI2050_DIST_ALLOW_SYNTHETIC=1 + 独立 AI2050_DIST_DIR）：
    python scripts/run_language_family.py --family KN-ISA --center http://127.0.0.1:5010 --synthetic

所有步骤都走中心节点 HTTP 接口与 deploy/node_agent.py，和客户端看到的是同一份数据；
运行结束后在客户端「空间 · 实验响应 → 3D 响应空间 / 规律分析」查看。
合成模式产生的结果、报告都标注 SYNTHETIC，只验证方法与流程，不能写入研究记录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'deploy' / 'node_agent.py'


def http(base, path, method='GET', payload=None, admin=''):
    req = urllib.request.Request(base.rstrip('/') + path, method=method)
    req.add_header('Content-Type', 'application/json')
    if admin:
        req.add_header('X-Admin-Token', admin)
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=600) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:500]
        raise SystemExit(f'[{method} {path}] HTTP {exc.code}: {detail}')
    except urllib.error.URLError as exc:
        raise SystemExit(f'中心节点不可达：{base}（{exc.reason}）。先启动后端 server.server 或 deploy/distributed_service。')


PROGRESS = False   # --progress：额外输出机器可读进度行（研发 · 自动测试队列解析）


def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)


def progress(step, label, detail='', **extra):
    if PROGRESS:
        print('@@progress ' + json.dumps({'step': step, 'label': label, 'detail': detail, **extra}, ensure_ascii=False), flush=True)


def bucket_suffix(args):
    return f'{args.model_id}|{args.model_rev}|{args.dtype.lower()}'


def model_rev_of(model_path: str) -> str:
    cfg = Path(model_path) / 'config.json'
    if cfg.is_file():
        return 'cfg-' + hashlib.sha256(cfg.read_bytes()).hexdigest()[:12]
    return 'unknown'


def run_agent(args, tm_id, env):
    data_dir = Path(args.data_dir)
    if not (data_dir / 'node_credentials.json').is_file():
        subprocess.run([args.python, str(AGENT), '--server', args.center, '--data-dir', str(data_dir), 'register',
                        '--name', args.node_name, '--gpu', args.gpu, '--model', args.model_id, '--dtype', args.dtype],
                       check=True, env=env)
    cmd = [args.python, str(AGENT), '--server', args.center, '--data-dir', str(data_dir), 'run', '--once',
           '--tm-id', tm_id, '--model-id', args.model_id, '--model-rev', args.model_rev, '--dtype', args.dtype,
           '--gpu', args.gpu, '--python', args.python, '--run-timeout', str(args.run_timeout)]
    if args.model_path:
        cmd += ['--model-path', args.model_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit('node_agent 执行失败')
    return 'claimed' in proc.stdout and 'task failed' not in proc.stdout


def fmt(v, d=3):
    return '—' if v is None else (f'{v:.{d}f}' if isinstance(v, (int, float)) else str(v))


def write_report(args, family, tm, analysis, report, scoreboard, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    tag = 'SYNTHETIC-' if args.synthetic else ''
    base = out_dir / f'{tag}{tm}-{stamp}'
    base.with_suffix('.json').write_text(json.dumps({'analysis': analysis, 'report': report}, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    L = []
    if args.synthetic:
        L += ['> **SYNTHETIC — 方法与流程验证。** 数据由 runner_v3 合成模式按已知规律生成，不代表任何真实模型，不得写入研究记录。', '']
    L += [f'# 模式族测试报告：{family}（{tm}）', '',
          f'- 时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
          f'- 模型桶：`{analysis.get("bucket")}`；分片 {len(analysis.get("shards_present", []))}/{analysis.get("shards_expected")}',
          f'- 样本：{analysis.get("n_items")} 条进入分析（行为门排除 {analysis.get("n_excluded")} 条）；向量空间 {analysis.get("space")}',
          f'- 因子：{" × ".join(analysis.get("factors", []))}；判据：`{json.dumps(analysis.get("criteria", {}), ensure_ascii=False)}`', '',
          '## 规律陈述', '']
    for f in analysis.get('findings', []):
        mark = {'supported': '✓', 'negative': '✗', 'warning': '!', 'info': '·'}.get(f['kind'], '·')
        L.append(f'- **{mark} {f["title"]}**（{f.get("level", "")}）：{f["text"]}' + (f' 适用范围：{f["scope"]}' if f.get('scope') else ''))
    L += ['', '## 逐层指标', '', '| 层 | 因子 | 迹 η²（校正） | confirm 探针 | lockbox 探针 | lockbox_frame 探针 | lockbox 方向 cos（白化） | 置换 p（探针 / 方向） |',
          '|---|---|---|---|---|---|---|---|']
    for layer in analysis.get('layers', []):
        if layer.get('insufficient'):
            continue
        for f, r in layer['factors'].items():
            sp = r['splits']
            perm = r.get('perm') or {}
            L.append(f'| {layer["tag"]} | {f} | {fmt(layer["trace_eta2"].get(f))} | {fmt((sp.get("confirm") or {}).get("probe_acc"))} | '
                     f'{fmt((sp.get("lockbox") or {}).get("probe_acc"))} | {fmt((sp.get("lockbox_frame") or {}).get("probe_acc"))} | '
                     f'{fmt((sp.get("lockbox") or {}).get("rep_cos_white"))} | {fmt(perm.get("acc_p"), 4)} / {fmt(perm.get("rep_p"), 4)} |')
    L += ['', '## 证据门链', '']
    for g in report.get('gates', []):
        L.append(f'- {g["label"]}：{g["status"]} — {g.get("note", "")}')
    L += ['', '## 行为门', '']
    for b in report.get('buckets', []):
        bad = [c for c, r in b['behavior'].items() if not r['pass']]
        L.append(f'- `{b["bucket"]}`：{len(b["behavior"]) - len(bad)}/{len(b["behavior"])} 格达标' + (f'；未达标：{", ".join(bad)}' if bad else ''))
    L += ['', '## 可视化', '', '客户端 → 空间 → 实验响应 → 选择模板 `' + tm + '` → 「3D 响应空间」「规律分析」。', '',
          '## 边界', '', '- 规律陈述来自观测统计与留出预测，属于候选稳定观测；选择性/充分性、必要性与自然生成需要干预实验。',
          '- 证据等级在 Research OS 登记；本报告不自动晋级。']
    base.with_suffix('.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return base.with_suffix('.md')


def main():
    ap = argparse.ArgumentParser(description='Run one language-pattern family end to end (dist_protocol v3)')
    ap.add_argument('--family', required=True, help='shared/dist_protocol/grammars/ 中的 family_id，如 KN-ISA')
    ap.add_argument('--center', default=os.environ.get('AI2050_DIST_SERVER', 'http://127.0.0.1:5001'))
    ap.add_argument('--admin-token', default=os.environ.get('AI2050_DIST_ADMIN_TOKEN', 'ai2050-admin-dev'))
    ap.add_argument('--model-path', default='')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--model-rev', default='')
    ap.add_argument('--dtype', default='bf16')
    ap.add_argument('--gpu', default='')
    ap.add_argument('--node-name', default='family-runner')
    ap.add_argument('--data-dir', default=str(Path.home() / '.ai2050_agent'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--run-timeout', type=float, default=7200.0)
    ap.add_argument('--pilot-per-cell', type=int, default=3)
    ap.add_argument('--skip-pilot', action='store_true')
    ap.add_argument('--min-pilot-pass', type=float, default=0.5, help='试跑行为门达标格比例低于该值时停止（不冻结正式模板）')
    ap.add_argument('--synthetic', action='store_true', help='合成模式：只用于沙盒中心验证流程')
    ap.add_argument('--report-dir', default=str(ROOT / 'research' / 'family_runs'))
    ap.add_argument('--progress', action='store_true', help='输出 @@progress / @@result 机器可读行（供中心作业队列解析）')
    args = ap.parse_args()
    global PROGRESS
    PROGRESS = args.progress

    env = dict(os.environ)
    if args.synthetic:
        args.model_id = args.model_id or 'synthetic-family'
        args.model_rev = args.model_rev or 'synthetic-v1'
        args.model_path = ''
        env['AI2050_RUNNER_MODE'] = 'synthetic'
        env['AI2050_SYNTHETIC_SALT'] = args.model_id
        args.report_dir = str(Path(args.report_dir).parent / 'family_runs_synthetic')
        overview = http(args.center, '/api/pipeline/overview')
        if not overview.get('sandbox_synthetic'):
            raise SystemExit('合成模式只能连接沙盒中心（启动时设置 AI2050_DIST_ALLOW_SYNTHETIC=1 与独立的 AI2050_DIST_DIR）。')
    else:
        if not args.model_path or not Path(args.model_path).is_dir():
            raise SystemExit('真实模式需要 --model-path 指向本机 HF 模型目录（或使用 --synthetic 验证流程）')
        args.model_id = args.model_id or Path(args.model_path).name
        args.model_rev = args.model_rev or model_rev_of(args.model_path)
        env.pop('AI2050_RUNNER_MODE', None)

    progress(1, '文法预览', f'{args.family} × {args.model_id}')
    lib = http(args.center, '/api/grammar/library')
    spec = next((g['spec'] for g in lib['grammars'] if g.get('family_id') == args.family), None)
    if not spec:
        raise SystemExit(f'模板族 {args.family} 不在文法库中：{[g.get("family_id") for g in lib["grammars"]]}')
    pv = http(args.center, '/api/grammar/preview', 'POST', {'spec': spec})
    if not pv['ok']:
        raise SystemExit('文法错误：' + '；'.join(pv['errors']))
    log(f'① 文法预览：{pv["n_items"]} 条 · {pv["by_split"]} · 分片 {pv["shards"]}')
    for w in pv['warnings']:
        log(f'   告警：{w}')

    progress(1, '文法预览', f'{pv["n_items"]} 条 · {pv["shards"]} 分片')

    if not args.skip_pilot:
        progress(2, '试跑与行为门', '冻结试跑版')
        pilot = http(args.center, '/api/grammar/register', 'POST',
                     {'spec': spec, 'mode': 'pilot', 'pilot_per_cell': args.pilot_per_cell}, args.admin_token)
        log(f'② 冻结试跑版 {pilot["tm_id"]} v{pilot["version"]}（{pilot["n_items"]} 条）')
        progress(2, '试跑与行为门', f'{pilot["tm_id"]} · 节点运行中', pilot_tm_id=pilot['tm_id'])
        run_agent(args, pilot['tm_id'], env)
        rep = http(args.center, f'/api/grammar/{urllib.parse.quote(pilot["tm_id"])}/report')
        own = [b for b in rep['buckets'] if b['bucket'].endswith(bucket_suffix(args))]   # 只看本模型的试跑桶
        cells = [c for b in own for c in b['behavior'].values()]
        passed = sum(1 for c in cells if c['pass'])
        log(f'   试跑行为门：{passed}/{len(cells)} 格达标')
        progress(2, '试跑与行为门', f'{passed}/{len(cells)} 格达标', pilot_tm_id=pilot['tm_id'])
        if not cells:
            raise SystemExit('试跑没有产生本模型的行为结果：检查 node_agent 输出。')
        if passed / len(cells) < args.min_pilot_pass:
            log(f'试跑行为门达标比例 {passed}/{len(cells)} 低于 {args.min_pilot_pass}，停止：先修改文法或换模型，再重新运行。')
            sys.exit(3)

    final = http(args.center, '/api/grammar/register', 'POST', {'spec': spec, 'mode': 'final'}, args.admin_token)
    tm = final['tm_id']
    log(f'③ 冻结正式模板 {tm} v{final["version"]} · design {final["design_sha"][:12]} · {final["shards"]} 分片')
    progress(3, '冻结正式模板', f'{tm} v{final["version"]} · {final["shards"]} 分片', tm_id=tm)

    for i in range(final['shards'] + 2):
        rep = http(args.center, f'/api/grammar/{urllib.parse.quote(tm)}/report')
        mine = [b for b in rep['buckets'] if b['bucket'].endswith(bucket_suffix(args))]
        if mine and mine[0]['complete']:
            break
        have = len(mine[0]['shards_present']) if mine else 0
        log(f'④ 采集分片（第 {i + 1} 次领取）')
        progress(4, '分片采集', f'已完成 {have}/{final["shards"]} 分片', tm_id=tm)
        if not run_agent(args, tm, env):
            log('   没有领到任务（可能其他节点在跑），10 秒后重查')
            time.sleep(10)
    rep = http(args.center, f'/api/grammar/{urllib.parse.quote(tm)}/report')
    mine = [b for b in rep['buckets'] if b['bucket'].endswith(bucket_suffix(args))]
    if not mine:
        raise SystemExit('中心没有可合并的结果：检查 node_agent 输出（smoke 结果不参与合并）。')
    bucket = mine[0]['bucket']
    log(f'⑤ 合并：{bucket} 分片 {len(mine[0]["shards_present"])}/{mine[0]["shards_expected"]}')
    progress(5, '合并', f'分片 {len(mine[0]["shards_present"])}/{mine[0]["shards_expected"]}', tm_id=tm, bucket=bucket)
    if not mine[0]['complete']:
        raise SystemExit(f'分片未采集完整（{len(mine[0]["shards_present"])}/{mine[0]["shards_expected"]}）：检查 node_agent 输出后重跑。')
    progress(6, '规律分析', '计算中（置换检验需要一点时间）', tm_id=tm, bucket=bucket)

    analysis = http(args.center, f'/api/analysis/{urllib.parse.quote(tm)}?bucket={urllib.parse.quote(bucket)}')
    if not analysis.get('available'):
        raise SystemExit('规律分析不可用：' + analysis.get('reason', ''))
    sb = http(args.center, '/api/scoreboard')
    log(f'⑥ 规律分析：{len(analysis["findings"])} 条陈述')
    for f in analysis['findings']:
        log(f'   [{f["kind"]}] {f["title"]}')
    path = write_report(args, args.family, tm, analysis, rep, sb, Path(args.report_dir))
    log(f'⑦ 报告：{path}')
    kinds = {}
    for f in analysis['findings']:
        kinds[f['kind']] = kinds.get(f['kind'], 0) + 1
    progress(7, '报告', str(path), tm_id=tm, bucket=bucket)
    if PROGRESS:
        print('@@result ' + json.dumps({
            'tm_id': tm, 'bucket': bucket, 'report': str(path), 'synthetic': bool(args.synthetic),
            'n_items': analysis.get('n_items'), 'n_excluded': analysis.get('n_excluded'), 'findings': kinds,
            'headlines': [{'kind': f['kind'], 'title': f['title']} for f in analysis['findings']][:12],
        }, ensure_ascii=False), flush=True)
    log('   客户端：空间 → 实验响应 → 模板 ' + tm + ' → 「3D 响应空间」「规律分析」；待审计结果在 研发 → ④ 审计队列。')


if __name__ == '__main__':
    main()
