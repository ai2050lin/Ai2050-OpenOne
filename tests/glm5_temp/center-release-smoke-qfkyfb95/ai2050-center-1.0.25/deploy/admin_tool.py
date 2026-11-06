#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中心管理命令行（只依赖标准库）：在中心服务器上建普查作业、合并、处理复算不一致、开关作业。

管理令牌读取顺序：--token、环境变量 AI2050_DIST_ADMIN_TOKEN、/etc/ai2050/admin_token。

    python deploy/admin_tool.py census-create --job-id CJ-qwen3-4b-20M --model-dir /opt/ai2050/models/qwen3-4b \\
        --corpus-fp 2ca5f0a756d36c8a --tokens-per-slice 2000000 --hf-repo Qwen/Qwen3-4B --model-name qwen3-4b
    python deploy/admin_tool.py census-list
    python deploy/admin_tool.py census-merge CJ-qwen3-4b-20M
    python deploy/admin_tool.py census-review
    python deploy/admin_tool.py census-resolve <result_id> accept
    python deploy/admin_tool.py snapshot /path/to/model     # 在持有权重的机器上算快照，中心没有权重时用 --snapshot-file 传入
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def token(args) -> str:
    if args.token:
        return args.token
    if os.environ.get('AI2050_DIST_ADMIN_TOKEN'):
        return os.environ['AI2050_DIST_ADMIN_TOKEN']
    p = Path('/etc/ai2050/admin_token')
    return p.read_text().strip() if p.is_file() else ''


def call(args, path: str, body=None, method: str = '') -> dict:
    req = urllib.request.Request(args.server.rstrip('/') + path, method=method or ('POST' if body is not None else 'GET'),
                                 data=None if body is None else json.dumps(body).encode('utf-8'))
    req.add_header('Content-Type', 'application/json')
    t = token(args)
    if t:
        req.add_header('X-Admin-Token', t)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read().decode('utf-8') or '{}')
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{e.code}: {e.read().decode("utf-8", "replace")[:500]}')


def show(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='AI2050 center admin tool')
    ap.add_argument('--server', default=os.environ.get('AI2050_CENTER', 'http://127.0.0.1:5010'))
    ap.add_argument('--token', default='')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('census-create', help='建普查作业（模型目录与语料都在中心本机）')
    c.add_argument('--job-id', required=True)
    c.add_argument('--model-dir', required=True, help='中心本机目录：config.json + tokenizer.json（有权重时自动算快照）')
    c.add_argument('--snapshot-file', default='', help='中心没有权重时：由 snapshot 子命令在持有权重的机器上生成的 JSON')
    c.add_argument('--corpus-fp', required=True, help='<AI2050_NATIVE_DIR>/_census/corpus/<指纹>')
    c.add_argument('--tokens-per-slice', type=int, default=2_000_000)
    c.add_argument('--max-slices', type=int, default=None)
    c.add_argument('--group-size', type=int, default=4, help='每个分片的层数')
    c.add_argument('--replica-rate', type=float, default=0.1, help='派给另一个账号复算的分片比例')
    c.add_argument('--model-name', default='')
    c.add_argument('--hf-repo', default='')
    c.add_argument('--hf-revision', default='')
    sub.add_parser('census-list', help='作业与进度')
    m = sub.add_parser('census-merge', help='合并已完成的切片（中心后台也会定时自动合并）')
    m.add_argument('job_id')
    st = sub.add_parser('census-status', help='open / paused / closed')
    st.add_argument('job_id')
    st.add_argument('status', choices=['open', 'paused', 'closed'])
    sub.add_parser('census-review', help='复算不一致的结果')
    rs = sub.add_parser('census-resolve', help='处理复算不一致：接受一份（另一份驳回）或驳回')
    rs.add_argument('result_id')
    rs.add_argument('action', choices=['accept', 'reject'])
    rs.add_argument('--note', default='')
    sn = sub.add_parser('snapshot', help='计算本机模型目录的快照（权重与 tokenizer 文件 sha256）')
    sn.add_argument('model_dir')
    sn.add_argument('--out', default='')
    args = ap.parse_args()

    if args.cmd == 'census-create':
        body = {k: v for k, v in {
            'job_id': args.job_id, 'model_dir': args.model_dir, 'corpus_fp': args.corpus_fp,
            'tokens_per_slice': args.tokens_per_slice, 'max_slices': args.max_slices, 'group_size': args.group_size,
            'replica_rate': args.replica_rate, 'model_name': args.model_name, 'hf_repo': args.hf_repo,
            'hf_revision': args.hf_revision}.items() if v not in (None, '')}
        if args.snapshot_file:
            body['snapshot'] = json.loads(Path(args.snapshot_file).read_text(encoding='utf-8'))
        show(call(args, '/api/census/jobs', body))
    elif args.cmd == 'census-list':
        for j in call(args, '/api/census/jobs')['jobs']:
            print(f"{j['job_id']:28s} {j['status']:7s} {j['model'].get('name') or ''} · 分片 {j['n_shards']} · {j['counts']} · "
                  f"已完成切片 {len(j['slices_done'])}/{j['n_slices']}" + (f" · 已合并 {j['merged']['n_tokens']} token" if j.get('merged') else ''))
    elif args.cmd == 'census-merge':
        show(call(args, f'/api/census/jobs/{args.job_id}/merge', {}))
    elif args.cmd == 'census-status':
        show(call(args, f'/api/census/jobs/{args.job_id}/status', {'status': args.status}))
    elif args.cmd == 'census-review':
        show(call(args, '/api/census/review'))
    elif args.cmd == 'census-resolve':
        show(call(args, f'/api/census/results/{args.result_id}/review', {'action': args.action, 'note': args.note}))
    elif args.cmd == 'snapshot':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from node_agent import model_snapshot
        snap = model_snapshot(args.model_dir)
        text = json.dumps(snap, ensure_ascii=False, indent=1)
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        print(text if not args.out else f"{snap['sha256']} → {args.out}")


if __name__ == '__main__':
    main()
