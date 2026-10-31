# -*- coding: utf-8 -*-
"""批量审计工具：把审计队列里自动检查无 fail 的结果批量标记为通过。

背景（2026-10-31 线上核验）：76 份 pending 中 75 份 auto_verdict=warn、1 份 fail。
- warn 两类都是系统性提示而非质量问题：
  1) work_unit 未绑定 —— 开放领取（node participate）路径不创建平台任务，结果天然不带
     _research_work_unit；只有工作台「平台任务派发」路径的结果才有绑定。
  2) behavior 低于阈值 —— 分片级逐条准确率低于预注册阈值的条件格，合并（AGG-v2）时自动排除，
     属设计内行为；服务端审计只在 fail 时拒绝通过（409），warn 允许通过。
- fail（1 份，TM-04 legacy 结果 sha edd1f724…）：模板种子升级后 design_sha 变更，旧结果与当前
  冻结设计不符，不能通过，建议 needs_rerun（或 rejected）。

审计纪律：本工具只决定执行产物能否进入合并统计，不创建 Evidence（Research OS 登记不变）。

用法（在能访问中心的管理机上）：
  # 试运行（只打印计划，不写任何东西）
  python deploy/audit_batch.py --base https://server.ai2050.com
  # 实际执行：需要管理令牌（服务器上 /etc/ai2050/admin_token）
  python deploy/audit_batch.py --base https://server.ai2050.com --apply \
      --token-file /etc/ai2050/admin_token --reviewer <你的名字>
  # fail 项处置（默认 needs_rerun；可选 rejected / skip）
  python deploy/audit_batch.py --base https://server.ai2050.com --apply \
      --token-file /etc/ai2050/admin_token --fail-action needs_rerun
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS_NOTE = ('自动检查无 fail；warn=work_unit 未绑定（开放领取路径不带平台任务绑定）'
             '与/或行为门分片低于阈值（合并时自动排除），均为系统性提示非质量问题')
RERUN_NOTE = 'design_sha 与当前冻结模板不符（模板升级后旧结果过期）：需按当前设计重跑'


def req(base: str, path: str, method: str = 'GET', payload=None, token: str = '', timeout: float = 60):
    r = urllib.request.Request(base.rstrip('/') + path, method=method)
    r.add_header('Content-Type', 'application/json')
    if token:
        r.add_header('X-Admin-Token', token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(r, data=data, timeout=timeout) as resp:
            b = resp.read()
            return resp.status, (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {'raw': body[:300]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', required=True, help='中心地址，如 https://server.ai2050.com')
    ap.add_argument('--apply', action='store_true', help='实际提交审计裁决（缺省只打印计划）')
    ap.add_argument('--token', default='', help='管理令牌（--apply 时必填，除非 --token-file）')
    ap.add_argument('--token-file', default='', help='从文件读管理令牌（服务器上 /etc/ai2050/admin_token）')
    ap.add_argument('--reviewer', default='audit_batch', help='审计人署名')
    ap.add_argument('--fail-action', default='needs_rerun', choices=['needs_rerun', 'rejected', 'skip'],
                    help='fail 项处置：needs_rerun（默认）/ rejected / skip（留着人工处理）')
    ap.add_argument('--limit', type=int, default=500)
    args = ap.parse_args()

    token = args.token
    if args.token_file:
        with open(args.token_file, encoding='utf-8') as fh:
            token = fh.read().strip()

    st, q = req(args.base, f'/api/audit/queue?status=pending&limit={args.limit}')
    if st != 200 or not q:
        print(f'拉取审计队列失败：HTTP {st} {q}')
        return 1
    items = q['items']
    print(f"pending={q['counts']['pending']}  拉到 {len(items)} 项")
    plan = {'passed': [], 'needs_rerun': [], 'rejected': [], 'skip': []}
    for it in items:
        if it['auto_verdict'] == 'fail':
            plan['skip' if args.fail_action == 'skip' else args.fail_action].append(it)
        else:
            plan['passed'].append(it)
    print(f"计划：通过 {len(plan['passed'])} · needs_rerun {len(plan['needs_rerun'])} · "
          f"rejected {len(plan['rejected'])} · 留人工 {len(plan['skip'])}")
    for it in plan['needs_rerun'] + plan['rejected'] + plan['skip']:
        bad = '; '.join(f"{c['id']}:{c['status']}" for c in it['checks'] if c['status'] == 'fail')
        print(f"  {it['auto_verdict'].upper()} {it['tm_id']} shard={it['shard']} {it['model_id']} "
              f"sha={it['sha'][:12]} | {bad}")

    if not args.apply:
        print('\n（dry-run，未写任何数据；加 --apply --token-file /etc/ai2050/admin_token 执行）')
        return 0
    if not token:
        print('缺少管理令牌：--token 或 --token-file')
        return 1
    ok = err = 0
    for it in plan['passed']:
        st, r = req(args.base, f"/api/audit/{it['sha']}", 'POST',
                    {'status': 'passed', 'reviewer': args.reviewer, 'note': PASS_NOTE}, token=token)
        if st == 200:
            ok += 1
        else:
            err += 1
            print(f"  通过失败 {it['sha'][:12]} HTTP {st}: {str(r)[:120]}")
    for action in ('needs_rerun', 'rejected'):
        for it in plan[action]:
            st, r = req(args.base, f"/api/audit/{it['sha']}", 'POST',
                        {'status': action, 'reviewer': args.reviewer, 'note': RERUN_NOTE}, token=token)
            if st == 200:
                ok += 1
            else:
                err += 1
                print(f"  {action} 失败 {it['sha'][:12]} HTTP {st}: {str(r)[:120]}")
    print(f'\n已提交 {ok} 项，失败 {err} 项')
    st, q2 = req(args.base, '/api/audit/queue?status=all&limit=1')
    if st == 200:
        print('审计队列现状：', json.dumps(q2['counts'], ensure_ascii=False))
    return 0 if err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
