#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""忘记密码时在中心服务器上重置账号密码（只依赖标准库，直接写中心数据库，不经接口、不需要管理令牌）。

数据目录读取顺序：--data-dir、环境变量 AI2050_DIST_DIR、/var/lib/ai2050。新密码从终端输入两次（不回显、不进命令行历史）。
重置后该账号的全部登录失效、锁定与停用解除；--make-admin 同时设为管理员（管理员账号全部丢失时用）。

    sudo -u ai2050 python3 deploy/reset_password.py --list
    sudo -u ai2050 python3 deploy/reset_password.py alice
"""
from __future__ import annotations

import argparse
import getpass
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.dist_protocol import accounts as A  # noqa: E402


def open_db(data_dir: str) -> sqlite3.Connection:
    p = Path(data_dir) / 'dist.sqlite3'
    if not p.is_file():
        raise SystemExit(f'找不到 {p}；用 --data-dir 指向中心的 AI2050_DIST_DIR（systemctl show ai2050-center -p Environment）')
    db = sqlite3.connect(str(p), timeout=30)
    db.row_factory = sqlite3.Row
    A.ensure_schema(db)
    return db


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='AI2050 center: reset an account password')
    ap.add_argument('username', nargs='?')
    ap.add_argument('--data-dir', default=os.environ.get('AI2050_DIST_DIR', '/var/lib/ai2050'))
    ap.add_argument('--list', action='store_true', help='列出账号')
    ap.add_argument('--make-admin', action='store_true', help='同时设为管理员')
    args = ap.parse_args(argv)
    db = open_db(args.data_dir)
    try:
        if args.list or not args.username:
            for u in A.list_users(db):
                print(f"{u['username']:<24} {u['role']:<7} {'停用' if u['disabled'] else ''}")
            return 0
        pw = getpass.getpass('新密码（至少 8 位）: ')
        if getpass.getpass('再输入一次: ') != pw:
            raise SystemExit('两次输入不一致，未修改')
        u = A.reset_password(db, args.username, pw, make_admin=args.make_admin)
    except A.AuthError as e:
        raise SystemExit(e.message)
    finally:
        db.close()
    print(f"已重置 {u['username']}（{u['role']}）；该账号的其他登录已失效")
    return 0


if __name__ == '__main__':
    sys.exit(main())
