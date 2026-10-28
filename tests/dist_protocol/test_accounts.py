# -*- coding: utf-8 -*-
"""accounts.v1：注册（首个账号为管理员）、登录、锁定、会话、改密码、注册策略、角色管理、节点绑定码。"""
import os
import sqlite3
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import accounts as A  # noqa: E402


class AccountsTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE nodes(node_id TEXT PRIMARY KEY, name TEXT)')
        A.ensure_schema(self.db)
        os.environ.pop('AI2050_REGISTRATION', None)

    def tearDown(self):
        self.db.close()

    def test_first_user_is_admin_then_members(self):
        st = A.status(self.db)
        self.assertFalse(st['has_users'])
        self.assertTrue(st['first_user_becomes_admin'])
        r = A.register(self.db, 'alice', 'secret-123', display_name='Alice')
        self.assertEqual(r['user']['role'], 'admin')
        self.assertTrue(r['token'])
        r2 = A.register(self.db, 'bob_2', 'another-456')
        self.assertEqual(r2['user']['role'], 'member')
        self.assertEqual(A.session_user(self.db, r2['token'])['username'], 'bob_2')
        row = self.db.execute("SELECT pw_hash, pw_salt FROM users WHERE username='alice'").fetchone()
        self.assertNotIn('secret', row['pw_hash'])                            # 只存哈希
        self.assertEqual(len(row['pw_salt']), 32)
        tok = self.db.execute('SELECT token_hash FROM sessions').fetchone()['token_hash']
        self.assertNotEqual(tok, r['token'])                                   # 只存令牌哈希

    def test_validation(self):
        for name, pw in [('ab', 'secret-123'), ('bad name', 'secret-123'), ('carol', 'short1'),
                         ('carol', '12345678901'), ('carol', 'abcdefghij')]:
            with self.assertRaises(A.AuthError) as cm:
                A.register(self.db, name, pw)
            self.assertEqual(cm.exception.status, 400)
        A.register(self.db, 'carol', 'secret-123')
        with self.assertRaises(A.AuthError) as cm:
            A.register(self.db, 'CAROL', 'secret-123')                         # 用户名不区分大小写
        self.assertEqual(cm.exception.status, 409)

    def test_login_lockout_and_logout(self):
        A.register(self.db, 'dave', 'secret-123')
        for _ in range(A.MAX_FAILS - 1):
            with self.assertRaises(A.AuthError) as cm:
                A.login(self.db, 'dave', 'wrong-pass1')
            self.assertEqual(cm.exception.status, 401)
        with self.assertRaises(A.AuthError) as cm:
            A.login(self.db, 'dave', 'wrong-pass1')
        self.assertIn('锁定', cm.exception.message)
        with self.assertRaises(A.AuthError) as cm:
            A.login(self.db, 'dave', 'secret-123')                             # 锁定期内正确密码也拒绝
        self.assertEqual(cm.exception.status, 423)
        self.db.execute("UPDATE login_failures SET locked_until=? WHERE username='dave'", (time.time() - 1,))
        r = A.login(self.db, 'dave', 'secret-123')
        self.assertEqual(A.require_user(self.db, f'Bearer {r["token"]}')['username'], 'dave')
        A.logout(self.db, r['token'])
        self.assertIsNone(A.session_user(self.db, r['token']))
        with self.assertRaises(A.AuthError) as cm:
            A.login(self.db, 'nobody', 'secret-123')
        self.assertEqual(cm.exception.status, 401)

    def test_session_expiry_and_password_change(self):
        a = A.register(self.db, 'erin', 'secret-123')
        b = A.login(self.db, 'erin', 'secret-123')
        A.change_password(self.db, a['user']['user_id'], 'secret-123', 'new-secret-9', keep_token=a['token'])
        self.assertIsNotNone(A.session_user(self.db, a['token']))               # 当前会话保留
        self.assertIsNone(A.session_user(self.db, b['token']))                  # 其他会话失效
        with self.assertRaises(A.AuthError):
            A.change_password(self.db, a['user']['user_id'], 'secret-123', 'x-new-pass-1')
        self.db.execute('UPDATE sessions SET expires_at=?', (time.time() - 1,))
        self.assertIsNone(A.session_user(self.db, a['token']))
        with self.assertRaises(A.AuthError) as cm:
            A.require_user(self.db, f'Bearer {a["token"]}')
        self.assertEqual(cm.exception.status, 401)

    def test_closed_registration_and_admin_management(self):
        admin = A.register(self.db, 'root', 'secret-123')                      # 首个账号总能注册
        with mock.patch.dict(os.environ, {'AI2050_REGISTRATION': 'closed'}):
            self.assertFalse(A.status(self.db)['can_register'])
            with self.assertRaises(A.AuthError) as cm:
                A.register(self.db, 'frank', 'secret-123')
            self.assertEqual(cm.exception.status, 403)
            frank = A.create_user(self.db, 'frank', 'secret-123', by_admin=True)
        actor = admin['user']
        with self.assertRaises(A.AuthError):
            A.update_user(self.db, actor, actor['user_id'], role='member')      # 至少保留一个管理员
        A.update_user(self.db, actor, frank['user_id'], role='admin')
        A.update_user(self.db, actor, actor['user_id'], role='member')
        f = A.login(self.db, 'frank', 'secret-123')
        A.update_user(self.db, A.session_user(self.db, f['token']), admin['user']['user_id'], disabled=True)
        self.assertIsNone(A.session_user(self.db, admin['token']))              # 停用后会话失效
        with self.assertRaises(A.AuthError) as cm:
            A.login(self.db, 'root', 'secret-123')
        self.assertEqual(cm.exception.status, 403)
        with self.assertRaises(A.AuthError):
            A.require_admin(self.db, f'Bearer {A.register(self.db, "gina", "secret-123")["token"]}')
        self.assertEqual([u['username'] for u in A.list_users(self.db)], ['root', 'frank', 'gina'])

    def test_server_side_password_reset(self):
        A.register(self.db, 'alice', 'secret-123')
        bob = A.register(self.db, 'bob_2', 'another-456')
        for _ in range(A.MAX_FAILS):
            with self.assertRaises(A.AuthError):
                A.login(self.db, 'bob_2', 'wrong-pass')
        self.db.execute("UPDATE users SET disabled=1 WHERE username='bob_2'")
        self.db.commit()
        u = A.reset_password(self.db, 'bob_2', 'brand-new-789', make_admin=True)
        self.assertEqual(u['role'], 'admin')
        self.assertFalse(u['disabled'])
        self.assertIsNone(A.session_user(self.db, bob['token']))              # 旧会话失效
        self.assertTrue(A.login(self.db, 'bob_2', 'brand-new-789')['token'])  # 锁定已解除
        with self.assertRaises(A.AuthError):
            A.login(self.db, 'bob_2', 'another-456')
        with self.assertRaises(A.AuthError) as e:
            A.reset_password(self.db, 'nobody', 'brand-new-789')
        self.assertEqual(e.exception.status, 404)
        with self.assertRaises(A.AuthError):
            A.reset_password(self.db, 'alice', 'short')

    def test_reset_password_cli(self):
        import tempfile
        sys.path.insert(0, str(ROOT / 'deploy'))
        import reset_password as R
        with tempfile.TemporaryDirectory() as d:
            db = sqlite3.connect(os.path.join(d, 'dist.sqlite3'))
            db.row_factory = sqlite3.Row
            A.ensure_schema(db)
            A.register(db, 'alice', 'secret-123')
            db.close()
            with mock.patch('getpass.getpass', side_effect=['brand-new-789', 'brand-new-789']):
                self.assertEqual(R.main(['--data-dir', d, 'alice']), 0)
            with mock.patch('getpass.getpass', side_effect=['brand-new-000', 'mismatch-000']):
                with self.assertRaises(SystemExit):
                    R.main(['--data-dir', d, 'alice'])
            db = R.open_db(d)
            self.assertTrue(A.login(db, 'alice', 'brand-new-789')['token'])
            db.close()

    def test_node_bind_code_is_one_time(self):
        u = A.register(self.db, 'hank', 'secret-123')['user']
        code = A.new_bind_code(self.db, u['user_id'])['code']
        self.assertEqual(A.consume_bind_code(self.db, code, 'NODE-1'), u['user_id'])
        self.assertIsNone(A.consume_bind_code(self.db, code, 'NODE-2'))          # 一次性
        old = A.new_bind_code(self.db, u['user_id'])['code']
        self.db.execute('UPDATE node_bind_codes SET expires_at=?', (time.time() - 1,))
        self.assertIsNone(A.consume_bind_code(self.db, old, 'NODE-3'))           # 过期
        self.assertIsNone(A.consume_bind_code(self.db, 'B-NOPE', 'NODE-4'))
        cols = {r[1] for r in self.db.execute('PRAGMA table_info(nodes)')}
        self.assertIn('owner_user_id', cols)


if __name__ == '__main__':
    unittest.main()
