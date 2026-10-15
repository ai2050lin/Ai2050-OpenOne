# -*- coding: utf-8 -*-
"""账号与会话（accounts.v1）：客户端注册 / 登录、会话令牌、角色、节点绑定码。

- 密码：PBKDF2-HMAC-SHA256（20 万次迭代、16 字节随机盐），只存哈希；比较用 hmac.compare_digest。
- 会话：32 字节随机令牌，库里只存 SHA-256；默认 30 天，使用时顺延（滑动过期）。
- 第一个注册的账号自动成为管理员（admin），其余为成员（member）。
- 注册策略 AI2050_REGISTRATION：open（默认，任何人可注册）/ closed（只有管理员能创建账号；第一个账号仍可自助注册）。
- 防暴力破解：同一用户名连续失败 5 次，锁定 5 分钟。
- 节点绑定码：登录用户生成一次性码（30 分钟），GPU 节点注册时带上，节点即归属该账号（贡献者可署名）。
只依赖标准库；调用方传入 sqlite3 连接并负责加锁。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA = 'accounts.v1'
PBKDF2_ITER = 200_000
SESSION_TTL = 30 * 24 * 3600
BIND_TTL = 30 * 60
MAX_FAILS = 5
LOCK_SECONDS = 300
USERNAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$')
ROLES = ('admin', 'member')

DDL = '''
CREATE TABLE IF NOT EXISTS users(
    user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE, display_name TEXT,
    email TEXT, pw_hash TEXT NOT NULL, pw_salt TEXT NOT NULL, pw_iter INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member', disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL, last_login REAL);
CREATE TABLE IF NOT EXISTS sessions(
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL, expires_at REAL,
    last_seen REAL, client TEXT);
CREATE TABLE IF NOT EXISTS login_failures(
    username TEXT PRIMARY KEY COLLATE NOCASE, fails INTEGER NOT NULL DEFAULT 0, locked_until REAL);
CREATE TABLE IF NOT EXISTS node_bind_codes(
    code_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL, expires_at REAL, used_by TEXT);
'''


class AuthError(Exception):
    """status: 400 输入不合法 / 401 未登录或凭据错误 / 403 无权限 / 409 冲突 / 423 锁定"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def ensure_schema(db: sqlite3.Connection) -> None:
    db.executescript(DDL)
    cols = {r[1] for r in db.execute('PRAGMA table_info(nodes)')} if _has_table(db, 'nodes') else set()
    if cols and 'owner_user_id' not in cols:
        db.execute('ALTER TABLE nodes ADD COLUMN owner_user_id TEXT')
    db.commit()


def _has_table(db, name) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def registration_policy() -> str:
    v = os.environ.get('AI2050_REGISTRATION', 'open').strip().lower()
    return v if v in ('open', 'closed') else 'open'


def _hash_pw(password: str, salt: bytes, iters: int) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iters).hex()


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _check_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < 8:
        raise AuthError(400, '密码至少 8 位')
    if len(password) > 128:
        raise AuthError(400, '密码最长 128 位')
    if password.isdigit() or password.isalpha():
        raise AuthError(400, '密码需同时包含字母和数字（或符号）')


def public_user(row) -> Dict[str, Any]:
    return {'user_id': row['user_id'], 'username': row['username'], 'display_name': row['display_name'] or row['username'],
            'email': row['email'] or '', 'role': row['role'], 'disabled': bool(row['disabled']),
            'created_at': row['created_at'], 'last_login': row['last_login']}


def user_count(db) -> int:
    return db.execute('SELECT COUNT(*) FROM users').fetchone()[0]


def status(db) -> Dict[str, Any]:
    n = user_count(db)
    policy = registration_policy()
    return {'schema': SCHEMA, 'has_users': n > 0, 'registration': policy,
            'can_register': n == 0 or policy == 'open', 'first_user_becomes_admin': n == 0}


def _new_session(db, user_id: str, client: str = '') -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.execute('INSERT INTO sessions(token_hash,user_id,created_at,expires_at,last_seen,client) VALUES(?,?,?,?,?,?)',
               (_sha(token), user_id, now, now + SESSION_TTL, now, (client or '')[:120]))
    db.execute('DELETE FROM sessions WHERE expires_at < ?', (now,))
    return token


def create_user(db, username: str, password: str, *, display_name: str = '', email: str = '',
                role: Optional[str] = None, by_admin: bool = False) -> Dict[str, Any]:
    username = (username or '').strip()
    if not USERNAME_RE.match(username):
        raise AuthError(400, '用户名 3–32 位：字母、数字、_ . -，以字母或数字开头')
    _check_password(password)
    email = (email or '').strip()[:120]
    if email and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        raise AuthError(400, '邮箱格式不正确')
    first = user_count(db) == 0
    if not first and not by_admin and registration_policy() != 'open':
        raise AuthError(403, '本中心关闭了自助注册，请联系管理员创建账号')
    if db.execute('SELECT 1 FROM users WHERE username=?', (username,)).fetchone():
        raise AuthError(409, '用户名已被使用')
    role = 'admin' if first else (role if by_admin and role in ROLES else 'member')
    salt = secrets.token_bytes(16)
    uid = 'U-' + secrets.token_hex(5)
    now = time.time()
    db.execute('INSERT INTO users(user_id,username,display_name,email,pw_hash,pw_salt,pw_iter,role,disabled,created_at,last_login) '
               'VALUES(?,?,?,?,?,?,?,?,0,?,NULL)',
               (uid, username, (display_name or '').strip()[:60] or username, email,
                _hash_pw(password, salt, PBKDF2_ITER), salt.hex(), PBKDF2_ITER, role, now))
    db.commit()
    return public_user(db.execute('SELECT * FROM users WHERE user_id=?', (uid,)).fetchone())


def register(db, username: str, password: str, *, display_name: str = '', email: str = '', client: str = '') -> Dict[str, Any]:
    user = create_user(db, username, password, display_name=display_name, email=email)
    token = _new_session(db, user['user_id'], client)
    db.execute('UPDATE users SET last_login=? WHERE user_id=?', (time.time(), user['user_id']))
    db.commit()
    return {'token': token, 'user': user, 'expires_in': SESSION_TTL}


def login(db, username: str, password: str, *, client: str = '') -> Dict[str, Any]:
    username = (username or '').strip()
    now = time.time()
    f = db.execute('SELECT fails,locked_until FROM login_failures WHERE username=?', (username,)).fetchone()
    if f and f['locked_until'] and f['locked_until'] > now:
        raise AuthError(423, f'登录失败次数过多，请 {int(f["locked_until"] - now) + 1} 秒后再试')
    row = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    ok = False
    if row is not None:
        ok = hmac.compare_digest(_hash_pw(password or '', bytes.fromhex(row['pw_salt']), int(row['pw_iter'])), row['pw_hash'])
    else:
        _hash_pw(password or '', b'\0' * 16, PBKDF2_ITER)          # 用户不存在时同样耗时，避免枚举用户名
    if not ok:
        fails = (f['fails'] if f else 0) + 1
        locked = now + LOCK_SECONDS if fails >= MAX_FAILS else None
        db.execute('INSERT INTO login_failures(username,fails,locked_until) VALUES(?,?,?) '
                   'ON CONFLICT(username) DO UPDATE SET fails=excluded.fails, locked_until=excluded.locked_until',
                   (username, 0 if locked else fails, locked))
        db.commit()
        raise AuthError(401, '用户名或密码错误' + ('（已锁定 5 分钟）' if locked else ''))
    if row['disabled']:
        raise AuthError(403, '该账号已被停用，请联系管理员')
    db.execute('DELETE FROM login_failures WHERE username=?', (username,))
    token = _new_session(db, row['user_id'], client)
    db.execute('UPDATE users SET last_login=? WHERE user_id=?', (now, row['user_id']))
    db.commit()
    return {'token': token, 'user': public_user(row), 'expires_in': SESSION_TTL}


def bearer(authorization: str) -> str:
    a = (authorization or '').strip()
    return a[7:].strip() if a.lower().startswith('bearer ') else ''


def session_user(db, token: str, *, touch: bool = True) -> Optional[Dict[str, Any]]:
    """有效会话 → 用户；无效、过期、账号停用 → None。使用时顺延过期时间。"""
    if not token:
        return None
    now = time.time()
    s = db.execute('SELECT * FROM sessions WHERE token_hash=?', (_sha(token),)).fetchone()
    if s is None or s['expires_at'] < now:
        return None
    row = db.execute('SELECT * FROM users WHERE user_id=?', (s['user_id'],)).fetchone()
    if row is None or row['disabled']:
        return None
    if touch and now - (s['last_seen'] or 0) > 300:
        db.execute('UPDATE sessions SET last_seen=?, expires_at=? WHERE token_hash=?', (now, now + SESSION_TTL, s['token_hash']))
        db.commit()
    return public_user(row)


def require_user(db, authorization: str) -> Dict[str, Any]:
    u = session_user(db, bearer(authorization))
    if u is None:
        raise AuthError(401, '未登录或会话已过期')
    return u


def require_admin(db, authorization: str) -> Dict[str, Any]:
    u = require_user(db, authorization)
    if u['role'] != 'admin':
        raise AuthError(403, '需要管理员账号')
    return u


def logout(db, token: str) -> None:
    db.execute('DELETE FROM sessions WHERE token_hash=?', (_sha(token),))
    db.commit()


def change_password(db, user_id: str, old: str, new: str, *, keep_token: str = '') -> None:
    row = db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone()
    if row is None or not hmac.compare_digest(_hash_pw(old or '', bytes.fromhex(row['pw_salt']), int(row['pw_iter'])), row['pw_hash']):
        raise AuthError(401, '原密码不正确')
    _check_password(new)
    salt = secrets.token_bytes(16)
    db.execute('UPDATE users SET pw_hash=?, pw_salt=?, pw_iter=? WHERE user_id=?',
               (_hash_pw(new, salt, PBKDF2_ITER), salt.hex(), PBKDF2_ITER, user_id))
    # 改密码后其他设备上的会话全部失效，只保留当前会话
    db.execute('DELETE FROM sessions WHERE user_id=? AND token_hash<>?', (user_id, _sha(keep_token) if keep_token else ''))
    db.commit()


def list_users(db) -> List[Dict[str, Any]]:
    out = []
    owned = {}
    if _has_table(db, 'nodes'):
        cols = {r[1] for r in db.execute('PRAGMA table_info(nodes)')}
        if 'owner_user_id' in cols:
            for r in db.execute('SELECT owner_user_id, COUNT(*) AS n FROM nodes WHERE owner_user_id IS NOT NULL GROUP BY owner_user_id'):
                owned[r['owner_user_id']] = r['n']
    for row in db.execute('SELECT * FROM users ORDER BY created_at'):
        u = public_user(row)
        u['nodes'] = owned.get(row['user_id'], 0)
        out.append(u)
    return out


def update_user(db, actor: Dict[str, Any], user_id: str, *, role: Optional[str] = None, disabled: Optional[bool] = None) -> Dict[str, Any]:
    row = db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone()
    if row is None:
        raise AuthError(404, '账号不存在')
    if role is not None and role not in ROLES:
        raise AuthError(400, '角色只能是 admin 或 member')
    demote = (role == 'member' and row['role'] == 'admin') or (disabled and row['role'] == 'admin')
    if demote:
        admins = db.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND disabled=0").fetchone()[0]
        if admins <= 1:
            raise AuthError(409, '至少保留一个可用的管理员账号')
    if user_id == actor['user_id'] and disabled:
        raise AuthError(409, '不能停用自己的账号')
    if role is not None:
        db.execute('UPDATE users SET role=? WHERE user_id=?', (role, user_id))
    if disabled is not None:
        db.execute('UPDATE users SET disabled=? WHERE user_id=?', (1 if disabled else 0, user_id))
        if disabled:
            db.execute('DELETE FROM sessions WHERE user_id=?', (user_id,))
    db.commit()
    return public_user(db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone())


# ---------------- 节点绑定码 ----------------

def new_bind_code(db, user_id: str) -> Dict[str, Any]:
    code = 'B-' + secrets.token_hex(4).upper() + '-' + secrets.token_hex(4).upper()
    now = time.time()
    db.execute('INSERT INTO node_bind_codes(code_hash,user_id,created_at,expires_at,used_by) VALUES(?,?,?,?,NULL)',
               (_sha(code), user_id, now, now + BIND_TTL))
    db.execute('DELETE FROM node_bind_codes WHERE expires_at < ?', (now - 86400,))
    db.commit()
    return {'code': code, 'expires_in': BIND_TTL}


def consume_bind_code(db, code: str, node_id: str) -> Optional[str]:
    """一次性：有效则返回 user_id 并标记已用；无效 / 过期 / 已用返回 None。"""
    if not code:
        return None
    row = db.execute('SELECT * FROM node_bind_codes WHERE code_hash=?', (_sha(code.strip()),)).fetchone()
    if row is None or row['used_by'] or row['expires_at'] < time.time():
        return None
    db.execute('UPDATE node_bind_codes SET used_by=? WHERE code_hash=?', (node_id, row['code_hash']))
    return row['user_id']
