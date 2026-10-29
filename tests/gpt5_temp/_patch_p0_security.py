# -*- coding: utf-8 -*-
"""P0-7b/7c 安全补丁：ADMIN_TOKEN 去默认值 + grammar/preview 加鉴权。"""
import io

P = 'deploy/distributed_service.py'
s = io.open(P, encoding='utf-8').read()

# 1) import secrets
old1 = 'import re\nimport shutil'
new1 = 'import re\nimport secrets\nimport shutil'
assert old1 in s, 'anchor1 missing'
s = s.replace(old1, new1, 1)

# 2) ADMIN_TOKEN 默认值
old2 = "ADMIN_TOKEN = os.environ.get('AI2050_DIST_ADMIN_TOKEN', 'ai2050-admin-dev')"
new2 = (
    "# 管理令牌：生产必须显式设置 AI2050_DIST_ADMIN_TOKEN（center_app 入口强制）。\n"
    "# 未设置时（本机开发/临时进程）生成一次性随机令牌并打印到日志——不使用固定默认值，\n"
    "# 防止绕过 center_app 直接启动的服务带着可猜测的开发令牌上线（client_architecture_review P0-3）。\n"
    "ADMIN_TOKEN = os.environ.get('AI2050_DIST_ADMIN_TOKEN', '')\n"
    "if not ADMIN_TOKEN:\n"
    "    ADMIN_TOKEN = secrets.token_hex(16)\n"
    "    print('[distributed_service] AI2050_DIST_ADMIN_TOKEN 未设置：本次进程使用随机管理令牌 '\n"
    "          f'{ADMIN_TOKEN}（管理员账号会话同样可用；写进环境变量可固定）', flush=True)"
)
assert old2 in s, 'anchor2 missing'
s = s.replace(old2, new2, 1)

# 3) grammar/preview 鉴权
old3 = (
    "@router.post('/grammar/preview')\n"
    "def grammar_preview(b: GrammarPreviewIn) -> dict:\n"
    "    _proto()\n"
    "    return _grammar.preview(b.spec, dims=b.dims)"
)
new3 = (
    "@router.post('/grammar/preview')\n"
    "def grammar_preview(b: GrammarPreviewIn, x_admin_token: str = Header(default='')) -> dict:\n"
    '    """模板预览有真实计算成本（生成式展开+功效近似），与其它管理写操作同级鉴权。"""\n'
    "    _auth_admin(x_admin_token)\n"
    "    _proto()\n"
    "    return _grammar.preview(b.spec, dims=b.dims)"
)
assert old3 in s, 'anchor3 missing'
s = s.replace(old3, new3, 1)

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('patched ok: secrets import + ADMIN_TOKEN random fallback + preview auth')
