# -*- coding: utf-8 -*-
"""修正 disk_verify_phase11.py 的两处错误期望（键类型 / v2 备份路径），并加一条备份内容断言。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase11\disk_verify_phase11.py'
s = io.open(P, encoding='utf-8').read()
PAIRS = [
    ("    ('baseline phase_headings == 11', mb.get('phase_headings') == 11),",
     "    ('baseline phase_headings == 11', len(mb.get('phase_headings', [])) == 11),\n"
     "    ('baseline phase_headings 为 11 个行号', isinstance(mb.get('phase_headings'), list) and mb['phase_headings'][-1] == 2305),"),
    ("    ('Ledger 备份存在', os.path.exists(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger_backup_pre_phase11.json'))),",
     "    ('Ledger 备份存在 (Phase11 目录, v2 约定)', os.path.exists(os.path.join(T11, 'atlas_ledger_backup_pre_phase11.json'))),\n"
     "    ('Ledger 备份为 pre-append 293 条', len(rd(os.path.join(T11, 'atlas_ledger_backup_pre_phase11.json'))['measurements']) == 293),"),
]
for old, new in PAIRS:
    assert s.count(old) == 1, 'count!=1 %r' % (old[:60],)
    s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='').write(s)
back = io.open(P, encoding='utf-8').read()
for old, new in PAIRS:
    assert back.count(new) == 1 and old not in back, 'readback fail %r' % (new[:60],)
print('PATCH OK: 2 处期望修正 + 1 条新断言')
