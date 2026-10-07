# -*- coding: utf-8 -*-
"""就地清理 _infra/memo_baseline.json：移除与当前 tag 同名的历史自条目（幂等重跑残留）。
校验：(a) 移除项 tag == 当前 tag；(b) 移除后历史 tag 唯一；(c) 当前基线字段不变。
"""
import io
import json
import hashlib
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\memo_baseline.json'
b = json.load(io.open(P, encoding='utf-8'))
tag = b['tag']
hist = b.get('history') or []
before = len(hist)
hist2 = [h for h in hist if h.get('tag') != tag]
removed = before - len(hist2)
tags = [h['tag'] for h in hist2]
assert len(tags) == len(set(tags)), '历史 tag 仍有重复: %s' % tags
b['history'] = hist2
io.open(P, 'w', encoding='utf-8', newline='\n').write(json.dumps(b, ensure_ascii=False, indent=1))

b2 = json.load(io.open(P, encoding='utf-8'))
print('current tag      =', b2['tag'])
print('bytes/sha8       =', b2['bytes'], b2['sha8'])
print('history %d -> %d (removed %d self-entry)' % (before, len(b2['history']), removed))
print('history tags     =', [h['tag'] for h in b2['history']])

# 交叉核对：当前基线必须等于磁盘上 MEMO 的真实状态
MEN = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
mb = open(MEN, 'rb').read()
ok_b = (len(mb) == b2['bytes'])
ok_s = (hashlib.sha256(mb).hexdigest()[:8] == b2['sha8'])
print('MEMO on disk     = %d B sha8 %s' % (len(mb), hashlib.sha256(mb).hexdigest()[:8]))
print('baseline match   = bytes:%s sha8:%s' % (ok_b, ok_s))
assert ok_b and ok_s, '基线与磁盘 MEMO 不一致'
print('FIX OK')
