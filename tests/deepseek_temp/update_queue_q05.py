# -*- coding: utf-8 -*-
"""Q05 队列更新: Q05 -> sealed；sealed_items 追加 Q05。"""
import os, json, hashlib, shutil

AT = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\atlas'
QP = os.path.join(AT, 'phase_queue_v1.json')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\result'

raw = open(QP, 'rb').read()
before = hashlib.sha256(raw).hexdigest()[:8]
q = json.loads(raw.decode('utf-8-sig'))
agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))

shutil.copy2(QP, os.path.join(AT, 'phase_queue_v1_backup_pre_q05.json'))

hit = False
for item in q['queue']:
    if item['id'] == 'Q05':
        assert item['status'] == 'pending', item['status']
        item['status'] = 'sealed'
        item['sealed_at'] = '2026-10-03 Q05'
        item['seal_record'] = 'tests/deepseek/result/q05_result.json'
        item['res_sha8'] = agg['res_sha8']
        item['note'] = ('全面板 738 cells × K=16 × 4 arm（4b bf16 / 4b nf4 / 14b nf4 / 9b nf4）；'
                        '形状 %s/%s/%s；D4 精度桥 %s；独立复核见 verify_q05.txt'
                        % (agg['shape']['qwen3-4b__bf16']['shape'],
                           agg['shape']['qwen3-14b__nf4']['shape'],
                           agg['shape']['glm4-9b__nf4']['shape'],
                           'PASS' if agg['precision_bridge']['pass_'] else 'FAIL'))
        hit = True
assert hit, 'Q05 not found'
if 'Q05' not in q['sealed_items']:
    q['sealed_items'].append('Q05')
q['status_updated_at'] = '2026-10-03 Q05'
q['status_updated_by'] = 'tests/deepseek_temp/update_queue_q05.py'
json.dump(q, open(QP, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('queue', before, '->', hashlib.sha256(open(QP, 'rb').read()).hexdigest()[:8],
      '| sealed_items=%s' % ','.join(q['sealed_items']))
