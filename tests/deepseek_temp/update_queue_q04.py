# -*- coding: utf-8 -*-
"""Q04 落账（队列侧）：phase_queue_v1.json 的 Q04 -> device_built（非 sealed，装置就绪非测量冻结）。"""
import os, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
QD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json')
TS = time.strftime('%Y-%m-%d %H:%M')
before = hashlib.sha256(open(QD, 'rb').read()).hexdigest()[:8]
qd = json.loads(open(QD, 'rb').read().decode('utf-8-sig'))

hit = 0
for it in qd['queue']:
    if it['id'] == 'Q04':
        it['status'] = 'device_built'
        it['device_built_at'] = TS
        it['seal_record'] = 'tests/deepseek/result/q04_smoke_result.json'
        it['prereg_design_sha'] = '33ddf69d'
        it['K'] = 16
        it['note'] = ('装置 + K=16 + 预注册已冻结；SMOKE（qwen3-4b，180 cells，K=4，900 forwards）'
                      '4/4 装置门通过，独立复核 14 PASS / 0 FAIL；全面板三模型曲线见 Q05。'
                      'E_ar 口径已登记 metric_dict v3（content 72a3d2c4）。')
        hit += 1
assert hit == 1, 'Q04 not found uniquely (%d)' % hit

qd['device_built_items'] = sorted(set(qd.get('device_built_items', []) + ['Q04']))
qd['status_updated_at'] = TS
qd['status_updated_by'] = 'tests/deepseek_temp/update_queue_q04.py'
with open(QD, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(qd, f, ensure_ascii=False, indent=1)

qd2 = json.loads(open(QD, 'rb').read().decode('utf-8-sig'))
q04 = [x for x in qd2['queue'] if x['id'] == 'Q04'][0]
print('QUEUE %s -> %s' % (before, hashlib.sha256(open(QD, 'rb').read()).hexdigest()[:8]))
print('  Q04.status=%s  device_built_at=%s  K=%s' % (q04['status'], q04['device_built_at'], q04['K']))
print('  sealed_items=%s' % qd2['sealed_items'])
print('  device_built_items=%s' % qd2['device_built_items'])
print('  next pending =', [x['id'] for x in qd2['queue'] if x['status'] == 'pending'][:3])
print('DONE')
