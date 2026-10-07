# -*- coding: utf-8 -*-
"""
Q05 登记: metric_dict v3 -> v4（E_ar.status: device_built -> measured；写入全面板三模型曲线 + D4 精度桥 + 形状）。
断言: metrics 数不变、E_read 逐位不变。备份 v3。
"""
import os, json, hashlib, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
AT = os.path.join(ROOT, 'research', 'deepseek', 'atlas')
MD = os.path.join(AT, 'metric_dict.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')

def content_hash(d):
    c = dict(d); c.pop('content_sha256_8', None)
    return hashlib.sha256(json.dumps(c, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]

raw = open(MD, 'rb').read()
v3_file = hashlib.sha256(raw).hexdigest()[:8]
d = json.loads(raw.decode('utf-8-sig'))
assert d['format'] == 'metric_dict_v3', d['format']
assert content_hash(d) == d['content_sha256_8'], 'v3 self-hash mismatch'
v3_content = d['content_sha256_8']
assert d['metrics'].__len__() == 7, 'metrics count drift'
E_read_before = json.dumps(d['global_kpis']['E_read'], ensure_ascii=False, sort_keys=True)
E_ar = d['global_kpis']['E_ar']
assert E_ar['status'] == 'device_built', E_ar['status']

agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))
K = agg['K']
def arm_of(m):
    for a in agg['arms']:
        if a.startswith(m + '__'):
            return a
    raise KeyError(m)
MODELS = [('qwen3-4b', arm_of('qwen3-4b')), ('qwen3-14b', arm_of('qwen3-14b')), ('glm4-9b', arm_of('glm4-9b'))]

cur = {}
for m, a in MODELS:
    cur[m] = dict(
        prec=a.split('__')[1],
        E_ar={str(k): agg['curves'][a][str(k)]['E_ar'] for k in range(K + 1)},
        E_ar_rel={str(k): agg['curves'][a][str(k)]['E_ar_rel'] for k in range(K + 1)},
        scale={str(k): agg['curves'][a][str(k)]['scale'] for k in range(K + 1)},
        shape=agg['shape'][a]['shape'], half_life_k=agg['shape'][a]['half_life_k'],
        drift_G=agg['shape'][a]['G'],
        S_rel=agg['s_rel'][a], per_arm_verdict=agg['per_arm_verdict'][a], res_sha8=agg['per_arm_res_sha8'][a],
    )
cur['precision_bridge'] = agg['precision_bridge']
cur['precision_policy'] = ('qwen3-4b=bf16（与 E_read 同精度）；'
                           'qwen3-14b/glm4-9b=nf4（16 GB 显存无法 bf16 常驻）；D4 桥约束可比性')
cur['K'] = K
cur['panel_sha8'] = agg['panel_sha8']
cur['aggregate_res_sha8'] = agg['res_sha8']

# 备份 v3
shutil.copy2(MD, os.path.join(AT, 'metric_dict_v3_backup.json'))

E_ar['status'] = 'measured'
E_ar['measured_in'] = 'Q05'
E_ar['current'] = cur
E_ar['available_precisions'] = {'qwen3-4b': ['bf16', 'nf4'], 'qwen3-14b': ['nf4'], 'glm4-9b': ['nf4']}
E_ar.pop('to_measure_in', None)

d['format'] = 'metric_dict_v4'
d['version'] = 4
d['frozen_at'] = '2026-10-03 Q05'
d['generated_by'] = 'tests/deepseek/register_q05_metric_v4.py'
d['supersedes'] = {'format': 'metric_dict_v3', 'file_sha8': v3_file, 'content_sha256_8': v3_content,
                   'backup': 'research/deepseek/atlas/metric_dict_v3_backup.json'}
d['content_sha256_8'] = content_hash(d)

# 断言: E_read 未变、metrics 数未变
assert json.dumps(d['global_kpis']['E_read'], ensure_ascii=False, sort_keys=True) == E_read_before, 'E_read mutated!'
assert len(d['metrics']) == 7

json.dump(d, open(MD, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('metric_dict v3->v4 OK  file_sha8=%s  content=%s (was %s)  E_ar.status=%s'
      % (hashlib.sha256(open(MD, 'rb').read()).hexdigest()[:8], d['content_sha256_8'], v3_content, E_ar['status']))
print('E_ar B4b shape=%s D4=%s' % (cur['qwen3-4b']['shape'], agg['precision_bridge']['pass_']))
