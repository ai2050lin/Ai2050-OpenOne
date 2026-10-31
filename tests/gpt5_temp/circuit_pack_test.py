# -*- coding: utf-8 -*-
"""circuit 附加包构建与命中验证（模拟 build_extras 的 circuit 分支 + 客户端 packHint/canonical 规则）。"""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(r'D:\AI2050\Ai2050-OpenOne')
sys.path.insert(0, str(ROOT / 'deploy'))
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault('AI2050_NATIVE_DIR', 'D:/ai2050_native')

from datapack import (EXTRA_SCHEMA, _write_extra_gz, _write_extras, canonical,
                      circuit_source, extra_file, extras_plan, pack_hint, read_extras)

out_lines = []


def P(name, ok, extra=''):
    out_lines.append(('PASS' if ok else 'FAIL') + ' | ' + name + ((' | ' + extra) if extra else ''))


data_dir = ROOT / 'distributed_data'

# 1 extras_plan 含 circuit 包
plan = [e for e in extras_plan() if e['kind'] == 'circuit']
P('extras_plan has circuit entry', len(plan) >= 1 and plan[0]['id'].startswith('circuit:'),
  str([e['id'] for e in plan]))

e = plan[0]
mk = e['model_key']
src = circuit_source(data_dir, mk)
P('circuit source file exists', src.is_file(), str(src.name))

# 2 版本 = 源哈希 + 研究版本（与 build_extras 相同）
from datapack import research_stamp
cards_version = research_stamp(ROOT / 'research')
version = hashlib.sha256(src.read_bytes()).hexdigest()[:8] + '-' + cards_version[:8]

with gzip.open(src, 'rt', encoding='utf-8') as fh:
    entries = json.load(fh).get('entries', {})
P('entries loaded', len(entries) > 0, f'{len(entries)} entries')

# 3 打包（与 build_extras 相同调用）并登记 extras.json
head = {'schema': EXTRA_SCHEMA, 'id': e['id'], 'version': version,
        'created_at': time.time(), 'notes': [f'预计算源 {src.name}']}
size, raw = _write_extra_gz(extra_file(data_dir, e['id']), head, entries)
extras = read_extras(data_dir)
extras[e['id']] = {**e, 'version': version, 'created_at': head['created_at'],
                   'n_entries': len(entries), 'bytes': size, 'raw_bytes': raw, 'notes': head['notes']}
_write_extras(data_dir, extras)
P('extra pack written', extra_file(data_dir, e['id']).is_file(), f'{size / 2**20:.1f} MB raw {raw / 2**20:.1f} MB')

# 4 客户端命中模拟：SpaceLens 的请求路径 → canonical → packHint → entries 里存在
run = json.load(gzip.open(src, 'rt', encoding='utf-8')).get('run', '')
sample_key = next(iter(entries))          # 任取一条，反解出 block/unit
from urllib.parse import parse_qs, urlsplit
q = dict(parse_qs(urlsplit(sample_key).query))
blk, unit = int(q['block'][0]), int(q['unit'][0])
client_path = f'/api/neurons/circuit?block={blk}&unit={unit}&k=5&model_key={mk}&run={run}'
hint = pack_hint(client_path)
P('pack_hint circuit', hint == f'circuit:{mk}', f'{hint}')
key = canonical(client_path)
doc = entries.get(key)
P('canonical entry hit', doc is not None, key[:90])
P('doc shape matches endpoint', bool(doc) and doc.get('available') is True and doc.get('schema') == 'neuron_circuit.v1'
  and isinstance(doc.get('upstream'), list) and isinstance(doc.get('downstream'), list),
  f"up={len(doc['upstream'])} down={len(doc['downstream'])} first={doc['upstream'][0] if doc['upstream'] else None}")

# 5 空 run 与其他 k 的请求：packHint 仍命中（但条目只在 run 一致时命中——与单元卡同约束）
hint2 = pack_hint('/api/neurons/circuit?block=1&unit=2&k=5&model_key=' + mk + '&run=')
P('pack_hint with empty run still maps', hint2 == f'circuit:{mk}', str(hint2))

# 6 与单元卡 run 一致性：附加包 run 应等于 pick_census 缺省 run
from server import neuron_census_service as NS
P('run == pick_census default', run == NS.pick_census(mk, None)['run'], run)

report = '\n'.join(out_lines)
(ROOT / 'tempdata' / 'circuit_pack_test.txt').write_text(report, encoding='utf-8')
print(report)
fails = [l for l in out_lines if l.startswith('FAIL')]
print(f"TOTAL {'PASS' if not fails else 'FAIL'}: {len(out_lines) - len(fails)}/{len(out_lines)}")
