# -*- coding: utf-8 -*-
"""冒烟：/api/neurons/circuit 权重级上下游电路（flat 导入，不依赖包路径）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.neuron_circuit_service as ncs

app = FastAPI()
app.include_router(ncs.router)
c = TestClient(app)


def main():
    # 1) 结构守卫：块越界 → available False（不触发模型加载，先测）
    r = c.get('/api/neurons/circuit', params={'block': 999, 'unit': 0})
    assert r.status_code == 200, r.text
    d = r.json()
    print('guard(bad block):', d['available'], d.get('reason', '')[:40])

    # 2) 真实电路：块 20 · 单元 0（会触发模型加载，可能 503 → available False + reason）
    r = c.get('/api/neurons/circuit', params={'block': 20, 'unit': 0, 'k': 5})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['schema'] == 'neuron_circuit.v1'
    print('circuit available:', d['available'], d.get('reason', ''))
    if d['available']:
        for side in ('upstream', 'downstream'):
            rows = d[side]
            assert len(rows) == 5, (side, len(rows))
            for h in rows:
                assert set(h) >= {'unit', 'score', 'via', 'block'}
                assert h['via'] in ('direct', 'head')
                assert (h['head'] is not None) == (h['via'] == 'head')
                assert h['score'] > 0
            best = rows[0]
            print(f"{side}: top B{best['block']}·U{best['unit']} score={best['score']} via={best['via']}"
                  f"{'#' + str(best['head']) if best.get('head') is not None else ''}"
                  f"{' · ' + best['type_name'] if best.get('type_name') else ''}")
    print('SMOKE PASS')


if __name__ == '__main__':
    main()
