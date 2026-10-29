# -*- coding: utf-8 -*-
"""lang_bench_service 冒烟：数据集读取、C-Eval 提示格式、选项字母 token 候选、
overview/status 端点（不起真模型，不打 GPU）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'server'))


def main() -> int:
    import lang_bench_service as lbs

    # 1) 数据集（题库规模自适应：v1=169 仅 C-Eval；v2=~42k C-Eval+CMMLU+CMExam）
    qs = lbs._load_questions()
    assert len(qs) >= 169, len(qs)
    cats = {}
    for q in qs:
        assert q['category'] in lbs.CATEGORIES, q['category']
        assert q['answer'] in 'ABCDE'
        assert 4 <= len(q['choices']) <= 5
        cats[q['category']] = cats.get(q['category'], 0) + 1
    print('questions ok:', len(qs), cats)

    # 2) 提示格式
    p = lbs._build_prompt(qs[0])
    assert p.startswith('以下是中国关于'), p[:40]
    assert p.rstrip().endswith('答案：'), p[-20:]
    print('prompt ok:', p[:50].replace('\n', ' / '), '…')

    # 3) 字母 token 候选（用本机 tokenizer 真实验证）
    import live_analysis_service as las
    d = las.model_dir()
    assert d is not None, '本机没有模型目录，无法验证 tokenizer'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d))
    lids = lbs._letter_token_ids(tok)
    for letter in 'ABCD':
        assert lids[letter], f'{letter} 没有单 token 候选'
    print('letter ids ok:', {k: v for k, v in lids.items()})

    # 4) 端点（模型未加载状态）
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(lbs.router)
    c = TestClient(app)
    r = c.get('/api/lang/bench/overview')
    assert r.status_code == 200, r.text
    ov = r.json()
    assert ov['available'] is True and ov['meta']['n_questions'] == len(qs)
    print('overview ok: model_status =', ov['model_status'], '| last_result =', bool(ov['last_result']))
    r = c.get('/api/lang/bench/status')
    assert r.status_code == 200 and r.json()['state'] in ('idle', 'loading', 'running', 'done', 'error')
    print('status ok:', r.json()['state'])

    # 5) 结果文件（若已有历史结果则校验 schema）
    if lbs.RESULTS_PATH.exists():
        res = json.loads(lbs.RESULTS_PATH.read_text(encoding='utf-8'))
        assert res['schema'] == 'lang_bench_result.v1'
        assert abs(res['acc'] - res['correct'] / res['n_questions']) < 1e-3
        print('existing result ok:', res['correct'], '/', res['n_questions'], '=', res['acc'])
    print('SMOKE PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
