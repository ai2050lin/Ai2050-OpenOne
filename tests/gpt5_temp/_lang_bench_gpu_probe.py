# -*- coding: utf-8 -*-
"""真实 GPU 小样：加载 Qwen3-4B，按 lang_bench 的打分方法实测 8 题，人工核对正确率。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'server'))

import lang_bench_service as lbs
import live_analysis_service as las


def main() -> int:
    d = las.model_dir()
    assert d is not None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d))
    try:
        model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, attn_implementation='eager')
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.bfloat16, attn_implementation='eager')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    las._state.update(status='ready', model=model, tokenizer=tok, device=device, dir=str(d), meta={})

    lids = lbs._letter_token_ids(tok)
    qs = lbs._load_questions()
    picks = [qs[0], qs[3], qs[20], qs[60], qs[90], qs[120], qs[140], qs[168]]
    n_ok = 0
    for q in picks:
        pred = lbs._score_one(model, tok, device, lids, q)
        ok = pred == q['answer']
        n_ok += ok
        print(f"[{q['subject_cn']}] pred={pred} ans={q['answer']} {'OK' if ok else 'MISS'}  {q['question'][:28]}…")
    print(f'{n_ok}/{len(picks)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
