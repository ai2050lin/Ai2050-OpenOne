# -*- coding: utf-8 -*-
"""下载 C-Eval（ceval/ceval-exam，HuggingFace Hub）选中科目的完整 val 集到本地，
按「知识网络 / 功能推理 / 语法规则」三个图谱大类组织，写成 data/lang_bench/ceval_val.v1.jsonl。
只在本机跑一次（或手动重跑），属于数据包的一部分，不再依赖外网。

科目映射（subject config → 图谱大类）：
  知识网络  high_school_biology / high_school_history / high_school_geography      （事实知识网络）
  功能推理  logic / civil_servant                                                  （论证与逻辑推理）
  语法规则  chinese_language_and_literature / high_school_chinese                  （汉语语法与用法）
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'lang_bench'
OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = {
    'high_school_biology': '知识网络',
    'high_school_history': '知识网络',
    'high_school_geography': '知识网络',
    'logic': '功能推理',
    'civil_servant': '功能推理',
    'chinese_language_and_literature': '语法规则',
    'high_school_chinese': '语法规则',
}
SUBJ_CN = {
    'high_school_biology': '高中生物', 'high_school_history': '高中历史',
    'high_school_geography': '高中地理', 'logic': '逻辑学',
    'civil_servant': '公务员考试', 'chinese_language_and_literature': '汉语言文学',
    'high_school_chinese': '高中语文',
}
BASE = 'https://datasets-server.huggingface.co/rows?dataset=ceval%2Fceval-exam&config={s}&split=val&offset={o}&length=100'


def fetch_subject(subj: str) -> list:
    rows, offset = [], 0
    while True:
        url = BASE.format(s=subj, o=offset)
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.loads(r.read().decode('utf-8'))
        batch = d.get('rows', [])
        if not batch:
            break
        rows.extend(b['row'] for b in batch)
        offset += len(batch)
        if offset >= int(d.get('num_rows_total', 0)):
            break
    return rows


def main() -> None:
    out_path = OUT / 'ceval_val.v1.jsonl'
    n_total, per_cat, per_subj = 0, {}, {}
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        for subj, cat in SUBJECTS.items():
            rows = fetch_subject(subj)
            for r in rows:
                opts = [r['A'], r['B'], r['C'], r['D']]
                if not r.get('question') or not all(opts) or r.get('answer') not in 'ABCD':
                    continue
                doc = {
                    'id': f'ceval-{subj}-{r["id"]}',
                    'source': 'C-Eval val', 'subject': subj, 'subject_cn': SUBJ_CN[subj],
                    'category': cat,
                    'question': r['question'].strip(),
                    'choices': opts, 'answer': r['answer'],
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                n_total += 1
                per_cat[cat] = per_cat.get(cat, 0) + 1
                per_subj[subj] = per_subj.get(subj, 0) + 1
            print(f'{subj:32s} {cat} {len(rows)} 题')
    meta = {'schema': 'lang_bench_meta.v1', 'source': 'C-Eval (ceval/ceval-exam) val split via HF datasets-server',
            'n_questions': n_total, 'by_category': per_cat, 'by_subject': per_subj,
            'subjects': {k: SUBJ_CN[k] for k in SUBJECTS}}
    (OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TOTAL', n_total, per_cat)
    print('WROTE', out_path)


if __name__ == '__main__':
    sys.exit(main())
