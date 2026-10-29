# -*- coding: utf-8 -*-
"""扩充图谱测试题库（questions.v2.jsonl）：三个公开数据集组合，共约 4.2 万道中文选择题。

  知识网络  C-Eval val 7 科目（复用已下载的 ceval_val.v1.jsonl，169 题）
  综合知识  CMMLU test 全量 67 科目（ehasler/cmmlu 镜像，~11.5k 题）
  医学知识  CMExam train 随机抽样 30,000 题（fzkuji/CMExam parquet，总 54,497 题）
  功能推理 / 语法规则仍来自 C-Eval（logic / civil_servant / 汉语言文学 / 高中语文）

只在本机跑一次（或手动重跑），产出 data/lang_bench/questions.v2.jsonl + meta.json，
之后运行时不再依赖外网。抽样种子固定，可复现。"""
import io
import json
import random
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'lang_bench'
CEVAL_PATH = OUT / 'ceval_val.v1.jsonl'
QUESTIONS_PATH = OUT / 'questions.v2.jsonl'
META_PATH = OUT / 'meta.json'

CMEXAM_SAMPLE = 30000
SEED = 20261029
CMEXAM_PARQUET = ('https://huggingface.co/datasets/fzkuji/CMExam/resolve/refs%2Fconvert%2Fparquet'
                  '/default/train/0000.parquet')
CMMLU_BASE = ('https://datasets-server.huggingface.co/rows?dataset=ehasler%2Fcmmlu'
              '&config={s}&split=test&offset={o}&length=100')

CMMLU_CN = {
    'agronomy': '农学', 'anatomy': '解剖学', 'ancient_chinese': '古代汉语', 'arts': '艺术',
    'astronomy': '天文学', 'business_ethics': '商业伦理', 'chinese_civil_service_exam': '公务员考试',
    'chinese_driving_rule': '驾驶规则', 'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策', 'chinese_history': '中国历史', 'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '教师资格', 'clinical_knowledge': '临床知识',
    'college_actuarial_science': '精算学', 'college_education': '教育学',
    'college_engineering_thermodynamics': '工程热力学', 'college_medicine': '临床医学',
    'computer_science': '计算机科学', 'computer_security': '计算机安全', 'concept_physics': '概念物理',
    'construction_management': '建筑管理', 'economics': '经济学', 'education': '教育学',
    'electrical_engineering': '电气工程', 'elementary_mathematics': '初等数学', 'food_science': '食品科学',
    'genetics': '遗传学', 'global_facts': '全球事实', 'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学', 'high_school_physics': '高中物理', 'high_school_politics': '高中政治',
    'human_aging': '人类衰老', 'international_law': '国际法', 'journalisim': '新闻学',
    'jurisprudence': '法理学', 'legal_and_moral_basis': '法律与道德基础', 'logic': '逻辑学',
    'machine_learning': '机器学习', 'management': '管理学', 'marketing': '市场营销',
    'marxist_theory': '马克思主义理论', 'modern_chinese': '现代汉语', 'nutrition': '营养学',
    'philosophy': '哲学', 'political_science': '政治学', 'psychology': '心理学', 'rights': '权利',
    'security_study': '安全研究', 'sociology': '社会学', 'sports_science': '体育学',
    'traditional_chinese_medicine': '中医', 'virology': '病毒学', 'world_history': '世界历史',
    'world_religions': '世界宗教',
}


def _json_url(url: str, timeout: int = 60, retries: int = 8):
    """datasets-server 有速率限制且偶发网关错误：带 UA、限速、429/5xx 指数退避重试。"""
    import time
    req = urllib.request.Request(url, headers={'User-Agent': 'ai2050-lang-bench/2.0'})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 1.8, 30.0)
                    continue
            raise
    raise RuntimeError(f'重试后仍失败：{url}')


def load_ceval() -> list:
    """复用已下载的 C-Eval val 题目（169 题，含知识网络/功能推理/语法规则三大类）。"""
    if not CEVAL_PATH.exists():
        raise SystemExit(f'缺少 {CEVAL_PATH}：先运行 scripts/download_lang_bench.py')
    with open(CEVAL_PATH, encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()]
    print(f'ceval  {len(docs)} 题（复用现有文件）')
    return docs


def cmmlu_subjects() -> list:
    d = _json_url('https://datasets-server.huggingface.co/splits?dataset=ehasler%2Fcmmlu')
    return sorted({s['config'] for s in d['splits'] if s['split'] == 'test'})


def fetch_cmmlu() -> list:
    import time
    cache = OUT / '_cache_cmmlu.jsonl'
    if cache.exists():                              # 增量重跑：CMMLU 拉一次即可
        docs = [json.loads(l) for l in cache.read_text(encoding='utf-8').splitlines() if l.strip()]
        print(f'cmmlu  合计 {len(docs)} 题（缓存）')
        return docs
    docs, dup = [], set()
    for subj in cmmlu_subjects():
        offset, n_subj = 0, 0
        while True:
            d = _json_url(CMMLU_BASE.format(s=subj, o=offset))
            rows = d.get('rows', [])
            if not rows:
                break
            for b in rows:
                r = b['row']
                q, ans = (r.get('Question') or '').strip(), r.get('Answer', '')
                opts = [str(r.get(k, '')).strip() for k in 'ABCD']
                if not q or not all(opts) or ans not in 'ABCD':
                    continue
                key = hash((q, tuple(opts)))
                if key in dup:
                    continue
                dup.add(key)
                docs.append({'id': f'cmmlu-{subj}-{r.get("id", n_subj)}', 'source': 'CMMLU test',
                             'subject': subj, 'subject_cn': CMMLU_CN.get(subj, subj),
                             'category': '综合知识', 'question': q, 'choices': opts, 'answer': ans})
                n_subj += 1
            offset += len(rows)
            if offset >= int(d.get('num_rows_total', 0)):
                break
            time.sleep(0.6)                     # 限速礼貌间隔，避免 datasets-server 429
        print(f'cmmlu  {subj:36s} {n_subj} 题')
    print(f'cmmlu  合计 {len(docs)} 题')
    cache.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in docs), encoding='utf-8')
    return docs


def fetch_cmexam() -> list:
    import time
    import pandas as pd
    buf = io.BytesIO()
    delay = 3.0
    for attempt in range(8):                        # parquet 下载同样带退避重试
        try:
            req = urllib.request.Request(CMEXAM_PARQUET, headers={'User-Agent': 'ai2050-lang-bench/2.0'})
            with urllib.request.urlopen(req, timeout=300) as r:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    buf.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and attempt < 7:
                time.sleep(delay)
                delay = min(delay * 1.8, 30.0)
                continue
            raise
    buf.seek(0)
    df = pd.read_parquet(buf)
    print(f'cmexam parquet 行数 {len(df)}，抽样 {CMEXAM_SAMPLE}（seed={SEED}）')
    df = df.sample(n=min(CMEXAM_SAMPLE, len(df)), random_state=SEED)
    docs, dup = [], set()
    for i, row in enumerate(df.itertuples(index=False)):
        q = str(row.Question).strip()
        raw = row.Options
        if raw is None:                             # Options 是 numpy 数组，不能用 `or []`
            raw = []
        opts = [(o.get('key', ''), str(o.get('value', '')).strip())
                for o in raw if isinstance(o, dict)]
        opts = [v for _, v in sorted(opts)[:5]]                    # 只保留 A–E，最多 5 项
        ans = str(row.Answer).strip()
        if not q or len(opts) < 4 or ans not in 'ABCDE' or 'ABCDE'.index(ans) >= len(opts):
            continue
        key = hash((q, tuple(opts)))
        if key in dup:
            continue
        dup.add(key)
        docs.append({'id': f'cmexam-{i}', 'source': 'CMExam train', 'subject': 'cmexam',
                     'subject_cn': '医学考试', 'category': '医学知识',
                     'question': q, 'choices': opts, 'answer': ans})
    print(f'cmexam  合计 {len(docs)} 题')
    return docs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = load_ceval() + fetch_cmmlu() + fetch_cmexam()

    by_cat, by_sub, by_src = OrderedDict(), OrderedDict(), OrderedDict()
    seen = set()
    uniq = []
    for d in docs:                                  # 跨数据集最终去重（同题干+选项）
        key = hash((d['question'], tuple(d['choices'])))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
        by_cat[d['category']] = by_cat.get(d['category'], 0) + 1
        by_sub[d['subject_cn']] = by_sub.get(d['subject_cn'], 0) + 1
        by_src[d['source']] = by_src.get(d['source'], 0) + 1

    with open(QUESTIONS_PATH, 'w', encoding='utf-8', newline='') as f:
        for d in uniq:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    meta = {
        'schema': 'lang_bench_meta.v2',
        'source': 'C-Eval val + CMMLU test (ehasler/cmmlu) + CMExam train sample (fzkuji/CMExam)',
        'seed': SEED,
        'n_questions': len(uniq),
        'by_category': by_cat, 'by_subject': by_sub, 'by_source': by_src,
        'sample': {'CMExam train': {'total': 54497, 'sampled': CMEXAM_SAMPLE, 'seed': SEED}},
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('CATEGORY', dict(by_cat))
    print('TOTAL', len(uniq), '→', QUESTIONS_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
