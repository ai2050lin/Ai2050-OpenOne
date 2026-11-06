# -*- coding: utf-8 -*-
"""重建图谱测试题库（questions.v3.jsonl）：C-Eval / MMLU / M3Exam 三个公开数据集。

  知识网络 + 功能推理 + 语法规则 的题全部来自 C-Eval val（169 题，7 科目）
  MMLU test 全量 57 科目（cais/mmlu，MIT），~14.0k 题，英文 4 选 1
  M3Exam 中文 test（chiayewken/m3exam 镜像，原数据集 CC BY-NC-SA 4.0），682 题，8 科目

与 v2 的区别（这是口径变更，不是换了个文件名）：
  1. `category` 只放**三类**（知识网络 / 功能推理 / 语法规则），映射不到就写 null。
     v2 把数据集自带的题材（综合知识 / 医学知识）直接塞进同一个字段，
     于是「三类各多少题」和「题材分布」混在一列里，任何按 category 的统计都会被污染。
  2. 数据集自带的题材另存 `topic` 字段（M3Exam 的 subject_category、MMLU 的科目组）。
  3. CMMLU / CMExam 不进 v3。它们的题仍在 questions.v2.jsonl 里，没有删——
     但主口径不再混入，避免「换个数据集结论还成立吗」这件事被 4 万题的大盘稀释掉。

只在本机跑一次；产出后运行时不再依赖外网（parquet 缓存在 data/lang_bench/raw/）。
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol.bench_categories import ALL_CATS, category_of   # noqa: E402
from shared.dist_protocol.bench_sources import SOURCES as SRC_REGISTRY    # noqa: E402

OUT = ROOT / 'data' / 'lang_bench'
RAW = OUT / 'raw'
CEVAL_PATH = OUT / 'ceval_val.v1.jsonl'
QUESTIONS_PATH = OUT / 'questions.v3.jsonl'
META_PATH = OUT / 'meta.v3.json'

# 直链与 bench_sources.SOURCES[prefix]['repo'] + ['files'] 一一对应；改上游路径时两处一起改。
MMLU_URL = ('https://huggingface.co/datasets/cais/mmlu/resolve/main/'
            'all/test-00000-of-00001.parquet')
M3EXAM_URL = ('https://huggingface.co/datasets/chiayewken/m3exam/resolve/main/'
              'chinese/test-00000-of-00001-a9ef43568bebf5d0.parquet')

# 许可与拆分不再在这里写第二份：唯一真源是 shared/dist_protocol/bench_sources.py。
SOURCES = {s['id']: {'license': s['license'], 'n_note': f"{s['split']} split"}
           for s in SRC_REGISTRY.values()}

MMLU_CN = {
    'abstract_algebra': '抽象代数', 'anatomy': '解剖学', 'astronomy': '天文学',
    'business_ethics': '商业伦理', 'clinical_knowledge': '临床知识', 'college_biology': '大学生物',
    'college_chemistry': '大学化学', 'college_computer_science': '大学计算机',
    'college_mathematics': '大学数学', 'college_medicine': '大学医学', 'college_physics': '大学物理',
    'computer_security': '计算机安全', 'conceptual_physics': '概念物理',
    'econometrics': '计量经济学', 'electrical_engineering': '电气工程',
    'elementary_mathematics': '初等数学', 'formal_logic': '形式逻辑',
    'global_facts': '全球事实', 'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学', 'high_school_computer_science': '高中计算机',
    'high_school_european_history': '高中欧洲史', 'high_school_geography': '高中地理',
    'high_school_government_and_politics': '高中政治', 'high_school_macroeconomics': '高中宏观经济学',
    'high_school_mathematics': '高中数学', 'high_school_microeconomics': '高中微观经济学',
    'high_school_physics': '高中物理', 'high_school_psychology': '高中心理学',
    'high_school_statistics': '高中统计', 'high_school_us_history': '高中美国史',
    'high_school_world_history': '高中世界史', 'human_aging': '人类衰老',
    'human_sexuality': '人类性行为', 'international_law': '国际法', 'jurisprudence': '法理学',
    'logical_fallacies': '逻辑谬误', 'machine_learning': '机器学习', 'management': '管理学',
    'marketing': '市场营销', 'medical_genetics': '医学遗传学', 'miscellaneous': '杂项',
    'moral_disputes': '道德争议', 'moral_scenarios': '道德情境', 'nutrition': '营养学',
    'philosophy': '哲学', 'prehistory': '史前史', 'professional_accounting': '专业会计',
    'professional_law': '专业法律', 'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学', 'public_relations': '公共关系',
    'security_studies': '安全研究', 'sociology': '社会学', 'us_foreign_policy': '美国外交',
    'virology': '病毒学', 'world_religions': '世界宗教',
}

M3_CN = {'history': '历史', 'chinese': '语文', 'politics': '政治', 'math': '数学',
         'biology': '生物', 'chemistry': '化学', 'physics': '物理', 'geography': '地理'}

LETTERS = 'ABCDEFGH'


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f'  复用缓存 {dest.name}（{dest.stat().st_size / 1e6:.1f} MB）')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  下载 {url.rsplit("/", 1)[-1]} …')
    req = urllib.request.Request(url, headers={'User-Agent': 'ai2050-lang-bench/3.0'})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f'  写入 {dest.name}（{len(data) / 1e6:.1f} MB）')
    return dest


def load_ceval() -> list:
    if not CEVAL_PATH.exists():
        raise SystemExit(f'缺少 {CEVAL_PATH}：先运行 scripts/download_lang_bench.py')
    docs = []
    with open(CEVAL_PATH, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            docs.append({
                'id': d['id'], 'source': 'C-Eval val', 'subject': d['subject'],
                'subject_cn': d.get('subject_cn') or d['subject'], 'topic': '',
                'category': category_of('C-Eval val', d['subject']),
                'question': d['question'], 'choices': d['choices'], 'answer': d['answer'],
            })
    print(f'  C-Eval val  {len(docs)} 题')
    return docs


def load_mmlu() -> list:
    import pyarrow.parquet as pq
    t = pq.read_table(_download(MMLU_URL, RAW / 'mmlu_test.parquet'))
    cols = {k: t.column(k).to_pylist() for k in t.schema.names}
    docs = []
    for i in range(t.num_rows):
        subj = cols['subject'][i]
        ch = [str(c) for c in cols['choices'][i]]
        idx = int(cols['answer'][i])
        docs.append({
            'id': f'mmlu-{subj}-{i}', 'source': 'MMLU test', 'subject': subj,
            'subject_cn': MMLU_CN.get(subj, subj), 'topic': subj,
            'category': category_of('MMLU test', subj),
            'question': cols['question'][i], 'choices': ch,
            'answer': LETTERS[idx] if 0 <= idx < len(LETTERS) else 'A',
        })
    print(f'  MMLU test   {len(docs)} 题')
    return docs


def load_m3exam() -> list:
    import pyarrow.parquet as pq
    t = pq.read_table(_download(M3EXAM_URL, RAW / 'm3exam_zh_test.parquet'))
    cols = {k: t.column(k).to_pylist() for k in t.schema.names}
    docs = []
    for i in range(t.num_rows):
        subj = cols['subject'][i] or ''
        opts = []
        for o in (cols['options'][i] or []):
            s = str(o).strip()
            # 选项自带 "A. " 前缀：剥掉，字母由位置决定（与题库其它数据集的 choices 对齐）
            if len(s) > 2 and s[0] in LETTERS and s[1] in '.、)．':
                s = s[2:].strip()
            opts.append(s)
        ans = str(cols['answer_text'][i] or '').strip()
        letter = ans[0].upper() if ans and ans[0].upper() in LETTERS else 'A'
        docs.append({
            'id': f'm3exam-zh-{subj}-{i}', 'source': 'M3Exam zh test', 'subject': subj,
            'subject_cn': M3_CN.get(subj, subj), 'topic': cols['subject_category'][i] or '',
            'category': category_of('M3Exam zh test', subj),
            'question': (cols['background'][i] or '') + cols['question_text'][i],
            'choices': opts, 'answer': letter,
        })
    print(f'  M3Exam zh   {len(docs)} 题')
    return docs


def build_meta(docs: list) -> dict:
    """按正文统计 meta 摘要。by_dataset 是中心唯一能拿到的按数据集三类分布——
    发布包不带正文后，中心 / 数据包全靠这一段回答「MMLU 知识网络多少题」。"""
    by_source, by_cat, by_sub, unmapped = {}, {c: 0 for c in ALL_CATS}, {}, {}
    by_dataset = {}
    for d in docs:
        src = d['source']
        by_source[src] = by_source.get(src, 0) + 1
        by_sub[d['subject_cn']] = by_sub.get(d['subject_cn'], 0) + 1
        blk = by_dataset.get(src)
        if blk is None:
            info = next((s for s in SRC_REGISTRY.values() if s['id'] == src), {})
            blk = by_dataset[src] = {
                'prefix': info.get('prefix', src), 'dataset_key': info.get('dataset_key', ''),
                'repo': info.get('repo', ''), 'split': info.get('split', ''),
                'source_url': info.get('repo_url', ''), 'license': info.get('license', ''),
                'n_questions': 0, 'n_unmapped': 0,
                'by_category': {c: 0 for c in ALL_CATS}, 'subjects': {},
            }
        blk['n_questions'] += 1
        s = blk['subjects'].setdefault(d['subject'], {'key': d['subject'], 'name': d['subject_cn'], 'n': 0})
        s['n'] += 1
        if d['category']:
            by_cat[d['category']] += 1
            blk['by_category'][d['category']] += 1
        else:
            unmapped[src] = unmapped.get(src, 0) + 1
            blk['n_unmapped'] += 1

    for blk in by_dataset.values():
        blk['by_category'] = {c: n for c, n in blk['by_category'].items() if n}
        blk['subjects'] = sorted(blk['subjects'].values(), key=lambda s: -s['n'])
        blk['n_subjects'] = len(blk['subjects'])
        blk['graph_n'] = sum(blk['by_category'].values())
        blk['testable'] = blk['graph_n'] > 0

    return {
        'schema': 'lang_bench_meta.v4',
        'source': ' + '.join(f"{s['id']} ({s['repo']})" for s in SRC_REGISTRY.values()),
        'n_questions': len(docs),
        'by_source': by_source,
        'by_category': by_cat,
        'unmapped': unmapped,
        'by_subject': dict(sorted(by_sub.items(), key=lambda kv: -kv[1])),
        # 按数据集拆开的同一批统计：题量、科目、三类分布、许可与源地址。
        # 中心没有正文时按这一段重建数据集清单（shared/dist_protocol/bench_index.py）。
        'by_dataset': by_dataset,
        'licenses': {k: v['license'] for k, v in SOURCES.items()},
        'notes': {
            'category': '只有三类（知识网络 / 功能推理 / 语法规则）；映射不到为 null，不设「综合知识」兜底',
            'topic': '数据集自带的题材分组（M3Exam = subject_category、MMLU = 科目名）',
            'body': '题库正文 questions.v3.jsonl 不随发布包分发（CC BY-NC-SA / MIT 原样引用），'
                    '中心只用本文件回答数据集清单；题目按 bench_sources 里的源地址在本机取。',
            'CMMLU/CMExam': '不在 v3 主口径内；旧题仍在 questions.v2.jsonl，未删除',
        },
    }


def load_existing() -> list:
    docs = []
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def _loaders() -> dict:
    return {'ceval': lambda: load_ceval(), 'mmlu': lambda: load_mmlu(), 'm3exam': lambda: load_m3exam()}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description='重建图谱测试题库 v3 与 meta 摘要')
    ap.add_argument('--meta-only', action='store_true',
                    help='不重建正文，只按现有 questions.v3.jsonl 重算 meta.v3.json（改口径或补字段时用）')
    ap.add_argument('--only', default='', choices=['', 'ceval', 'mmlu', 'm3exam'],
                    help='只重建指定数据集，其余沿用现有正文')
    args = ap.parse_args()

    loaders = _loaders()
    if args.meta_only:
        docs = load_existing()
        if not docs:
            raise SystemExit(f'没有 {QUESTIONS_PATH}：先不带 --meta-only 跑一次')
        print(f'只重算 meta：读入现有 {len(docs)} 题')
    elif args.only:
        key = args.only
        keep = [d for d in load_existing() if key not in (d['id'] or '')]
        print(f'只重建 {key}：删除旧条目后剩 {len(keep)} 题，重新构建 …')
        docs = keep + loaders[key]()
        order = [s['id'] for s in SRC_REGISTRY.values()]
        docs.sort(key=lambda d: order.index(d['source']) if d['source'] in order else 99)
    else:
        print('重建题库 v3：')
        docs = load_ceval() + load_mmlu() + load_m3exam()

    if not args.meta_only:
        with open(QUESTIONS_PATH, 'w', encoding='utf-8', newline='\n') as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    meta = build_meta(docs)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'\n写出 {QUESTIONS_PATH.name}：{meta["n_questions"]} 题' if not args.meta_only
          else f'\n写出 {META_PATH.name}（按现有正文 {meta["n_questions"]} 题）')
    print(f'  按来源 {meta["by_source"]}')
    print(f'  按三类 {meta["by_category"]}（映射不到 {meta["unmapped"]}）')
    for src, blk in meta['by_dataset'].items():
        print(f'  {src}: {blk["n_questions"]} 题 / {blk["n_subjects"]} 科目 / 可映射 {blk["graph_n"]} 题')
    print(f'  科目 {len(meta["by_subject"])} 个；meta {META_PATH.stat().st_size / 1024:.0f} KB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
