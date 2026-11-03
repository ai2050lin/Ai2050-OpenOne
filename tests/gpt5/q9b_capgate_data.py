# -*- coding: utf-8 -*-
"""Q9-B W6 能力保持门 · 四数据集电池构建脚本（预注册增补 A1 配套）

输出: data/lang_bench/q9b_capgate.v1.jsonl + q9b_capgate_manifest.v1.json
四个数据集（互斥，统一 schema 对齐 lang_bench）:
  1. C-Eval val        —— 本地已有 ceval_val.v1.jsonl（169 题），确定性子采样 n=100
  2. MMLU test         —— HF cais/mmlu 'all' test，确定性子采样 n=100
  3. M3Exam            —— HF 镜像 chiayewken/m3exam，zh+en dev 全量（文本题）+ test 子样各 50
  4. 本地测试数据集     —— 项目池 questions.v2.jsonl 剔除 C-Eval-val 重叠 id，确定性子采样 n=100

统一 schema: {id, source, subject, subject_cn, category, question, choices(4), answer(A-D)}
子采样 rng = default_rng(20261102)，消费序固定：ceval → mmlu → m3exam(dev zh, dev en, test zh, test en) → pool
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "data" / "lang_bench"
SEED = 20261102
N_PER = 100
M3_TEST_PER_LANG = 50

items = []
seen_ids = set()


def add(**kw):
    iid = kw["id"]
    if iid in seen_ids:
        raise AssertionError(f"duplicate id {iid}")
    if len(kw["choices"]) != 4 or kw["answer"] not in ("A", "B", "C", "D") or len(kw["answer"]) != 1:
        return False  # 非常规题型（如 5 选项）跳过
    seen_ids.add(iid)
    items.append(kw)
    return True


def clean_option(o):
    s = str(o).strip()
    s = re.sub(r"^\(?[A-D][.、)．]\s*", "", s)
    return s


def img_free(t):
    return not re.search(r"<image|如图|下图|图片|图像|images?/", t)


rng = np.random.default_rng(SEED)

# ---------- 1. C-Eval val（本地） ----------
ceval = [json.loads(x) for x in (LB / "ceval_val.v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
n_ceval = 0
for i in rng.permutation(len(ceval)):
    if n_ceval >= N_PER:
        break
    e = ceval[int(i)]
    if add(id=e["id"], source="C-Eval val (local)", subject=e["subject"], subject_cn=e.get("subject_cn", ""),
           category=e.get("category", "综合知识"), question=e["question"], choices=e["choices"], answer=e["answer"]):
        n_ceval += 1

# ---------- 2. MMLU test ----------
from datasets import load_dataset

mmlu = load_dataset("cais/mmlu", "all", split="test")
perm = rng.permutation(len(mmlu))
taken = 0
for i in perm:
    ex = mmlu[int(i)]
    if len(ex["choices"]) != 4:
        continue
    add(id=f"mmlu-{ex['subject']}-{int(i)}", source="MMLU test (cais/mmlu)", subject=ex["subject"], subject_cn="",
        category="综合知识", question=ex["question"], choices=[str(c) for c in ex["choices"]],
        answer="ABCD"[int(ex["answer"])])
    taken += 1
    if taken >= N_PER:
        break
n_mmlu = len(items) - n_ceval

# ---------- 3. M3Exam（zh+en，dev 全量文本题 + test 子样） ----------
n_m3_dev = 0
for lang in ["chinese", "english"]:
    dev = load_dataset("parquet", data_files=f"hf://datasets/chiayewken/m3exam/{lang}/dev-*.parquet", split="train")
    for j, ex in enumerate(dev):
        opts = [clean_option(o) for o in ex["options"]]
        ans = str(ex["answer_text"]).strip()
        if len(opts) != 4 or ans not in ("A", "B", "C", "D"):
            continue
        blob = str(ex["question_text"]) + str(ex.get("background") or "") + " ".join(opts)
        if not img_free(blob):
            continue
        add(id=f"m3exam-{lang}-dev-{j}", source=f"M3Exam {lang} dev (chiayewken/m3exam)",
            subject=str(ex["subject"]), subject_cn="", category="综合知识",
            question=str(ex["question_text"]), choices=opts, answer=ans)
        n_m3_dev += 1

n_m3_test = 0
for lang in ["chinese", "english"]:
    test = load_dataset("parquet", data_files=f"hf://datasets/chiayewken/m3exam/{lang}/test-*.parquet", split="train")
    cand = []
    for j, ex in enumerate(test):
        opts = [clean_option(o) for o in ex["options"]]
        ans = str(ex["answer_text"]).strip()
        if len(opts) != 4 or ans not in ("A", "B", "C", "D"):
            continue
        blob = str(ex["question_text"]) + str(ex.get("background") or "") + " ".join(opts)
        if not img_free(blob):
            continue
        cand.append(j)
    pick = rng.permutation(len(cand))[:M3_TEST_PER_LANG]
    for i in sorted(pick):
        ex = test[int(cand[int(i)])]
        opts = [clean_option(o) for o in ex["options"]]
        ans = str(ex["answer_text"]).strip()
        add(id=f"m3exam-{lang}-test-{int(cand[int(i)])}", source=f"M3Exam {lang} test (chiayewken/m3exam, 镜像答案未经验证)",
            subject=str(ex["subject"]), subject_cn="", category="综合知识",
            question=str(ex["question_text"]), choices=opts, answer=ans)
        n_m3_test += 1

# ---------- 4. 本地测试数据集（questions.v2 剔除 C-Eval-val 重叠） ----------
pool = [json.loads(x) for x in (LB / "questions.v2.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
pool = [p for p in pool if not p["id"].startswith("ceval-")]
n_pool = 0
for i in rng.permutation(len(pool)):
    if n_pool >= N_PER:
        break
    e = pool[int(i)]
    if add(id="poolq-" + e["id"], source="lang_bench questions.v2 (local, 剔除 C-Eval val 重叠)",
           subject=e["subject"], subject_cn=e.get("subject_cn", ""), category=e.get("category", "综合知识"),
           question=e["question"], choices=e["choices"], answer=e["answer"]):
        n_pool += 1

# ---------- 落盘 ----------
out = LB / "q9b_capgate.v1.jsonl"
with open(out, "w", encoding="utf-8", newline="") as f:
    for e in items:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

manifest = {
    "schema": "q9b_capgate.v1",
    "seed": SEED,
    "n_total": len(items),
    "counts": {"C-Eval val": n_ceval, "MMLU test": n_mmlu, "M3Exam dev": n_m3_dev, "M3Exam test": n_m3_test,
               "local pool": n_pool},
    "datasets": ["C-Eval val", "MMLU test", "M3Exam", "lang_bench questions.v2 (local)"],
    "notes": [
        "M3Exam 镜像 chiayewken/m3exam；官方 test 答案未公开，镜像答案未经验证——门判据只用 base vs edited 配对比较",
        "M3Exam 已过滤图像引用题与非常规答案；options 去除 'A. ' 前缀",
        "本地池剔除 ceval- 前缀 id 保证与 C-Eval val 互斥；CMMLU 不在四个数据集内（用户指定四集）",
        "子采样 rng=default_rng(20261102)，消费序见脚本头部；n=100 各集 + M3Exam test 每语言 50",
    ],
    "script": "tests/gpt5/q9b_capgate_data.py",
}
mpath = LB / "q9b_capgate_manifest.v1.json"
mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="")

print(f"TOTAL={len(items)} ceval={n_ceval} mmlu={n_mmlu} m3_dev={n_m3_dev} m3_test={n_m3_test} pool={n_pool}")
print("written:", out, mpath)
