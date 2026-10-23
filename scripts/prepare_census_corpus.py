# -*- coding: utf-8 -*-
"""MLP 神经元普查（mlp_census.v1）的自然语料准备：从 Hugging Face 公开数据集按行组流式读取，分词成 token 流。

每个来源取前若干篇文档（按文件中的顺序），每篇最多 doc_cap 个 token，累计到 max_tokens 为止；文档之间插入
<|endoftext|>。产物写在节点本地（不上传、不进仓库）：
    <AI2050_NATIVE_DIR>/_census/corpus/<语料指纹>/tokens_<source>.npz（tokens int32、doc_starts int64）+ manifest.json
语料指纹 = 各来源 token 流 sha256 依次拼接后的 sha256；manifest 记录数据集修订号、文件、读取的行组与计数。

用法：python scripts/prepare_census_corpus.py --tokenizer models/hf/qwen3-4b --max-tokens 5000000
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SOURCES = [
    {'source': 'fineweb-edu-en', 'repo': 'HuggingFaceFW/fineweb-edu', 'files': ['sample/10BT/000_00000.parquet'], 'lang': 'en',
     'license': 'ODC-By'},
    {'source': 'wikipedia-en', 'repo': 'wikimedia/wikipedia', 'files': ['20231101.en/train-00000-of-00041.parquet'], 'lang': 'en',
     'license': 'CC BY-SA 3.0'},
    {'source': 'fineweb-edu-zh', 'repo': 'opencsg/Fineweb-Edu-Chinese-V2.1', 'files': ['4_5/000000.parquet', '4_5/000001.parquet'],
     'lang': 'zh', 'license': 'Apache-2.0'},
    {'source': 'wikipedia-zh', 'repo': 'wikimedia/wikipedia', 'files': ['20231101.zh/train-00000-of-00006.parquet'], 'lang': 'zh',
     'license': 'CC BY-SA 3.0'},
]


def main(argv=None) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--max-tokens', type=int, default=5_000_000, help='每个来源的 token 上限')
    ap.add_argument('--doc-cap', type=int, default=2048, help='每篇文档最多取的 token 数')
    ap.add_argument('--root', default=os.path.join(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'), '_census', 'corpus'))
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok.convert_tokens_to_ids('<|endoftext|>')
    api, fs = HfApi(), HfFileSystem()
    tmp = Path(args.root) / '_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    out_sources = []
    t0 = time.time()
    for s in SOURCES:
        rev = api.dataset_info(s['repo']).sha
        toks, starts, n_docs, used = [], [], 0, []
        total = 0
        for fn in s['files']:
            if total >= args.max_tokens:
                break
            path = f"datasets/{s['repo']}@{rev}/{fn}"
            with fs.open(path, 'rb', block_size=8 * 1024 * 1024) as fh:
                pf = pq.ParquetFile(fh)
                col = 'text' if 'text' in pf.schema_arrow.names else pf.schema_arrow.names[0]
                for rg in range(pf.num_row_groups):
                    if total >= args.max_tokens:
                        break
                    texts = pf.read_row_group(rg, columns=[col]).column(0).to_pylist()
                    used.append({'file': fn, 'row_group': rg, 'rows': len(texts)})
                    for text in texts:
                        if total >= args.max_tokens:
                            break
                        if not text or not text.strip():
                            continue
                        ids = tok(text, add_special_tokens=False)['input_ids'][:args.doc_cap]
                        ids = ids[:args.max_tokens - total]
                        starts.append(total)
                        toks.extend(ids)
                        toks.append(sep)
                        total += len(ids) + 1
                        n_docs += 1
        arr = np.asarray(toks, dtype=np.int32)
        buf = io.BytesIO()
        np.savez(buf, tokens=arr, doc_starts=np.asarray(starts, dtype=np.int64))
        data = buf.getvalue()
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        (tmp / f"tokens_{s['source']}.npz").write_bytes(data)
        out_sources.append({**{k: s[k] for k in ('source', 'repo', 'lang', 'license')}, 'revision': rev, 'row_groups': used,
                            'n_docs': n_docs, 'n_tokens': int(len(arr)), 'tokens_sha256': sha,
                            'file': f"tokens_{s['source']}.npz", 'file_sha256': hashlib.sha256(data).hexdigest()})
        print(f"[corpus] {s['source']}: {n_docs} 篇 · {len(arr)} token · 行组 {len(used)} · {time.time() - t0:.0f}s", flush=True)
    fp = hashlib.sha256(''.join(x['tokens_sha256'] for x in out_sources).encode()).hexdigest()[:16]
    dest = Path(args.root) / fp
    dest.mkdir(parents=True, exist_ok=True)
    for x in out_sources:
        os.replace(tmp / x['file'], dest / x['file'])
    manifest = {'schema': 'census_corpus.v1', 'fingerprint': fp, 'tokenizer': str(Path(args.tokenizer).as_posix()),
                'separator': '<|endoftext|>', 'separator_id': int(sep), 'max_tokens_per_source': args.max_tokens,
                'doc_cap': args.doc_cap, 'sources': out_sources,
                'note': '每个来源按文件顺序取前若干篇文档；文体以教育筛选网页与百科为主，不含对话、代码与广告页'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[corpus] fingerprint', fp, dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
