# -*- coding: utf-8 -*-
"""原生神经元选择性（neurons.v1）驱动：读取节点本地的 MLP 中间单元，对一个因子对照做选择性统计并写出 JSON。

用法：
    python scripts/run_neuron_selectivity.py --family GR-Q --tm TG-GR-Q --version 9 --model-key 6be1034f0b55a6df-bf16 \
        --factor mood --hi interrog --lo decl --model-path models/hf/qwen3-4b --out research/neurons/GR-Q-v9-qwen3-4b-mood.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import neurons as N  # noqa: E402

NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--family', required=True)
    ap.add_argument('--tm', required=True)
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--factor', required=True)
    ap.add_argument('--hi', required=True)
    ap.add_argument('--lo', default=None)
    ap.add_argument('--readouts', nargs='*', default=None)
    ap.add_argument('--model-path', default=None, help='权重目录：给出时读取 down_proj 权重，报告残差贡献排序与分解')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    dirs = sorted(glob.glob(str(NATIVE / args.family / f'{args.tm}-v{args.version}' / '*' / args.model_key / 'shard-*')))
    if not dirs:
        raise SystemExit('没有找到原生全场目录')
    corpus = json.loads((ROOT / 'distributed_data/bundles' / args.tm / f'v{args.version}' / 'corpus.json').read_text(encoding='utf-8'))
    items = {it['id']: it for it in corpus['items'] if not it.get('canary')}
    field = N.MlpField(dirs)
    down = N.down_from_checkpoint(args.model_path) if args.model_path else None
    res = N.selectivity(field, items, args.factor, args.hi, args.lo, readouts=args.readouts, down=down)
    res.update({'family': args.family, 'tm_id': args.tm, 'version': args.version, 'model_key': args.model_key,
                'native_dirs': dirs, 'd_ff': field.d_ff,
                'mlp_manifest_sha256': [hashlib.sha256((Path(d) / 'mlp' / 'mlp.json').read_bytes()).hexdigest() for d in dirs],
                'seconds': round(time.time() - t0, 1)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in (args.readouts or field.readouts):
        tags = [t for t in res['tags'] if t['readout'] == r]
        print(f'== {r}: 入选单元数（按块）', ' '.join(f"B{t['block']}:{t['n_selected']}" for t in tags))
        print('   阈值 |d|（族错误零分布 99 分位）', ' '.join(f"{t['null_max_d']:.2f}" for t in tags[::6]))
        if res.get('contribution'):
            print('   入选单元占残差差异份额', ' '.join(f"B{t['block']}:{t['share_selected']:.2f}" for t in tags[::3]))
    print('[neurons]', args.out, res['seconds'], 's')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
