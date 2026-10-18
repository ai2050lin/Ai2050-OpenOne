# -*- coding: utf-8 -*-
"""ENT-XF 部件归因驱动（entangle.v1/attribution-run）：对指定因子对、读出位置与目的层做部件归因，不重算纠缠谱。

行为门与 entangle CLI 相同（文法 entangle.noun_gate / gate_probes，按属性分组）。目的层缺省为末块之后；
--dest-from-spectrum 时取纠缠谱 JSON 中该因子对在该读出位置的可靠交互比例峰值层（hs 下标即目的层：残差 = hs[0] + 前 dest 块部件）。

用法：
    python scripts/run_entangle_attribution.py --spec shared/dist_protocol/grammars/ENT-XF.json \
        --native "D:/ai2050_native/ENT-XF/TG-ENT-XF-v2/<fp>/<model>-bf16/shard-*" --behavior .../behavior.json ... \
        --readout target --pairs color:task color:lang size:task noun:lang --out ...json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import entangle as E, grammar  # noqa: E402


def peak_dest(spectrum: dict, pair: tuple, readout: str):
    """纠缠谱中该因子对在该读出位置的 E1 峰值层（hs 下标）。"""
    key = '×'.join(pair)
    best = None
    for t in spectrum.get('tags', []):
        if t.get('readout') != readout:
            continue
        p = (t.get('pairs') or {}).get(key) or {}
        v = p.get('E1')                                   # 可靠交互比例（分半可靠的交互能量占比）
        if isinstance(v, (int, float)) and (best is None or v > best[1]):
            best = (int(t['hs_index']), float(v))
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--spec', required=True)
    ap.add_argument('--native', required=True, nargs='+')
    ap.add_argument('--behavior', nargs='*', default=[])
    ap.add_argument('--readout', default='target')
    ap.add_argument('--pairs', nargs='+', required=True, help='因子对，如 color:task noun:lang')
    ap.add_argument('--dest', type=int, default=None)
    ap.add_argument('--dest-from-spectrum', default=None, help='纠缠谱 JSON：目的层取各因子对的 E1 峰值层')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    dirs = sorted({p for pat in args.native for p in glob.glob(pat)})
    field = E.NativeField(dirs)
    corr = {}
    for p in args.behavior:
        for b in json.loads(Path(p).read_text(encoding='utf-8')).get('items', []):
            corr[b['id']] = bool(b.get('correct'))
    corpus = grammar.generate(spec, include_canary=False)
    design = E.design_from_corpus(corpus, spec, field.ids, corr if args.behavior else None)
    ent = spec.get('entangle') or {}
    pairs = [tuple(p.split(':')) for p in args.pairs]
    policy = ent.get('noun_gate', 'all_probes') if args.behavior else 'none'
    groups = E.gate_groups(design, pairs, policy, ent.get('gate_probes'))
    spectrum = json.loads(Path(args.dest_from_spectrum).read_text(encoding='utf-8')) if args.dest_from_spectrum else None

    results = []
    for g in groups:
        for p in g['pairs']:
            dest, peak = args.dest, None
            if spectrum is not None:
                pk = peak_dest(spectrum, p, args.readout)
                if pk is not None:
                    dest, peak = pk[0], pk[1]
            t0 = time.time()
            r = E.attribute(field, g['design'], p, args.readout, dest, task_alignment=ent.get('task_alignment'))
            r['gate'] = g['name']
            r['dest_source'] = 'spectrum_peak' if peak is not None else ('cli' if args.dest is not None else 'default_last')
            if peak is not None:
                r['spectrum_peak_reliable_frac'] = peak
            r['seconds'] = round(time.time() - t0, 1)
            print(f'[attribution] {p} {args.readout} dest={r.get("dest")} {r["seconds"]}s '
                  f'entanglers={len(r.get("entanglers") or [])} disentanglers={len(r.get("disentanglers") or [])}', flush=True)
            results.append(r)
    out = {'schema': 'entangle.v1/attribution-run', 'native_dirs': dirs, 'spec_sha256': grammar.spec_sha(spec),
           'manifest_sha256': [hashlib.sha256((Path(d) / 'manifest.json').read_bytes()).hexdigest() for d in dirs],
           'gates': {g['name']: {k: v for k, v in g['report'].items() if k != 'dropped_nouns'} for g in groups},
           'readout': args.readout, 'results': results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=E._json_default), encoding='utf-8')
    print('[attribution] wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
