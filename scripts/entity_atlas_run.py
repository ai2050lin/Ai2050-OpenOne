# -*- coding: utf-8 -*-
"""EA-v1：实体编码图谱 G1（层级与差分）+ G2（共享属性算子）的预注册、采集与分析。

  prereg   写出 research/entity_atlas/EA-v1.prereg.json（材料、句式、行为门、判据、切分）与 sha256；已存在且内容不同则拒绝。
  collect  对每个实体（ENT-XF 词表 v2，210 个）：
             行为门——颜色（完整答案强制选择：可接受 = 主色 ∪ 备选色的全部表面形式，对照 = 其余调色板颜色）与
                     大小（更大 / 更小两组表面形式），中英各一次；问法沿用 ENT-XF v2（Phase 250 复核过的问答式）；
             编码——3 个中性句式 × 中英，实体在句末，取最后一个 token 在全部 hs 层的残差，3 个句式取平均。
           产物写在节点本地 <AI2050_NATIVE_DIR>/_entity_atlas/<model_key>/EA-v1/（codes.npz + gate.json + manifest.json）。
  analyze  shared/dist_protocol/entity_atlas.py 的固定判据，报告写到 research/entity_atlas/EA-v1-<模型>/report.json。

用法：python scripts/entity_atlas_run.py prereg|collect|analyze --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import entity_atlas as E  # noqa: E402

LEXICON = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv'
PREREG = ROOT / 'research/entity_atlas/EA-v1.prereg.json'
FRAMES = {
    'zh': ['我今天想到了{X}', '这一段说的是{X}', '请看这个{X}'],
    'en': ['Today I thought about the {E}', 'This passage is about the {E}', 'Please look at this {E}'],
}


def lexicon():
    rows = []
    for line in LEXICON.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split('\t')
        rows.append({'id': f[0], 'cat': f[1], 'super': f[3], 'zh': f[4], 'en': f[5], 'color': f[6],
                     'color_alt': [c for c in f[7].split(',') if c], 'size': f[8]})
    return rows


def prereg_doc():
    import build_ent_grammar as B
    return {
        'id': 'EA-v1', 'schema': 'entity_atlas_prereg.v1', 'created': '2026-11-01',
        'question': 'G1：实体编码是否由「大类 + 类别 + 个体差分」构成（层级共享、差分低维），并能预测未见实体的类别？'
                    'G2：颜色、大小是否由跨实体共享的线性算子读出（超出类别与 token 身份基线），算子是否落在个体差分里、'
                    '能否跨语言复用、需要多少维？中英两套编码的实体间相对结构是否一致（RSA）？',
        'material': {'lexicon': str(LEXICON.relative_to(ROOT)).replace('\\', '/'),
                     'lexicon_sha256': hashlib.sha256(LEXICON.read_bytes()).hexdigest(), 'n_entities': len(lexicon())},
        'frames': FRAMES, 'readout': 'last token of the entity name (entity at the end of the prompt); mean over 3 frames',
        'gate': {'color_prompt': {'zh': B.COLOR_PROMPT_ZH, 'en': B.COLOR_PROMPT_EN},
                 'color_forms': {'zh': B.COLOR_FORMS_ZH, 'en': B.COLOR_FORMS_EN},
                 'size_prompt': {'zh': '跟成年人相比，{X}的个头通常', 'en': 'Compared with an adult human, the {E} is usually'},
                 'size_forms': {f'{k[0]}:{k[1]}': v for k, v in B.SIZE_FORMS.items()},
                 'rule': '完整答案对数概率强制选择：max(可接受) > max(对照)；颜色可接受 = 主色 ∪ 备选色，对照 = 其余调色板颜色（不含 transparent / amber）；'
                         '颜色分析只用两种语言都过门的实体，大小同理'},
        'criteria': E.CRITERIA,
        'analysis': 'G1 / G2 的定义见 shared/dist_protocol/entity_atlas.py 文件头；层由 discovery 5 折交叉验证选定，heldout 只评估一次',
    }


def do_prereg():
    doc = prereg_doc()
    text = json.dumps(doc, ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：不得在看到结果后修改，改动须升版本（EA-v2）')
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    (PREREG.with_suffix('.sha256')).write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def native_dir(key):
    return Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native')) / '_entity_atlas' / key / 'EA-v1'


def do_collect(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol.census_runner import model_snapshot
    torch.set_grad_enabled(False)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    snap = model_snapshot(model_path)['sha256']
    key = snap[:16] + '-bf16'
    out = native_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    t0 = time.time()

    def lp(prompt, answers):
        p = enc(prompt)
        res = {}
        for a in answers:
            ai = enc(a)
            lg = model(input_ids=torch.as_tensor([p + ai], device='cuda')).logits[0].float()
            res[a] = float(torch.log_softmax(lg[len(p) - 1:-1], -1).gather(-1, torch.as_tensor(ai, device='cuda')[:, None]).sum())
        return res
    g = pre['gate']
    rows = lexicon()
    gate = []
    for r in rows:
        rec = {'id': r['id']}
        for lang, name in (('zh', r['zh']), ('en', r['en'])):
            forms = g['color_forms'][lang]
            ok = {r['color'], *r['color_alt']}
            pr = g['color_prompt'][lang].replace('{X}', name).replace('{E}', name)
            cand = sorted({f for c, fs in forms.items() if c in ok or c not in ('transparent', 'amber') for f in fs})
            s = lp(pr, cand)
            acc = max(s[f] for c, fs in forms.items() if c in ok for f in fs if f in s)
            con = max(s[f] for c, fs in forms.items() if c not in ok and c not in ('transparent', 'amber') for f in fs)
            rec[f'color_{lang}'] = acc - con
            sz = g['size_forms']
            want, other = (('L', 'S') if r['size'] == 'L' else ('S', 'L'))
            ps = g['size_prompt'][lang].replace('{X}', name).replace('{E}', name)
            s2 = lp(ps, sz[f'{lang}:{want}'] + sz[f'{lang}:{other}'])
            rec[f'size_{lang}'] = max(s2[f] for f in sz[f'{lang}:{want}']) - max(s2[f] for f in sz[f'{lang}:{other}'])
        gate.append(rec)
    n_layers = model.config.num_hidden_layers + 1
    codes = {}
    for lang in ('zh', 'en'):
        arr = np.zeros((len(rows), n_layers, model.config.hidden_size), dtype=np.float32)
        for i, r in enumerate(rows):
            acc = 0
            for fr in pre['frames'][lang]:
                text = fr.replace('{X}', r['zh']).replace('{E}', r['en'])
                o = model(input_ids=torch.as_tensor([enc(text)], device='cuda'), output_hidden_states=True)
                hs = torch.stack([h[0, -1].float() for h in o.hidden_states])     # hs[-1] 已过最终归一化（HF 约定）
                acc = acc + hs.cpu().numpy()
            arr[i] = acc / len(pre['frames'][lang])
        codes[lang] = arr
    np.savez_compressed(out / 'codes.npz', zh=codes['zh'].astype(np.float16), en=codes['en'].astype(np.float16))
    (out / 'gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=1), encoding='utf-8')
    man = {'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'n_entities': len(rows), 'n_layers_hs': n_layers, 'dtype': 'bf16 forward, float16 storage',
           'hs_last_is_final_norm': True, 'torch': torch.__version__, 'seconds': round(time.time() - t0, 1)}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'out': str(out), **man}, ensure_ascii=False))


def do_analyze(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap = model_snapshot(model_path)['sha256']
    src = native_dir(snap[:16] + '-bf16')
    z = np.load(src / 'codes.npz')
    gate = {g['id']: g for g in json.loads((src / 'gate.json').read_text(encoding='utf-8'))}
    rows = lexicon()
    ids = [r['id'] for r in rows]
    sup = np.array([r['super'] for r in rows]); cat = np.array([r['cat'] for r in rows])
    split = np.array([E.split_of(i) for i in ids])
    X = {lang: z[lang].astype(np.float64) for lang in ('zh', 'en')}
    L = X['zh'].shape[1]
    layers = {lang: [X[lang][:, l] for l in range(L)] for lang in X}
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
           'n_entities': len(ids), 'n_discovery': int((split == 'discovery').sum()), 'n_heldout': int((split == 'heldout').sum())}
    # ---- G1 ----
    g1 = {lang: [E.g1_layer(layers[lang][l], sup, cat, split) for l in range(L)] for lang in X}
    rep['G1'] = {'per_layer': g1}
    best = {}
    for lang in X:
        cv = []                                                         # 层选择只看 discovery（实体留一折外的最近中心）
        d = split == 'discovery'
        rng = np.random.default_rng(C['seed']); idx = np.where(d)[0]; parts = np.array_split(rng.permutation(idx), C['cv_folds'])
        for l in range(L):
            accs = [E.nearest_centroid_acc(layers[lang][l][np.setdiff1d(idx, p)], cat[np.setdiff1d(idx, p)], layers[lang][l][p], cat[p])
                    for p in parts]
            cv.append(float(np.mean(accs)))
        lb = int(np.argmax(cv))
        orth = E.hierarchy_orthogonality(layers[lang][lb], sup, cat, n_shuffle=C['rsa_perm'], seed=C['seed'])
        acc0 = g1[lang][0]['cat_acc_heldout']
        b = g1[lang][lb]
        best[lang] = {'layer': lb, 'cv_curve': cv, **b, 'hs0_cat_acc_heldout': acc0, 'orthogonality': orth,
                      'supported_hierarchy': bool(b['cat_acc_heldout'] >= C['G1_category_acc_min'] and b['cat_acc_heldout'] - acc0 >= C['G1_over_hs0_min']),
                      'supported_orthogonal': bool(orth['mean_abs_cos'] <= C['G1_orth_abs_cos_max'])}
    rep['G1']['selected'] = best
    # ---- RSA（中英相对结构）----
    rep['RSA_zh_en'] = [E.rsa(layers['zh'][l], layers['en'][l], n_perm=C['rsa_perm'], seed=C['seed']) for l in range(L)]
    # ---- G2 ----
    g2 = {}
    for attr in ('color', 'size'):
        keep = np.array([gate[i][f'{attr}_zh'] > 0 and gate[i][f'{attr}_en'] > 0 for i in ids])
        y = np.array([r[attr] for r in rows])
        cls, cnt = np.unique(y[keep], return_counts=True)
        ok_cls = set(cls[cnt >= C['min_class_n']])
        m = keep & np.isin(y, list(ok_cls))
        res = {'n_gated': int(keep.sum()), 'n_used': int(m.sum()), 'classes': {str(c): int(n) for c, n in zip(cls, cnt)},
               'classes_used': sorted(ok_cls)}
        if len(ok_cls) < 2:
            res['skipped'] = '过门实体不足以形成两类'
            g2[attr] = res
            continue
        for lang, other in (('zh', 'en'), ('en', 'zh')):
            lay = [x[m] for x in layers[lang]]
            op = E.operator_eval(lay, y[m], split[m], cat[m], crit=C, X_transfer=[x[m] for x in layers[other]])
            dl = E.operator_eval(lay, y[m], split[m], cat[m], crit=C, delta=True)
            h0 = E.operator_eval([lay[0]], y[m], split[m], cat[m], crit=C)['heldout_bal_acc']
            maj = E.category_majority_acc(y[m], split[m], cat[m])
            rank = E.lowrank_curve(lay[op['layer']], y[m], split[m], [1, 2, 4, 8, 16, 32, 64, 128], crit=C)
            full = op['heldout_bal_acc']
            rmin = next((r for r, a in sorted(rank.items()) if a >= C['lowrank_frac'] * full), None)
            res[lang] = {'operator': {k: v for k, v in op.items() if k != 'cv_curve'}, 'cv_curve': op['cv_curve'],
                         'delta_operator': {k: v for k, v in dl.items() if k != 'cv_curve'},
                         'hs0_bal_acc': h0, 'category_majority_bal_acc': maj, 'lowrank': rank, 'min_rank_90': rmin,
                         'supported_operator': bool(full - max(maj, h0) >= C['G2_margin_min']),
                         'supported_in_delta': bool(dl['heldout_bal_acc'] - maj >= C['G2_delta_margin_min']),
                         'supported_xlang': bool(op['transfer_bal_acc'] >= C['G2_xlang_ratio_min'] * full and op['transfer_bal_acc'] - max(maj, h0) >= C['G2_margin_min'])}
        g2[attr] = res
    rep['G2'] = g2
    out = ROOT / 'research/entity_atlas' / f'EA-v1-{Path(model_path).name}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    summ = {'G1': {l: {k: best[l][k] for k in ('layer', 'cat_acc_heldout', 'hs0_cat_acc_heldout', 'var_super', 'var_cat_within_super',
                                              'var_delta', 'pr_all', 'pr_delta', 'supported_hierarchy', 'supported_orthogonal')}
                   | {'orth': best[l]['orthogonality']} for l in best},
            'RSA_max': max(r['spearman'] for r in rep['RSA_zh_en']), 'RSA_hs0': rep['RSA_zh_en'][0]}
    for attr, r in g2.items():
        summ[attr] = {'n_used': r.get('n_used'), **{lang: {k: r[lang][k] for k in ('hs0_bal_acc', 'category_majority_bal_acc', 'min_rank_90',
                                                                                   'supported_operator', 'supported_in_delta', 'supported_xlang')}
                                                         | {'op': r[lang]['operator']['heldout_bal_acc'], 'layer': r[lang]['operator']['layer'],
                                                            'xfer': r[lang]['operator']['transfer_bal_acc'], 'delta': r[lang]['delta_operator']['heldout_bal_acc']}
                                                         for lang in ('zh', 'en') if lang in r}}
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': lambda: do_prereg(), 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
