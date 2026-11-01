# -*- coding: utf-8 -*-
"""EA-v2：共享属性算子与实体层级的确认检验（在全新实体上只评估一次）。设计依据：EA-v1 的预注册结果与事后诊断 D2（Phase 264）。

  发现集  EA-v1 的 210 个 ENT-XF 实体（编码沿用 EA-v1 的采集，句式相同）——只用于拟合算子、选层与 λ、估计类别中心；
  确认集  shared/dist_protocol/lexicon/EA-v2_new_entities.tsv 的 197 个新实体——判据冻结后才采集，只评估一次。
  目标    模型自己的回答（不需要人工标签）：
            color  11 色完整答案对数概率（各颜色表面形式取最大），减去该实体 11 色均值；
            size   「更大」组 − 「更小」组（与成年人比）；weight 「更重」−「更轻」（与一公斤比）；
            water  「水分多」−「水分少」；edible 「能吃」−「不能吃」（完整答案对数概率，各取组内最大）。
  判据    shared/dist_protocol/entity_atlas.py 的 CRITERIA_V2（随预注册冻结）：
            每个目标 × 语言：共享算子 / 个体差分 / 跨语言 三项判定；G1：新实体类别由发现集类别中心预测；RSA：新实体中英相对结构。
            总判定：颜色在两种语言都「共享算子」成立，且 size / weight / water / edible 中至少 2 个在两种语言都成立，记为「G2 确认」。

用法：python scripts/entity_atlas_v2.py prereg|collect|analyze --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import entity_atlas as E  # noqa: E402
import entity_atlas_run as V1  # noqa: E402

NEW = ROOT / 'shared/dist_protocol/lexicon/EA-v2_new_entities.tsv'
PREREG = ROOT / 'research/entity_atlas/EA-v2.prereg.json'
PALETTE = ['red', 'yellow', 'green', 'blue', 'white', 'black', 'brown', 'orange', 'purple', 'pink', 'gray']
BINARY = {
    'size': {'zh': ('跟成年人相比，{X}的个头通常', ['更大', '更高'], ['更小', '更矮']),
             'en': ('Compared with an adult human, the {E} is usually', [' bigger', ' larger', ' taller'], [' smaller', ' shorter'])},
    'weight': {'zh': ('问：{X}通常比一公斤重还是轻？\n答：{X}通常比一公斤', ['重', '更重'], ['轻', '更轻']),
               'en': ('Q: Is the {E} usually heavier or lighter than one kilogram?\nA: The {E} is usually', [' heavier'], [' lighter'])},
    'water': {'zh': ('问：{X}里面的水分通常多还是少？\n答：{X}里面的水分通常', ['多', '很多'], ['少', '很少']),
              'en': ('Q: Does the {E} usually contain a lot of water or very little water?\nA: The {E} usually contains',
                     [' a lot of water'], [' very little water'])},
    'edible': {'zh': ('问：{X}通常是人们拿来吃的吗？\n答：{X}通常', ['可以吃', '能吃'], ['不能吃', '不可以吃']),
               'en': ('Q: Is the {E} something people usually eat?\nA: People usually', [' eat it'], [' do not eat it'])},
}


def new_entities():
    rows = []
    for line in NEW.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            f = line.split('\t')
            rows.append({'id': f[0], 'cat': f[1], 'super': f[2], 'zh': f[3], 'en': f[4]})
    return rows


def prereg_doc():
    import build_ent_grammar as B
    return {
        'id': 'EA-v2', 'schema': 'entity_atlas_prereg.v2', 'created': '2026-11-01',
        'basis': 'EA-v1 预注册结果与事后诊断 D2（research/entity_atlas/EA-v1-qwen3-4b/，Phase 264）',
        'question': '在从未参与拟合、选层与选 λ 的新实体上：一个对全部实体共享的线性算子，能否从不含属性线索的实体编码预测模型自己对颜色、大小、重量、'
                    '含水、可食的回答（超出类别均值与 token 身份）？属性是否落在个体差分里、能否跨语言复用？新实体的类别能否由旧实体的类别中心预测？',
        'discovery': {'lexicon': str(V1.LEXICON.relative_to(ROOT)).replace('\\', '/'), 'sha256': hashlib.sha256(V1.LEXICON.read_bytes()).hexdigest(),
                      'n': len(V1.lexicon()), 'codes': 'EA-v1 采集（同一模型快照、同一句式）'},
        'confirmation': {'lexicon': str(NEW.relative_to(ROOT)).replace('\\', '/'), 'sha256': hashlib.sha256(NEW.read_bytes()).hexdigest(),
                         'n': len(new_entities())},
        'frames': V1.FRAMES,
        'targets': {'color': {'prompt': {'zh': B.COLOR_PROMPT_ZH, 'en': B.COLOR_PROMPT_EN}, 'palette': PALETTE,
                              'forms': {'zh': {c: B.COLOR_FORMS_ZH[c] for c in PALETTE}, 'en': {c: B.COLOR_FORMS_EN[c] for c in PALETTE}},
                              'definition': '完整答案对数概率，各颜色取表面形式最大值，再减去该实体 11 色均值'},
                    **{k: {lang: {'prompt': v[lang][0], 'pos': v[lang][1], 'neg': v[lang][2]} for lang in v} for k, v in BINARY.items()},
                    'binary_definition': 'max 完整答案对数概率(pos) − max(neg)'},
        'g1_layers': {'zh': 27, 'en': 10}, 'rsa_layer': 31,
        'criteria': E.CRITERIA_V2,
    }


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（EA-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def out_dir(key):
    return V1.native_dir(key).parent / 'EA-v2'


def do_collect(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol.census_runner import model_snapshot
    torch.set_grad_enabled(False)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    snap = model_snapshot(model_path)['sha256']
    out = out_dir(snap[:16] + '-bf16')
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    t0 = time.time()

    def lp(prompt, ans):
        p, a = enc(prompt), enc(ans)
        lg = model(input_ids=torch.as_tensor([p + a], device='cuda')).logits[0].float()
        return float(torch.log_softmax(lg[len(p) - 1:-1], -1).gather(-1, torch.as_tensor(a, device='cuda')[:, None]).sum())
    fill = lambda s, r: s.replace('{X}', r['zh']).replace('{E}', r['en'])
    T = pre['targets']
    sets = {'disc': V1.lexicon(), 'conf': new_entities()}
    arrays = {}
    for name, rows in sets.items():
        for lang in ('zh', 'en'):
            col = np.zeros((len(rows), len(PALETTE)))
            for i, r in enumerate(rows):
                pr = fill(T['color']['prompt'][lang], r)
                for j, c in enumerate(PALETTE):
                    col[i, j] = max(lp(pr, f) for f in T['color']['forms'][lang][c])
            arrays[f'{name}_{lang}_color'] = col
            for k in BINARY:
                spec = T[k][lang]
                arrays[f'{name}_{lang}_{k}'] = np.array([max(lp(fill(spec['prompt'], r), f) for f in spec['pos'])
                                                         - max(lp(fill(spec['prompt'], r), f) for f in spec['neg']) for r in rows])
        print(f'@@progress targets {name} {round(time.time() - t0)}s', flush=True)
    rows = sets['conf']
    L = model.config.num_hidden_layers + 1
    for lang in ('zh', 'en'):
        arr = np.zeros((len(rows), L, model.config.hidden_size), dtype=np.float32)
        for i, r in enumerate(rows):
            acc = 0
            for fr in pre['frames'][lang]:
                o = model(input_ids=torch.as_tensor([enc(fill(fr, r))], device='cuda'), output_hidden_states=True)
                acc = acc + torch.stack([h[0, -1].float() for h in o.hidden_states]).cpu().numpy()
            arr[i] = acc / len(pre['frames'][lang])
        arrays[f'conf_{lang}_codes'] = arr.astype(np.float16)
    np.savez_compressed(out / 'data.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'n_disc': len(sets['disc']), 'n_conf': len(rows), 'torch': torch.__version__, 'seconds': round(time.time() - t0, 1)}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap = model_snapshot(model_path)['sha256']
    key = snap[:16] + '-bf16'
    z = np.load(out_dir(key) / 'data.npz')
    v1 = np.load(V1.native_dir(key) / 'codes.npz')
    drows, crows = V1.lexicon(), new_entities()
    dcat = np.array([r['cat'] for r in drows]); ccat = np.array([r['cat'] for r in crows])
    dsup = np.array([r['super'] for r in drows]); csup = np.array([r['super'] for r in crows])
    D = {lang: v1[lang].astype(np.float64) for lang in ('zh', 'en')}
    Cc = {lang: z[f'conf_{lang}_codes'].astype(np.float64) for lang in ('zh', 'en')}
    L = D['zh'].shape[1]
    lay = lambda A: [A[:, l] for l in range(L)]
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C,
           'n_discovery': len(drows), 'n_confirmation': len(crows)}
    # G1
    g1 = {}
    for lang in ('zh', 'en'):
        l = pre['g1_layers'][lang]
        acc = E.nearest_centroid_acc(D[lang][:, l], dcat, Cc[lang][:, l], ccat)
        acc0 = E.nearest_centroid_acc(D[lang][:, 0], dcat, Cc[lang][:, 0], ccat)
        sacc = E.nearest_centroid_acc(D[lang][:, l], dsup, Cc[lang][:, l], csup)
        g1[lang] = {'layer': l, 'cat_acc': acc, 'super_acc': sacc, 'hs0_cat_acc': acc0,
                    'supported': bool(acc >= C['g1_acc_min'] and acc - acc0 >= C['g1_over_hs0_min'])}
    rep['G1_confirm'] = g1
    # RSA（新实体中英）
    rep['RSA_zh_en_new'] = {'layer': pre['rsa_layer'], **E.rsa(Cc['zh'][:, pre['rsa_layer']], Cc['en'][:, pre['rsa_layer']], n_perm=200, seed=0),
                            'by_layer': [round(E.rsa(Cc['zh'][:, l], Cc['en'][:, l], n_perm=20, seed=0)['spearman'], 4) for l in range(L)]}
    # G2
    g2 = {}
    for t in ['color'] + list(BINARY):
        g2[t] = {}
        for lang, other in (('zh', 'en'), ('en', 'zh')):
            Yd = z[f'disc_{lang}_{t}']; Yc = z[f'conf_{lang}_{t}']
            if t == 'color':
                Yd = Yd - Yd.mean(1, keepdims=True); Yc = Yc - Yc.mean(1, keepdims=True)
            res = E.confirm_operator(lay(D[lang]), Yd, dcat, lay(Cc[lang]), Yc, ccat, conf_other_layers=lay(Cc[other]), crit=C)
            if t == 'color':
                P = E.ridge_predict(E.ridge_fit(D[lang][:, res['layer']], Yd, res['lambda_rel']), Cc[lang][:, res['layer']])
                cm = {c: Yd[dcat == c].mean(0) for c in set(dcat)}
                res['top1_agree'] = float(np.mean(P.argmax(1) == Yc.argmax(1)))
                res['top1_agree_category_mean'] = float(np.mean(np.stack([cm[c] for c in ccat]).argmax(1) == Yc.argmax(1)))
            g2[t][lang] = res
    rep['G2_confirm'] = g2
    both = lambda t: all(g2[t][lang]['supported_operator'] for lang in ('zh', 'en'))
    rep['verdict'] = {'G1_confirmed': all(g1[l]['supported'] for l in g1),
                      'RSA_above_null': bool(rep['RSA_zh_en_new']['spearman'] > rep['RSA_zh_en_new']['null_p95']),
                      'color_operator_both_langs': both('color'),
                      'other_attributes_both_langs': [t for t in BINARY if both(t)],
                      'G2_confirmed': bool(both('color') and sum(both(t) for t in BINARY) >= 2)}
    dst = ROOT / 'research/entity_atlas' / f'EA-v2-{Path(model_path).name}'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    show = {t: {lang: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g2[t][lang].items()} for lang in g2[t]} for t in g2}
    print(json.dumps({'verdict': rep['verdict'], 'G1': g1, 'RSA': {k: rep['RSA_zh_en_new'][k] for k in ('spearman', 'null_p95')}, 'G2': show},
                     ensure_ascii=False, indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
