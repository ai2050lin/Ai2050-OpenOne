# -*- coding: utf-8 -*-
"""MV-v1：句型模式向量与内容解耦、可加与组合（G3 第一步）。

材料  实体：发现集 = EA-v1 的 210 个 ENT-XF 实体，确认集 = EA-v2 的 197 个新实体（只评估一次）。
      每个实体 × 语言 2×2×2 个句子，句末同为「。」/「.」（读出位置的 token 相同，hs[0] 在各格完全相同）：
        q 疑问（他问……是否）/ 陈述（他说……）；n 否定 / 肯定；t 类别真 / 假（假类别取另一大类，按 sha256 确定）。
      行为门：裸陈述句「{X}是一种{C}。」/「{X}不是一种{C}。」后接「这句话是」，比较「对的 / 错的」完整答案对数概率（英文 true / false）。
采集  句末 token 在全部 37 个 hs 层的残差，float16 存节点本地。
分析  mode_atlas.v1（判据 CRITERIA 随预注册冻结）：
  H1 解耦      确认集上格均值模型 R² ≥ 0.5（层由发现集分半 R² 选定），两种语言。
  H2 疑问可加  含 q 的交互（qn、qt、qnt）占发现集格结构方差 ≤ 0.10，两种语言（同一选定层）。
  H3 组合      陈述句句末的真值（t XOR n）探针在确认集平衡准确率 ≥ 0.75（层由发现集交叉验证选定）；只在行为门 ≥ 0.7 的语言上解释。
  H4 跨语言    q 与 n：中英效应向量 cos ≥ 0.3 且大于跨因子 |cos|；中文发现集探针用于英文确认集平衡准确率 ≥ 0.75。
用法：python scripts/mode_atlas_v1.py prereg|collect|analyze --model models/hf/qwen3-4b
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

from shared.dist_protocol import mode_atlas as M  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402

PREREG = ROOT / 'research/mode_atlas/MV-v1.prereg.json'
CAT = {'aquatic': ('水生动物', 'an aquatic animal'), 'bird': ('鸟', 'a bird'), 'mammal': ('哺乳动物', 'a mammal'),
       'fruit': ('水果', 'a fruit'), 'vegetable': ('蔬菜', 'a vegetable'), 'plant': ('植物', 'a plant'), 'food': ('食品', 'a food')}
SUPER = {'aquatic': 'animal', 'bird': 'animal', 'mammal': 'animal', 'fruit': 'plant', 'vegetable': 'plant', 'plant': 'plant', 'food': 'food'}
FRAMES = {   # 键 (q, n)
    'zh': {(0, 0): '他说{X}是一种{C}。', (0, 1): '他说{X}不是一种{C}。',
           (1, 0): '他问{X}是否是一种{C}。', (1, 1): '他问{X}是否不是一种{C}。'},
    'en': {(0, 0): 'He said the {E} is {C}.', (0, 1): 'He said the {E} is not {C}.',
           (1, 0): 'He asked whether the {E} is {C}.', (1, 1): 'He asked whether the {E} is not {C}.'},
}
GATE = {'zh': ({0: '{X}是一种{C}。', 1: '{X}不是一种{C}。'}, '这句话是', ['对的'], ['错的']),
        'en': ({0: 'The {E} is {C}.', 1: 'The {E} is not {C}.'}, ' This statement is', [' true'], [' false'])}


def false_cat(r):
    """另一大类中的类别（动物 ↔ 植物 / 食品），按 sha256 确定。"""
    pool = sorted(c for c in CAT if (SUPER[c] == 'animal') != (SUPER[r['cat']] == 'animal'))
    return pool[hashlib.sha256(f"MV-v1|{r['id']}".encode()).digest()[0] % len(pool)]


def sentences(r, lang):
    """8 个格（c = 4q + 2n + t）的句子。"""
    out = []
    for c in range(8):
        q, n, t = (c >> 2) & 1, (c >> 1) & 1, c & 1
        cat = r['cat'] if t else false_cat(r)
        out.append(FRAMES[lang][q, n].replace('{X}', r['zh']).replace('{E}', r['en']).replace('{C}', CAT[cat][0 if lang == 'zh' else 1]))
    return out


def rows():
    return {'disc': V1.lexicon(), 'conf': V2.new_entities()}


def prereg_doc():
    R = rows()
    return {'id': 'MV-v1', 'schema': 'mode_atlas_prereg.v1', 'created': '2026-11-01',
            'basis': 'G1–G5 总方案的 G3；G1 / G2 的实体材料（EA-v1 发现集、EA-v2 确认集）',
            'question': '疑问、否定这类句型模式，在句末状态里是否是与具体内容无关、可以叠加的方向？中英文是否共用？'
                        '否定与类别真假的组合（真值 = t XOR n）是否被编码？',
            'discovery': {'n': len(R['disc']), 'lexicon_sha256': hashlib.sha256(V1.LEXICON.read_bytes()).hexdigest()},
            'confirmation': {'n': len(R['conf']), 'lexicon_sha256': hashlib.sha256(V2.NEW.read_bytes()).hexdigest()},
            'frames': {lang: {f'q{k[0]}n{k[1]}': v for k, v in f.items()} for lang, f in FRAMES.items()},
            'categories': CAT, 'false_category_rule': false_cat.__doc__,
            'gate': {lang: {'statements': g[0], 'suffix': g[1], 'true': g[2], 'false': g[3]} for lang, g in GATE.items()},
            'readout': '句末 token（「。」/「.」）全部 37 个 hs 层；hs[-1] 为最终归一化之后',
            'cells': 'c = 4q + 2n + t', 'examples': {lang: sentences(R['disc'][0], lang) for lang in ('zh', 'en')},
            'criteria': M.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MV-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def out_dir(key):
    return V1.native_dir(key).parents[2] / '_mode_atlas' / key / 'MV-v1'


def do_collect(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol.census_runner import model_snapshot
    torch.set_grad_enabled(False)
    snap = model_snapshot(model_path)['sha256']
    out = out_dir(snap[:16] + '-bf16')
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    L, d = model.config.num_hidden_layers + 1, model.config.hidden_size
    t0 = time.time()

    def lp(prompt, ans):
        p, a = enc(prompt), enc(ans)
        lg = model(input_ids=torch.as_tensor([p + a], device='cuda')).logits[0].float()
        return float(torch.log_softmax(lg[len(p) - 1:-1], -1).gather(-1, torch.as_tensor(a, device='cuda')[:, None]).sum())
    arrays, final_tok = {}, {}
    for s, rs in rows().items():
        for lang in ('zh', 'en'):
            H = np.zeros((len(rs), 8, L, d), dtype=np.float16)
            gate = np.zeros((len(rs), 4))                  # [n, t] → 对的 − 错的
            for i, r in enumerate(rs):
                for c, text in enumerate(sentences(r, lang)):
                    ids = enc(text)
                    final_tok.setdefault(lang, set()).add(ids[-1])
                    o = model(input_ids=torch.as_tensor([ids], device='cuda'), output_hidden_states=True)
                    H[i, c] = torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16)
                st, suf, tr, fa = GATE[lang]
                for n in (0, 1):
                    for t in (0, 1):
                        cat = r['cat'] if t else false_cat(r)
                        pr = st[n].replace('{X}', r['zh']).replace('{E}', r['en']).replace('{C}', CAT[cat][0 if lang == 'zh' else 1]) + suf
                        gate[i, 2 * n + t] = max(lp(pr, f) for f in tr) - max(lp(pr, f) for f in fa)
            arrays[f'{s}_{lang}_H'] = H
            arrays[f'{s}_{lang}_gate'] = gate
            print(f'@@progress {s}/{lang} {round(time.time() - t0)}s', flush=True)
    np.savez(out / 'data.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'final_token_ids': {k: sorted(v) for k, v in final_tok.items()}, 'torch': torch.__version__,
           'seconds': round(time.time() - t0, 1), 'sha256_data': hashlib.sha256((out / 'data.npz').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap = model_snapshot(model_path)['sha256']
    z = np.load(out_dir(snap[:16] + '-bf16') / 'data.npz')
    H = {(s, lang): z[f'{s}_{lang}_H'] for s in ('disc', 'conf') for lang in ('zh', 'en')}      # float16；逐层切片时再转 float64
    L = H['disc', 'zh'].shape[2]
    rep = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C}
    # 行为门：裸陈述句的真假判断（真值 = t XOR n；判对 ⇔ 对的 − 错的 与真值同号）
    gate = {}
    for lang in ('zh', 'en'):
        g = np.concatenate([z[f'disc_{lang}_gate'], z[f'conf_{lang}_gate']])
        truth = np.array([((c >> 1) & 1) ^ (c & 1) for c in range(4)], bool)
        ok = (g > 0) == truth[None]
        gate[lang] = {'acc': float(ok.mean()), 'acc_by_cell_nt': [float(x) for x in ok.mean(0)],
                      'pass': bool(ok.mean() >= C['H3_gate_min'])}
    rep['gate'] = gate
    per = {}
    for lang in ('zh', 'en'):
        Gd = [M.center(H['disc', lang][:, :, l].astype(np.float64)) for l in range(L)]
        Gc = [M.center(H['conf', lang][:, :, l].astype(np.float64)) for l in range(L)]
        sh = [M.split_half_r2(Gd[l], C['seed']) for l in range(L)]
        l = int(np.nanargmax(sh))
        eff = M.effects(Gd[l])
        share = M.variance_shares(eff)
        rows_l = []
        for ll in range(L):
            e2 = M.effects(Gd[ll])
            rows_l.append({'layer': ll, 'split_half_r2': sh[ll], 'r2_cellmean_conf': M.r2_cells(Gc[ll], Gd[ll].mean(0)),
                           'r2_main_conf': M.r2_cells(Gc[ll], M.predict(e2, ['q', 'n', 't'])),
                           'shares': M.variance_shares(e2)})
        per[lang] = {'layer': l, 'r2_cellmean_conf': M.r2_cells(Gc[l], Gd[l].mean(0)),
                     'r2_main_conf': M.r2_cells(Gc[l], M.predict(eff, ['q', 'n', 't'])),
                     'r2_main_plus_nt_conf': M.r2_cells(Gc[l], M.predict(eff, ['q', 'n', 't', 'nt'])),
                     'shares': share, 'q_interaction_share': share['qn'] + share['qt'] + share['qnt'],
                     'by_layer': rows_l}
        # 真值探针：陈述句（q = 0）；逐层交叉验证选层
        def tdata(s):
            X = H[s, lang][:, :4]                         # c < 4 ⇔ q = 0（float16，按层切片后再转）
            y = np.tile([((c >> 1) & 1) ^ (c & 1) for c in range(4)], len(X)).astype(bool)
            g = np.repeat(np.arange(len(X)), 4)
            return X, y, g
        Xd, yd, gd = tdata('disc'); Xc, yc, _ = tdata('conf')
        pr = [M.probe(Xd[:, :, ll].astype(np.float64).reshape(-1, Xd.shape[-1]), yd, gd, Xc[:, :, ll].astype(np.float64).reshape(-1, Xc.shape[-1]), yc, C) for ll in range(L)]
        lt = int(np.argmax([p['cv_acc'] for p in pr]))
        per[lang]['truth_probe'] = {'layer': lt, **pr[lt], 'by_layer': [{'layer': ll, 'cv_acc': p['cv_acc'], 'acc': p['acc']} for ll, p in enumerate(pr)]}
        # 同一层上 n、t 各自的探针（主效应，对照）
        for f, bit in (('n', 1), ('t', 0), ('q', 2)):
            yd_f = np.tile([(c >> bit) & 1 for c in range(8)], len(H['disc', lang])).astype(bool)
            yc_f = np.tile([(c >> bit) & 1 for c in range(8)], len(H['conf', lang])).astype(bool)
            gd8 = np.repeat(np.arange(len(H['disc', lang])), 8)
            per[lang][f'{f}_probe'] = M.probe(H['disc', lang][:, :, lt].astype(np.float64).reshape(-1, H['disc', lang].shape[-1]), yd_f, gd8,
                                              H['conf', lang][:, :, lt].astype(np.float64).reshape(-1, H['conf', lang].shape[-1]), yc_f, C)
        print(f'@@progress {lang} layer={l} r2cm={per[lang]["r2_cellmean_conf"]:.3f} r2main={per[lang]["r2_main_conf"]:.3f} '
              f'qint={per[lang]["q_interaction_share"]:.3f} truth L{lt}={pr[lt]["acc"]:.3f}', flush=True)
    rep['per_lang'] = per
    # 跨语言：同一层（中英选定层的较小者 → 取中文选定层，预注册：用中文 H1 选定层）
    l = per['zh']['layer']
    effz, effe = M.effects(M.center(H['disc', 'zh'][:, :, l].astype(np.float64))), M.effects(M.center(H['disc', 'en'][:, :, l].astype(np.float64)))
    xl = {}
    for f, bit in (('q', 2), ('n', 1)):
        same = M.cos(effz[f], effe[f])
        cross = max(abs(M.cos(effz[f], effe[g])) for g in ('q', 'n', 't') if g != f)
        yd = np.tile([(c >> bit) & 1 for c in range(8)], len(H['disc', 'zh'])).astype(bool)
        yc = np.tile([(c >> bit) & 1 for c in range(8)], len(H['conf', 'en'])).astype(bool)
        gd = np.repeat(np.arange(len(H['disc', 'zh'])), 8)
        d = H['disc', 'zh'].shape[-1]
        pr = M.probe(H['disc', 'zh'][:, :, l].astype(np.float64).reshape(-1, d), yd, gd, H['conf', 'zh'][:, :, l].astype(np.float64).reshape(-1, d), yc,
                     C, X_te2=H['conf', 'en'][:, :, l].astype(np.float64).reshape(-1, d), y_te2=yc)
        xl[f] = {'cos_same': same, 'max_abs_cos_cross_factor': cross, **pr,
                 'pass': bool(same >= C['H4_cos_min'] and same > cross and pr['acc_transfer'] >= C['H4_transfer_min'])}
    xl['layer'] = l
    xl['cos_by_layer'] = {f: [M.cos(M.effects(M.center(H['disc', 'zh'][:, :, ll].astype(np.float64)))[f], M.effects(M.center(H['disc', 'en'][:, :, ll].astype(np.float64)))[f])
                              for ll in range(L)] for f in ('q', 'n', 't', 'nt')}
    rep['cross_lang'] = xl
    rep['verdict'] = {
        'H1_decouple': {lang: bool(per[lang]['r2_cellmean_conf'] >= C['H1_decouple_r2_min']) for lang in per},
        'H2_q_additive': {lang: bool(per[lang]['q_interaction_share'] <= C['H2_q_interaction_max']) for lang in per},
        'H3_truth_composed': {lang: (bool(per[lang]['truth_probe']['acc'] >= C['H3_truth_acc_min']) if gate[lang]['pass'] else 'gate_failed')
                              for lang in per},
        'H4_cross_lang': {f: xl[f]['pass'] for f in ('q', 'n')},
    }
    dst = ROOT / 'research/mode_atlas' / f'MV-v1-{Path(model_path).name}'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({'gate': gate, 'verdict': rep['verdict']}, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
