# -*- coding: utf-8 -*-
"""MV-v2：成对行为门、跨句式框架复现、中层跨语言复用、真值方向写入（Phase 269 / MV-v1 的预注册后续）。

材料  实体同 MV-v1（发现集 210、确认集 197）；新句式框架 B（「报道称 / 报道询问……是否」，英文 The report says / asks whether），
      格序 c = 4q + 2n + t 与 MV-v1 相同，读出 = 句末「。」/「.」；另采裸陈述句（MV-v1 行为门的句子）句末状态。
预测（判据 mode_atlas.CRITERIA_V2，随预注册冻结）：
  H1 成对行为门   同一实体、同一 n，真句的「对−错」间隔大于假句的比例 ≥ 0.7（每种语言）。
  H2 新框架真值   框架 B 陈述句真值探针（层由框架 B 发现集交叉验证在中层段 12–21 内选定），确认集平衡准确率 ≥ 0.75（每种语言；中文须 H1 成立）。
  H3 跨框架       同一层：MV-v1 框架 A 发现集训练的真值探针用于框架 B 确认集 ≥ 0.75；框架 A / B 发现集的 n、t、nt 效应向量 cos 都 ≥ 0.5（每种语言）。
  H4 中层跨语言   层 = 中层段内使框架 B 发现集中英 q、n 效应 cos 均值最大的层；q 与 n：cos ≥ 0.3 且大于跨因子 |cos|，
                  中文发现集探针用于英文确认集 ≥ 0.75。
  H5 真值写入     确认集 56 个实体（每类 8 个，按 sha256）× 4 个裸陈述句：在第 13 / 17 / 21 层、句末 token 上沿真值方向
                  （发现集裸陈述句 真 − 假 的均值差，单位化）加 k·σ·v，读出「对−错」间隔；entity_atlas.steer_verdict 成立的层
                  至少 1 个（每种语言）。对照：2 个随机方向；另报告类别真假主效应方向 t（描述性）。
用法：python scripts/mode_atlas_v2.py prereg|collect|analyze|steer --model models/hf/qwen3-4b
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
from shared.dist_protocol import mode_atlas as M  # noqa: E402
import mode_atlas_v1 as MV1  # noqa: E402

PREREG = ROOT / 'research/mode_atlas/MV-v2.prereg.json'
FRAMES_B = {
    'zh': {(0, 0): '报道称{X}是一种{C}。', (0, 1): '报道称{X}不是一种{C}。',
           (1, 0): '报道询问{X}是否是一种{C}。', (1, 1): '报道询问{X}是否不是一种{C}。'},
    'en': {(0, 0): 'The report says the {E} is {C}.', (0, 1): 'The report says the {E} is not {C}.',
           (1, 0): 'The report asks whether the {E} is {C}.', (1, 1): 'The report asks whether the {E} is not {C}.'},
}


def fill(s, r, cat, lang):
    return s.replace('{X}', r['zh']).replace('{E}', r['en']).replace('{C}', MV1.CAT[cat][0 if lang == 'zh' else 1])


def frame_b(r, lang):
    out = []
    for c in range(8):
        q, n, t = (c >> 2) & 1, (c >> 1) & 1, c & 1
        out.append(fill(FRAMES_B[lang][q, n], r, r['cat'] if t else MV1.false_cat(r), lang))
    return out


def bare(r, lang):
    """裸陈述句，格序 2n + t（与 MV-v1 行为门相同）。"""
    st = MV1.GATE[lang][0]
    return [fill(st[(c >> 1) & 1], r, r['cat'] if c & 1 else MV1.false_cat(r), lang) for c in range(4)]


def steer_entities():
    rs = MV1.rows()['conf']
    out = []
    for c in sorted({r['cat'] for r in rs}):
        out += sorted([r for r in rs if r['cat'] == c], key=lambda r: hashlib.sha256(f"MV-v2|{r['id']}".encode()).hexdigest())[:8]
    return out


def prereg_doc():
    R = MV1.rows()
    return {'id': 'MV-v2', 'schema': 'mode_atlas_prereg.v2', 'created': '2026-11-01',
            'basis': 'MV-v1（Phase 269）：预注册 H4 因选层规则（浅层）失败，事后诊断显示中层跨语言复用强；中文行为门绝对口径失败、成对口径区分真假',
            'question': '中文模型是否区分句子真假（成对口径）？真值组合与句型方向是否跨句式框架复现？中层是否跨语言复用？'
                        '沿真值方向写入句末状态，模型的真假判断是否随之改变？',
            'mv1_prereg_sha256': MV1.PREREG.with_suffix('.sha256').read_text().strip(),
            'frames_B': {lang: {f'q{k[0]}n{k[1]}': v for k, v in f.items()} for lang, f in FRAMES_B.items()},
            'bare_and_gate': {lang: {'statements': g[0], 'suffix': g[1], 'true': g[2], 'false': g[3]} for lang, g in MV1.GATE.items()},
            'examples': {lang: {'B': frame_b(R['disc'][0], lang), 'bare': bare(R['disc'][0], lang)} for lang in ('zh', 'en')},
            'steer': {'entities': [r['id'] for r in steer_entities()], 'position': '裸陈述句句末 token（「。」/「.」），提示 = ids(陈述句) + ids(后缀)',
                      'direction': '发现集裸陈述句在该层的 mean(真) − mean(假)，单位化；σ = 发现集 4 格投影的标准差',
                      'readout': '「对的 − 错的」（英文 true − false）完整答案对数概率差', 'verdict': 'entity_atlas.steer_verdict'},
            'criteria': M.CRITERIA_V2}


def do_prereg():
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MV-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def key_of(model_path):
    from shared.dist_protocol.census_runner import model_snapshot
    snap = model_snapshot(model_path)['sha256']
    return snap, snap[:16] + '-bf16'


def out_dir(key):
    return MV1.out_dir(key).parent / 'MV-v2'


def do_collect(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    snap, key = key_of(model_path)
    out = out_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    L, d = model.config.num_hidden_layers + 1, model.config.hidden_size
    t0 = time.time()

    def states(ids):
        o = model(input_ids=torch.as_tensor([ids], device='cuda'), output_hidden_states=True)
        return torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16)

    def lp(p, a):
        lg = model(input_ids=torch.as_tensor([p + a], device='cuda')).logits[0].float()
        return float(torch.log_softmax(lg[len(p) - 1:-1], -1).gather(-1, torch.as_tensor(a, device='cuda')[:, None]).sum())
    arrays, final_tok = {}, {}
    for s, rs in MV1.rows().items():
        for lang in ('zh', 'en'):
            HB = np.zeros((len(rs), 8, L, d), dtype=np.float16)
            HS = np.zeros((len(rs), 4, L, d), dtype=np.float16)
            gate = np.zeros((len(rs), 4))
            _, suf, tr, fa = MV1.GATE[lang]
            for i, r in enumerate(rs):
                for c, text in enumerate(frame_b(r, lang)):
                    ids = enc(text)
                    final_tok.setdefault(lang, set()).add(ids[-1])
                    HB[i, c] = states(ids)
                for c, text in enumerate(bare(r, lang)):
                    ids = enc(text)
                    final_tok.setdefault(lang, set()).add(ids[-1])
                    HS[i, c] = states(ids)
                    p = ids + enc(suf)
                    gate[i, c] = max(lp(p, enc(f)) for f in tr) - max(lp(p, enc(f)) for f in fa)
            arrays[f'{s}_{lang}_B'] = HB
            arrays[f'{s}_{lang}_bare'] = HS
            arrays[f'{s}_{lang}_gate'] = gate
            print(f'@@progress {s}/{lang} {round(time.time() - t0)}s', flush=True)
    np.savez(out / 'data.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(),
           'final_token_ids': {k: sorted(v) for k, v in final_tok.items()}, 'torch': torch.__version__,
           'seconds': round(time.time() - t0, 1), 'sha256_data': hashlib.sha256((out / 'data.npz').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


def report_path(model_path):
    d = ROOT / 'research/mode_atlas' / f'MV-v2-{Path(model_path).name}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, upd, verdict):
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    rep.setdefault('verdict', {}).update(verdict)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


def bits(cells, fn, n):
    return np.tile([fn(c) for c in cells], n).astype(bool)


def do_analyze(model_path):
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path)
    z = np.load(out_dir(key) / 'data.npz')
    za = np.load(MV1.out_dir(key) / 'data.npz')
    B = {(s, lang): z[f'{s}_{lang}_B'] for s in ('disc', 'conf') for lang in ('zh', 'en')}
    A = {(s, lang): za[f'{s}_{lang}_H'] for s in ('disc', 'conf') for lang in ('zh', 'en')}
    d = B['disc', 'zh'].shape[-1]
    band = C['mid_band']
    X = lambda arr, cells, l: arr[:, cells, l].astype(np.float64).reshape(-1, d)
    truth = lambda c: ((c >> 1) & 1) ^ (c & 1)
    nd, nc = len(B['disc', 'zh']), len(B['conf', 'zh'])
    gd4 = np.repeat(np.arange(nd), 4)
    upd, verdict = {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'criteria': C}, {}
    # H1 成对行为门（全部 407 个实体）
    gate = {lang: M.paired_gate(np.concatenate([z[f'disc_{lang}_gate'], z[f'conf_{lang}_gate']])) for lang in ('zh', 'en')}
    for lang in gate:
        gate[lang]['pass'] = bool(gate[lang]['acc'] >= C['H1_paired_gate_min'])
        gate[lang]['absolute_acc'] = float(np.mean((np.concatenate([z[f'disc_{lang}_gate'], z[f'conf_{lang}_gate']]) > 0)
                                                   == np.array([0, 1, 1, 0], bool)[None]))
    upd['H1_gate'] = gate
    verdict['H1'] = {lang: gate[lang]['pass'] for lang in gate}
    h2, h3 = {}, {}
    for lang in ('zh', 'en'):
        cells = list(range(4))
        yd, yc = bits(cells, truth, nd), bits(cells, truth, nc)
        pr = {l: M.probe(X(B['disc', lang], cells, l), yd, gd4, X(B['conf', lang], cells, l), yc, C) for l in band}
        l = max(band, key=lambda ll: pr[ll]['cv_acc'])
        h2[lang] = {'layer': l, **pr[l], 'by_layer': {ll: pr[ll]['acc'] for ll in band},
                    'pass': (bool(pr[l]['acc'] >= C['H2_truth_acc_min']) if gate[lang]['pass'] else 'gate_failed')}
        xa = M.probe(X(A['disc', lang], cells, l), yd, gd4, X(B['conf', lang], cells, l), yc, C)
        ea = M.effects(M.center(A['disc', lang][:, :, l].astype(np.float64)))
        eb = M.effects(M.center(B['disc', lang][:, :, l].astype(np.float64)))
        cs = {f: M.cos(ea[f], eb[f]) for f in ('q', 'n', 't', 'nt')}
        h3[lang] = {'layer': l, 'truth_A_to_B': xa['acc'], 'cos_A_B': cs,
                    'pass': bool(xa['acc'] >= C['H3_xframe_truth_min'] and all(cs[f] >= C['H3_xframe_cos_min'] for f in ('n', 't', 'nt')))}
        print(f'@@progress {lang} H2 L{l} acc={pr[l]["acc"]:.3f} H3 A→B={xa["acc"]:.3f} cos={ {k: round(v, 2) for k, v in cs.items()} }', flush=True)
    upd['H2_truth_frameB'], upd['H3_cross_frame'] = h2, h3
    verdict['H2'] = {lang: h2[lang]['pass'] for lang in h2}
    verdict['H3'] = {lang: h3[lang]['pass'] for lang in h3}
    # H4 中层跨语言（框架 B）
    effs = {ll: (M.effects(M.center(B['disc', 'zh'][:, :, ll].astype(np.float64))),
                 M.effects(M.center(B['disc', 'en'][:, :, ll].astype(np.float64)))) for ll in band}
    score = {ll: np.mean([M.cos(effs[ll][0][f], effs[ll][1][f]) for f in ('q', 'n')]) for ll in band}
    l = max(band, key=score.get)
    ez, ee = effs[l]
    h4 = {'layer': l, 'selection_score_by_layer': score}
    gd8 = np.repeat(np.arange(nd), 8)
    for f, bit in (('q', 2), ('n', 1)):
        same = M.cos(ez[f], ee[f])
        cross = max(abs(M.cos(ez[f], ee[g])) for g in ('q', 'n', 't') if g != f)
        fn = lambda c, b=bit: (c >> b) & 1
        pr = M.probe(X(B['disc', 'zh'], list(range(8)), l), bits(range(8), fn, nd), gd8, X(B['conf', 'zh'], list(range(8)), l),
                     bits(range(8), fn, nc), C, X_te2=X(B['conf', 'en'], list(range(8)), l), y_te2=bits(range(8), fn, nc))
        h4[f] = {'cos_same': same, 'max_abs_cos_cross_factor': cross, **pr,
                 'pass': bool(same >= C['H4_cos_min'] and same > cross and pr['acc_transfer'] >= C['H4_transfer_min'])}
    upd['H4_cross_lang_mid'] = h4
    verdict['H4'] = {f: h4[f]['pass'] for f in ('q', 'n')}
    save(model_path, upd, verdict)
    print(json.dumps(verdict, ensure_ascii=False))


def do_steer(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path)
    z = np.load(out_dir(key) / 'data.npz')
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    blocks = model.model.layers
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    ents = steer_entities()
    rng = np.random.default_rng(C['seed'])
    rand_dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    ctl = {'pos': None, 'vecs': None}

    def pre_hook(mod, args, kwargs):
        if ctl['vecs'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, ctl['pos']] += ctl['vecs'].to(h.dtype)[:, None, :]
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def margins(p, pos, vecs, layer, tr, fa):
        hk = blocks[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        ctl.update(pos=pos, vecs=torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda'))
        try:
            def best(forms):
                o = np.full(len(vecs), -1e9)
                for f in forms:
                    a = enc(f)
                    lg = model(input_ids=torch.as_tensor([p + a] * len(vecs), device='cuda')).logits.float()
                    lpa = torch.log_softmax(lg[:, len(p) - 1:len(p) - 1 + len(a)], -1)
                    o = np.maximum(o, lpa.gather(-1, torch.as_tensor(a, device='cuda')[None, :, None].expand(len(vecs), -1, 1))
                                   .sum((1, 2)).cpu().numpy())
                return o
            return best(tr) - best(fa)
        finally:
            ctl['vecs'] = None
            hk.remove()

    t0 = time.time()
    k = C['k_grid']
    out = {}
    for lang in ('zh', 'en'):
        Sd = z[f'disc_{lang}_bare']                     # [n, 4, L, d]，格序 2n + t
        _, suf, tr, fa = MV1.GATE[lang]
        out[lang] = {}
        for l in C['steer_layers']:
            S = Sd[:, :, l].astype(np.float64)
            tmask = np.array([0, 1, 1, 0], bool)
            v = S[:, tmask].mean((0, 1)) - S[:, ~tmask].mean((0, 1))
            vt = S[:, [1, 3]].mean((0, 1)) - S[:, [0, 2]].mean((0, 1))       # 类别真假主效应 t（描述性对照）
            dirs = {'truth': v / np.linalg.norm(v), 't_main': vt / np.linalg.norm(vt)}
            sig = float((S.reshape(-1, S.shape[-1]) @ dirs['truth']).std())
            sig_t = float((S.reshape(-1, S.shape[-1]) @ dirs['t_main']).std())
            vecs = [kk * sig * dirs['truth'] for kk in k] + [kk * sig_t * dirs['t_main'] for kk in k] + \
                   [kk * sig * rd for rd in rand_dirs for kk in k]
            vals = np.zeros((len(ents), 4, len(vecs)))
            for i, r in enumerate(ents):
                for c, text in enumerate(bare(r, lang)):
                    ids = enc(text)
                    vals[i, c] = margins(ids + enc(suf), [len(ids) - 1], vecs, l, tr, fa)
            nk = len(k)
            i0 = k.index(0)
            D = vals - vals[:, :, [i0]]                                       # Δ(k) = 间隔(k) − 间隔(0)
            items = D.reshape(-1, D.shape[-1])
            op, tm = items[:, :nk], items[:, nk:2 * nk]
            rnd = np.stack([items[:, (2 + j) * nk:(3 + j) * nk] for j in range(len(rand_dirs))])
            vt_ = E.steer_verdict(op, rnd, k, {'min_swing': C['min_swing']})
            base = vals[:, :, i0]
            truth_lbl = np.array([0, 1, 1, 0], bool)
            flips = {'false_judged_true_at_k0': float(np.mean(base[:, ~truth_lbl] > 0)),
                     'false_judged_true_at_kmax': float(np.mean(vals[:, ~truth_lbl, k.index(max(k))] > 0)),
                     'true_judged_true_at_k0': float(np.mean(base[:, truth_lbl] > 0)),
                     'true_judged_true_at_kmin': float(np.mean(vals[:, truth_lbl, k.index(min(k))] > 0)),
                     'paired_order_at_k0': float(np.mean(np.r_[base[:, 1] > base[:, 0], base[:, 2] > base[:, 3]])),
                     'false_plus_kmax_above_true_k0': float(np.mean(np.r_[vals[:, 0, k.index(max(k))] > base[:, 1],
                                                                         vals[:, 3, k.index(max(k))] > base[:, 2]]))}
            out[lang][l] = {'sigma': sig, 'verdict': vt_, 't_main': E.steer_verdict(tm, rnd, k, {'min_swing': C['min_swing']}),
                            'cos_truth_tmain': float(dirs['truth'] @ dirs['t_main']),
                            'mean_delta_by_cell': {f'n{c >> 1}t{c & 1}': [float(x) for x in D[:, c, :nk].mean(0)] for c in range(4)},
                            'judgement': flips}
            print(f'@@progress {lang} L{l} swing={vt_["swing"]:.2f} rand={[round(x, 2) for x in vt_["random_swings"]]} '
                  f'ok={vt_["supported_write"]} t_main={out[lang][l]["t_main"]["swing"]:.2f} {round(time.time() - t0)}s', flush=True)
    verdict = {'H5': {lang: any(out[lang][l]['verdict']['supported_write'] for l in out[lang]) for lang in out}}
    save(model_path, {'H5_steer': out, 'seconds_steer': round(time.time() - t0, 1), 'torch': torch.__version__}, verdict)
    print(json.dumps(verdict, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze', 'steer'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model), 'analyze': lambda: do_analyze(a.model),
     'steer': lambda: do_steer(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
