# -*- coding: utf-8 -*-
"""XM-v1：G1–G5 核心结论的跨模型复现。

同一套材料与提示（EA-v1 / v2 实体与属性问答、MV-v2 框架 B 与裸陈述句、CF-v2 冲突上下文），在每个被测模型上用批量前向重新采集；
与层号有关的选择一律按规则在该模型的发现集上重新选定（或按相对深度换算），判据在 CRITERIA 中随预注册冻结。
被测模型：qwen3-4b（驱动一致性核对：批量采集应复现原结果）、qwen3-1.7b（同家族、更小）、glm4-9b-chat-hf（不同厂家；
16 GB 显存放不下 bf16，按 bitsandbytes LLM.int8 加载，结果单独分桶记为 int8）。
复现条目（每个模型分别判定）：
  C1  G1 类别中心：确认集类别准确率 ≥ 0.6 且比 hs[0] 高 0.15（两种语言；层由发现集 5 折最近中心交叉验证选定）。
  C2  G2 共享算子：颜色两种语言 supported_operator，且 4 个二值属性中至少 2 个两种语言都成立（EA-v2 规则）。
  C3  截止层：颜色 / 含水 × 中 / 英 4 项中至少 3 项「写入窗口非空且最后可读层 > 最后可写层」（层网格步长 ⌈L/12⌉）。
  C4  真值组合：成对行为门 ≥ 0.7 的语言上，框架 B 陈述句真值探针（中层段 [L/3, 0.6L] 内交叉验证选层）确认集 ≥ 0.75；至少一种语言过门。
  C5  中层跨语言：q 与 n 的中英 cos ≥ 0.3 且大于跨因子 |cos|，中文探针读英文确认集 ≥ 0.75（MV-v2 H4 规则，中层段）。
  C6  真值写入：在相对深度 13/36、17/36、21/36 的层上，句末沿真值方向写入，steer_verdict 至少一层成立（每种语言）。
  C7  仲裁开关：r = 1 童话来源，书里上下文胜出 ≥ 0.7 且现实先验胜出 ≥ 0.7（两种语言）。
  C8  知识门控：现实范围 γ 的 95% 区间上界 < 0，且 |γ_书里| ≤ 0.5 |γ_现实|（两种语言）。
  C9  分工（DLA）：书里注意力直接作用之和 > MLP 之和；现实 MLP 之和 < 注意力之和（两种语言）。
  C10 仲裁写入：在颜色写入窗口内摆幅最大的层削弱先验色方向，现实范围 steer_verdict 成立，且现实摆幅 > 2 × 书里摆幅（两种语言）。
  C11 拼装：AM-v1 的 14 格中，图谱 KL 同时低于 hs0 与 category 的至少 11 格。
用法：python scripts/xm_replicate_v1.py prereg | collect --model M [--int8] | steer --model M [--int8] | analyze --model M [--int8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import arbitration as A  # noqa: E402
from shared.dist_protocol import assembly as S  # noqa: E402
from shared.dist_protocol import entity_atlas as E  # noqa: E402
from shared.dist_protocol import mode_atlas as M  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import entity_atlas_v3 as V3  # noqa: E402
import mode_atlas_v1 as MV1  # noqa: E402
import mode_atlas_v2 as MV2  # noqa: E402
import arbitration_v1 as CF  # noqa: E402

PREREG = ROOT / 'research/cross_model/XM-v1.prereg.json'
MODELS = {'qwen3-4b': 'bf16', 'qwen3-1.7b': 'bf16', 'glm4-9b-chat-hf': 'int8'}
ATTRS = ['color', 'size', 'weight', 'water', 'edible']
CRITERIA = {
    'schema': 'cross_model.v1',
    'C1_acc_min': 0.6, 'C1_over_hs0': 0.15,
    'C2': 'EA-v2 G2_confirmed 规则（CRITERIA_V2）',
    'C3_min_items': 3, 'C3_grid': 'range(0, L, ceil(L/12))', 'read_margin_min': 0.10, 'write_swing_min': 1.0, 'k': 6, 'n_random': 2,
    'C4_gate_min': 0.7, 'C4_truth_min': 0.75, 'band': '[round(L/3), round(0.6L)]',
    'C5_cos_min': 0.3, 'C5_transfer_min': 0.75,
    'C6_layers_rel': [13 / 36, 17 / 36, 21 / 36], 'k_grid': [-6, -3, 0, 3, 6], 'min_swing': 1.0,
    'C7_win_min': 0.7, 'C8_book_ratio_max': 0.5, 'C10_world_over_book': 2.0, 'C11_min_cells': 11,
    'n_boot': 1000, 'seed': 0,
}


# ---------------- 预注册 ----------------

def prereg_doc():
    return {'id': 'XM-v1', 'schema': 'cross_model_prereg.v1', 'created': '2026-11-01',
            'basis': 'Phase 264–272（Qwen3-4B）的核心结论；本版只换模型，不换材料与判据',
            'models': {k: {'path': f'models/hf/{k}', 'precision': v} for k, v in MODELS.items()},
            'precision_note': 'GLM4-9B bf16 约 18 GB，16 GB 显存放不下；CPU 卸载实测每次前向约 0.6 s，故用 bitsandbytes LLM.int8（单独分桶，不与 bf16 结果平均）',
            'materials': {'EA': 'research/entity_atlas/EA-v2.prereg.json（提示、实体）', 'MV': 'research/mode_atlas/MV-v2.prereg.json（框架 B、裸陈述句、行为门）',
                          'CF': 'research/arbitration/CF-v2.prereg.json（上下文、问题、条件）'},
            'collection': '批量前向（右侧补齐，按每行真实长度取末位）；完整答案对数概率逐答案求和；与原单条前向的一致性在 qwen3-4b 上核对',
            'power_note': 'AGENTS.md 13 的 McNemar 规划针对逐题二值的消融对比；本版判据为连续量与比例阈值，沿用原 Phase 的 n（发现 210 / 确认 197 / 干预 56）',
            'doc': __doc__, 'criteria': CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（XM-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


# ---------------- 模型与批量前向 ----------------

def key_of(model_path, int8):
    from shared.dist_protocol.census_runner import model_snapshot
    snap = model_snapshot(model_path)['sha256']
    return snap, snap[:16] + ('-int8' if int8 else '-bf16')


def out_dir(key):
    return V1.native_dir(key).parents[2] / '_cross_model' / key / 'XM-v1'


def load(model_path, int8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)} if int8 else {}
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda', **kw).eval()
    return tok, model


class Runner:
    """批量前向：右侧补齐，按真实长度取位置。可选在第 layer 块输入上对指定位置加向量（每行一个向量）。"""

    def __init__(self, tok, model):
        import torch
        self.torch, self.tok, self.model = torch, tok, model
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        self.blocks = model.model.layers
        self.L = model.config.num_hidden_layers
        self.ctl = {'vecs': None, 'pos': None}

    def enc(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def _batch(self, seqs):
        torch = self.torch
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.as_tensor(s)
            att[i, :len(s)] = 1
        return ids.cuda(), att.cuda()

    def forward(self, seqs, hidden=False):
        ids, att = self._batch(seqs)
        return self.model(input_ids=ids, attention_mask=att, output_hidden_states=hidden)

    def last_states(self, seqs):
        o = self.forward(seqs, hidden=True)
        idx = self.torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
        ar = self.torch.arange(len(seqs), device='cuda')
        return self.torch.stack([h[ar, idx].float() for h in o.hidden_states], 1).cpu().numpy()      # [B, L+1, d]

    def answer_lp(self, prompt_ids, answers, logits_fn=None):
        """每个候选答案的完整答案对数概率。"""
        seqs = [prompt_ids + self.enc(a) for a in answers]
        lg = (logits_fn or (lambda s: self.forward(s).logits))(seqs).float()
        lp = self.torch.log_softmax(lg, -1)
        out = []
        p = len(prompt_ids)
        for i, a in enumerate(answers):
            ai = self.enc(a)
            out.append(float(sum(lp[i, p - 1 + j, t] for j, t in enumerate(ai))))
        return out

    # 干预：第 layer 块输入，pos 处加 vecs[b]
    def steer_logits(self, seqs, layer, pos_per_row, vecs):
        torch = self.torch
        V = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda')

        def hook(mod, args, kwargs):
            h = (args[0] if args else kwargs['hidden_states']).clone()
            for b, ps in enumerate(pos_per_row):
                h[b, ps] += V[b].to(h.dtype)
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs['hidden_states'] = h
            return args, kwargs
        hk = self.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            return self.forward(seqs).logits.float()
        finally:
            hk.remove()


def positions(tok, text, name):
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    spans, i = [], text.find(name)
    while i >= 0:
        spans.append((i, i + len(name)))
        i = text.find(name, i + 1)
    return [j for j, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]


def rows():
    return {'disc': V1.lexicon(), 'conf': V2.new_entities()}


# ---------------- 采集 ----------------

def do_collect(model_path, int8):
    import torch
    snap, key = key_of(model_path, int8)
    out = out_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(model_path, int8)
    R = Runner(tok, model)
    enc = R.enc
    p2 = json.loads(V2.PREREG.read_text(encoding='utf-8'))
    T = p2['targets']
    palette, forms = T['color']['palette'], T['color']['forms']
    fill = lambda s, r: s.replace('{X}', r['zh']).replace('{E}', r['en'])
    t0 = time.time()
    arrays = {}
    # 1 EA：实体中性编码（3 句式平均）与属性目标
    for s, rs in rows().items():
        for lang in ('zh', 'en'):
            codes = np.zeros((len(rs), R.L + 1, model.config.hidden_size), dtype=np.float16)
            col = np.zeros((len(rs), len(palette)))
            binv = {a: np.zeros(len(rs)) for a in V2.BINARY}
            for i, r in enumerate(rs):
                codes[i] = R.last_states([enc(fill(fr, r)) for fr in V1.FRAMES[lang]]).mean(0)
                pr = enc(fill(T['color']['prompt'][lang], r))
                flat = [(c, f) for c in palette for f in forms[lang][c]]
                lp = R.answer_lp(pr, [f for _, f in flat])
                for j, c in enumerate(palette):
                    col[i, j] = max(v for (cc, _), v in zip(flat, lp) if cc == c)
                for a in V2.BINARY:
                    sp = T[a][lang]
                    lp = R.answer_lp(enc(fill(sp['prompt'], r)), sp['pos'] + sp['neg'])
                    binv[a][i] = max(lp[:len(sp['pos'])]) - max(lp[len(sp['pos']):])
            arrays[f'{s}_{lang}_codes'], arrays[f'{s}_{lang}_color'] = codes, col
            for a in V2.BINARY:
                arrays[f'{s}_{lang}_{a}'] = binv[a]
            print(f'@@progress EA {s}/{lang} {round(time.time() - t0)}s', flush=True)
    # 2 MV：框架 B 与裸陈述句句末状态、行为门
    for s, rs in rows().items():
        for lang in ('zh', 'en'):
            HB = np.zeros((len(rs), 8, R.L + 1, model.config.hidden_size), dtype=np.float16)
            HS = np.zeros((len(rs), 4, R.L + 1, model.config.hidden_size), dtype=np.float16)
            gate = np.zeros((len(rs), 4))
            _, suf, tr, fa = MV1.GATE[lang]
            for i, r in enumerate(rs):
                HB[i] = R.last_states([enc(t) for t in MV2.frame_b(r, lang)])
                bare = [enc(t) for t in MV2.bare(r, lang)]
                HS[i] = R.last_states(bare)
                for c, b in enumerate(bare):
                    lp = R.answer_lp(b + enc(suf), tr + fa)
                    gate[i, c] = max(lp[:len(tr)]) - max(lp[len(tr):])
            arrays[f'{s}_{lang}_B'], arrays[f'{s}_{lang}_bare'], arrays[f'{s}_{lang}_gate'] = HB, HS, gate
            print(f'@@progress MV {s}/{lang} {round(time.time() - t0)}s', flush=True)
    # 3 CF：冲突上下文（P / K 用本模型自己的颜色目标）与逐层直接作用
    conds = CF.conditions()
    cap = {}
    hooks = []
    for li, b in enumerate(R.blocks):
        hooks.append(b.self_attn.register_forward_hook(lambda m, a, o, li=li: cap.__setitem__(('a', li), o[0].float())))
        hooks.append(b.mlp.register_forward_hook(lambda m, a, o, li=li: cap.__setitem__(('m', li), o.float())))
    hooks.append(model.model.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0].float())))
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: cap.__setitem__('emb', o.float())))
    Wu = model.lm_head.weight
    nw = model.model.norm.weight.float()
    eps = getattr(model.model.norm, 'variance_epsilon', getattr(model.model.norm, 'eps', 1e-6))
    for s, rs in rows().items():
        for lang in ('zh', 'en'):
            LO = np.zeros((len(rs), len(conds)))
            DLA = np.zeros((len(rs), len(conds), 2 * R.L + 1), dtype=np.float32)
            PK = np.zeros((len(rs), 3))
            for i, r in enumerate(rs):
                P, K, sv = CF.pick_pk(arrays[f'{s}_{lang}_color'][i], r['id'], lang)
                PK[i] = (P, K, sv)
                kw = forms[lang][palette[K]][0].strip()
                tk, tp = enc(forms[lang][palette[K]][0])[0], enc(forms[lang][palette[P]][0])[0]
                seqs = [enc(CF.prompt(r, lang, src, rr, Sx, kw)) for (src, rr, Sx) in conds]
                lg = R.forward(seqs).logits.float()
                last = torch.as_tensor([len(q) - 1 for q in seqs], device='cuda')
                ar = torch.arange(len(seqs), device='cuda')
                LO[i] = (lg[ar, last, tk] - lg[ar, last, tp]).cpu().numpy()
                du = (Wu[tk] - Wu[tp]).float()
                pre = cap['pre'][ar, last]
                scale = nw[None] / torch.sqrt((pre ** 2).mean(-1, keepdim=True) + eps)
                proj = lambda v: ((v[ar, last] * scale) @ du).cpu().numpy()
                DLA[i, :, 0] = proj(cap['emb'])
                for li in range(R.L):
                    DLA[i, :, 1 + 2 * li] = proj(cap['a', li])
                    DLA[i, :, 2 + 2 * li] = proj(cap['m', li])
            arrays[f'{s}_{lang}_LO'], arrays[f'{s}_{lang}_DLA'], arrays[f'{s}_{lang}_PK'] = LO, DLA, PK
            print(f'@@progress CF {s}/{lang} recon_err={float(np.abs(DLA.sum(-1) - LO).max()):.3f} {round(time.time() - t0)}s', flush=True)
    for h in hooks:
        h.remove()
    np.savez(out / 'data.npz', **arrays)
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'int8 (bitsandbytes LLM.int8)' if int8 else 'bf16',
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'n_blocks': R.L, 'hidden': model.config.hidden_size,
           'torch': torch.__version__, 'seconds': round(time.time() - t0, 1),
           'sha256_data': hashlib.sha256((out / 'data.npz').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


# ---------------- 共用：层规则与拟合 ----------------

def band(L):
    return list(range(round(L / 3), round(0.6 * L) + 1))


def grid(L):
    return list(range(0, L, math.ceil(L / 12)))


def report_path(model_path, int8):
    d = ROOT / 'research/cross_model' / f'XM-v1-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(rep.get(k), dict):
            rep[k].update(v)
        else:
            rep[k] = v
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def target(z, s, lang, attr):
    Y = z[f'{s}_{lang}_{attr}'].astype(float)
    return Y - Y.mean(1, keepdims=True) if attr == 'color' else Y


def cat_arrays():
    R_ = rows()
    return np.array([r['cat'] for r in R_['disc']]), np.array([r['cat'] for r in R_['conf']])


# ---------------- 干预：截止层扫描、真值写入、仲裁写入 ----------------

def do_steer(model_path, int8):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path, int8)
    with np.load(out_dir(key) / 'data.npz') as f:
        z = {k_: f[k_] for k_ in f.files if not k_.endswith('_B')}
    tok, model = load(model_path, int8)
    R = Runner(tok, model)
    enc = R.enc
    L = R.L
    p2 = json.loads(V2.PREREG.read_text(encoding='utf-8'))
    T = p2['targets']
    palette, forms = T['color']['palette'], T['color']['forms']
    dcat, ccat = cat_arrays()
    crow = V2.new_entities()
    cidx = {r['id']: i for i, r in enumerate(crow)}
    ents = [r for r in crow if r['id'] in {x['id'] for x in V3.pick_entities()}]
    rng = np.random.default_rng(C['seed'])
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(C['n_random'], model.config.hidden_size))]
    t0 = time.time()
    out = {'C3': {}, 'C6': {}, 'C10': {}}
    # C3：颜色（红 / 绿 / 紫）与含水，逐层读出与写入
    for attr in ('color', 'water'):
        for lang in ('zh', 'en'):
            Yd, Yc = target(z, 'disc', lang, attr), target(z, 'conf', lang, attr)
            Yd2 = Yd if Yd.ndim == 2 else Yd[:, None]; Yc2 = Yc if Yc.ndim == 2 else Yc[:, None]
            cm = {c: Yd2[dcat == c].mean(0) for c in set(dcat)}
            r2_cat = E.r2(Yc2, np.stack([cm[c] for c in ccat]))
            rows_ = []
            for l in grid(L):
                Xd = z[f'disc_{lang}_codes'][:, l].astype(np.float64)
                Xc = z[f'conf_{lang}_codes'][:, l].astype(np.float64)
                _, lam, _ = E.select_layer_lambda([Xd], Yd2, E.CRITERIA_V2['lambdas_rel'], 5, 0)
                W = E.ridge_fit(Xd, Yd2, lam)['W']
                read = E.r2(Yc2, E.ridge_predict(E.ridge_fit(Xd, Yd2, lam), Xc))
                dirs = ({c: W[:, palette.index(c)] - W.mean(1) for c in ('red', 'green', 'purple')} if attr == 'color' else {'+': W[:, 0]})
                sw, rsw = [], []
                for c, v in dirs.items():
                    v = v / np.linalg.norm(v)
                    sg = float((Xd @ v).std())
                    vecs = [np.zeros_like(v), C['k'] * sg * v, -C['k'] * sg * v] + [x * C['k'] * sg * rd for rd in rand for x in (1, -1)]
                    vals = []
                    for r in ents:
                        name = r['zh'] if lang == 'zh' else r['en']
                        if attr == 'color':
                            text = T['color']['prompt'][lang].replace('{X}', r['zh']).replace('{E}', r['en'])
                            ids = enc(text)
                            lg = R.steer_logits([ids] * len(vecs), l, [positions(tok, text, name)] * len(vecs), vecs)[:, len(ids) - 1]
                            lp = torch.log_softmax(lg, -1)[:, [enc(forms[lang][col][0])[0] for col in palette]]
                            vals.append((lp[:, palette.index(c)] - lp.mean(1)).cpu().numpy())
                        else:
                            sp = T['water'][lang]
                            text = sp['prompt'].replace('{X}', r['zh']).replace('{E}', r['en'])
                            p = enc(text)
                            pos = positions(tok, text, name)

                            def best(fs):
                                o = np.full(len(vecs), -1e9)
                                for f in fs:
                                    a = enc(f)
                                    lg = R.steer_logits([p + a] * len(vecs), l, [pos] * len(vecs), vecs)
                                    lpa = torch.log_softmax(lg[:, len(p) - 1:len(p) - 1 + len(a)], -1)
                                    o = np.maximum(o, lpa.gather(-1, torch.as_tensor(a, device='cuda')[None, :, None].expand(len(vecs), -1, 1))
                                                   .sum((1, 2)).cpu().numpy())
                                return o
                            vals.append(best(sp['pos']) - best(sp['neg']))
                    m = np.mean(vals, 0)
                    sw.append(float(m[1] - m[2]))
                    rsw.append([float(m[3 + 2 * j] - m[4 + 2 * j]) for j in range(len(rand))])
                s_ = float(np.mean(sw))
                rs_ = [float(np.mean([x[j] for x in rsw])) for j in range(len(rand))]
                rows_.append({'layer': l, 'read_margin': read - r2_cat, 'swing': s_, 'random': rs_,
                              'in_read': bool(read - r2_cat >= C['read_margin_min']), 'in_write': bool(s_ >= C['write_swing_min'] and s_ > max(rs_))})
            Rw = [x['layer'] for x in rows_ if x['in_read']]; Ww = [x['layer'] for x in rows_ if x['in_write']]
            out['C3'][f'{attr}/{lang}'] = {'rows': rows_, 'read_window': Rw, 'write_window': Ww,
                                          'pass': bool(Ww and Rw and max(Rw) > max(Ww))}
            print(f'@@progress C3 {attr}/{lang} R={Rw[:1] + Rw[-1:]} W={Ww[:1] + Ww[-1:]} {round(time.time() - t0)}s', flush=True)
    # C6：真值方向写入（裸陈述句句末）
    k = C['k_grid']
    for lang in ('zh', 'en'):
        _, suf, tr, fa = MV1.GATE[lang]
        out['C6'][lang] = {}
        for rel in C['C6_layers_rel']:
            l = round(rel * L)
            Sd = z[f'disc_{lang}_bare'][:, :, l].astype(np.float64)
            tm = np.array([0, 1, 1, 0], bool)
            v = Sd[:, tm].mean((0, 1)) - Sd[:, ~tm].mean((0, 1))
            v = v / np.linalg.norm(v)
            sg = float((Sd.reshape(-1, Sd.shape[-1]) @ v).std())
            vecs = [kk * sg * v for kk in k] + [kk * sg * rd for rd in rand for kk in k]
            D = []
            for r in ents:
                for text in MV2.bare(r, lang):
                    ids = enc(text)
                    p = ids + enc(suf)
                    seqs = lambda a: [p + enc(a)] * len(vecs)

                    def best(fs):
                        o = np.full(len(vecs), -1e9)
                        for f in fs:
                            a = enc(f)
                            lg = R.steer_logits(seqs(f), l, [[len(ids) - 1]] * len(vecs), vecs)
                            lpa = torch.log_softmax(lg[:, len(p) - 1:len(p) - 1 + len(a)], -1)
                            o = np.maximum(o, lpa.gather(-1, torch.as_tensor(a, device='cuda')[None, :, None].expand(len(vecs), -1, 1)).sum((1, 2)).cpu().numpy())
                        return o
                    mg = best(tr) - best(fa)
                    D.append(mg - mg[k.index(0)])
            D = np.array(D)
            nk = len(k)
            vv = E.steer_verdict(D[:, :nk], np.stack([D[:, (1 + j) * nk:(2 + j) * nk] for j in range(len(rand))]), k, {'min_swing': C['min_swing']})
            out['C6'][lang][str(l)] = vv
            print(f'@@progress C6 {lang} L{l} swing={vv["swing"]:.2f} ok={vv["supported_write"]} {round(time.time() - t0)}s', flush=True)
    # C10：仲裁写入（颜色写入窗口内摆幅最大的层）
    for lang in ('zh', 'en'):
        c3 = out['C3'][f'color/{lang}']
        cand = [x for x in c3['rows'] if x['in_write']]
        if not cand:
            out['C10'][lang] = {'layer': None, 'note': '颜色写入窗口为空'}
            continue
        l = max(cand, key=lambda x: x['swing'])['layer']
        Xd = z[f'disc_{lang}_codes'][:, l].astype(np.float64)
        Yd = target(z, 'disc', lang, 'color')
        _, lam, _ = E.select_layer_lambda([Xd], Yd, E.CRITERIA_V2['lambdas_rel'], 5, 0)
        W = E.ridge_fit(Xd, Yd, lam)['W']
        res = {Sx: {'op': [], 'rnd': [[] for _ in rand]} for Sx in (0, 1)}
        for r in ents:
            P, K, _ = CF.pick_pk(z[f'conf_{lang}_color'][cidx[r['id']]], r['id'], lang)
            v = W[:, P] - W.mean(1)
            v = v / np.linalg.norm(v)
            sg = float((Xd @ v).std())
            kw = forms[lang][palette[K]][0].strip()
            tk, tp = enc(forms[lang][palette[K]][0])[0], enc(forms[lang][palette[P]][0])[0]
            vecs = [-kk * sg * v for kk in k] + [kk * sg * rd for rd in rand for kk in k]
            for Sx in (0, 1):
                text = CF.prompt(r, lang, 'story', 1, Sx, kw)
                ids = enc(text)
                lg = R.steer_logits([ids] * len(vecs), l, [positions(tok, text, r['zh'] if lang == 'zh' else r['en'])] * len(vecs), vecs)[:, -1]
                lo = (lg[:, tk] - lg[:, tp]).cpu().numpy()
                d = lo - lo[k.index(0)]
                nk = len(k)
                res[Sx]['op'].append(d[:nk])
                for j in range(len(rand)):
                    res[Sx]['rnd'][j].append(d[(1 + j) * nk:(2 + j) * nk])
        out['C10'][lang] = {'layer': l}
        for Sx in (0, 1):
            out['C10'][lang]['book' if Sx else 'world'] = E.steer_verdict(np.array(res[Sx]['op']), np.array([np.array(x) for x in res[Sx]['rnd']]),
                                                                       k, {'min_swing': C['min_swing']})
        print(f'@@progress C10 {lang} L{l} world={out["C10"][lang]["world"]["swing"]:.2f} book={out["C10"][lang]["book"]["swing"]:.2f} '
              f'{round(time.time() - t0)}s', flush=True)
    save(model_path, int8, {'steer': out, 'seconds_steer': round(time.time() - t0, 1)})


# ---------------- 分析 ----------------

def do_analyze(model_path, int8):
    from provenance import write_result
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    C = pre['criteria']
    snap, key = key_of(model_path, int8)
    with np.load(out_dir(key) / 'data.npz') as f:
        z = {k_: f[k_] for k_ in f.files}                      # 一次读入内存，避免逐次访问重复读盘
    dcat, ccat = cat_arrays()
    drow, crow = rows()['disc'], rows()['conf']
    L = z['disc_zh_codes'].shape[1] - 1
    res, verdict = {}, {}
    lay = lambda s, lang: [z[f'{s}_{lang}_codes'][:, l].astype(np.float64) for l in range(L + 1)]
    LD = {lang: lay('disc', lang) for lang in ('zh', 'en')}
    LC = {lang: lay('conf', lang) for lang in ('zh', 'en')}
    # C1
    c1 = {}
    for lang in ('zh', 'en'):
        cv = []
        for l in range(L + 1):
            acc = []
            for p in np.array_split(np.random.default_rng(0).permutation(len(dcat)), 5):
                tr = np.setdiff1d(np.arange(len(dcat)), p)
                acc.append(E.nearest_centroid_acc(LD[lang][l][tr], dcat[tr], LD[lang][l][p], dcat[p]))
            cv.append(np.mean(acc))
        l = int(np.argmax(cv))
        a = E.nearest_centroid_acc(LD[lang][l], dcat, LC[lang][l], ccat)
        a0 = E.nearest_centroid_acc(LD[lang][0], dcat, LC[lang][0], ccat)
        c1[lang] = {'layer': l, 'acc': a, 'hs0': a0, 'pass': bool(a >= C['C1_acc_min'] and a - a0 >= C['C1_over_hs0'])}
    res['C1'] = c1
    verdict['C1'] = all(x['pass'] for x in c1.values())
    # C2
    c2 = {}
    for attr in ATTRS:
        c2[attr] = {}
        for lang, other in (('zh', 'en'), ('en', 'zh')):
            c2[attr][lang] = E.confirm_operator(LD[lang], target(z, 'disc', lang, attr), dcat, LC[lang], target(z, 'conf', lang, attr), ccat,
                                                conf_other_layers=LC[other])
    both = lambda a: all(c2[a][lang]['supported_operator'] for lang in ('zh', 'en'))
    res['C2'] = c2
    verdict['C2'] = bool(both('color') and sum(both(a) for a in ATTRS[1:]) >= 2)
    # C4 / C5（MV）
    bd = band(L)
    nd, nc = len(drow), len(crow)
    d = z['disc_zh_B'].shape[-1]
    X = lambda arr, cells, l: arr[:, cells, l].astype(np.float64).reshape(-1, d)
    bits = lambda cells, fn, n: np.tile([fn(c) for c in cells], n).astype(bool)
    truth = lambda c: ((c >> 1) & 1) ^ (c & 1)
    c4 = {}
    for lang in ('zh', 'en'):
        g = M.paired_gate(np.concatenate([z[f'disc_{lang}_gate'], z[f'conf_{lang}_gate']]))
        B = {s: z[f'{s}_{lang}_B'] for s in ('disc', 'conf')}
        pr = {l: M.probe(X(B['disc'], [0, 1, 2, 3], l), bits(range(4), truth, nd), np.repeat(np.arange(nd), 4),
                         X(B['conf'], [0, 1, 2, 3], l), bits(range(4), truth, nc), M.CRITERIA_V2) for l in bd}
        l = max(bd, key=lambda ll: pr[ll]['cv_acc'])
        c4[lang] = {'gate': g, 'gate_pass': bool(g['acc'] >= C['C4_gate_min']), 'layer': l, 'truth_acc': pr[l]['acc'],
                    'pass': (bool(pr[l]['acc'] >= C['C4_truth_min']) if g['acc'] >= C['C4_gate_min'] else 'gate_failed')}
    res['C4'] = c4
    verdict['C4'] = bool(any(c4[x]['gate_pass'] for x in c4) and all(c4[x]['pass'] is True for x in c4 if c4[x]['gate_pass']))
    effs = {l: (M.effects(M.center(z['disc_zh_B'][:, :, l].astype(np.float64))), M.effects(M.center(z['disc_en_B'][:, :, l].astype(np.float64))))
            for l in bd}
    sc = {l: np.mean([M.cos(effs[l][0][f], effs[l][1][f]) for f in ('q', 'n')]) for l in bd}
    l = max(bd, key=sc.get)
    c5 = {'layer': l}
    for f, bit in (('q', 2), ('n', 1)):
        ez, ee = effs[l]
        same = M.cos(ez[f], ee[f]); cross = max(abs(M.cos(ez[f], ee[g])) for g in ('q', 'n', 't') if g != f)
        fn = lambda c, b=bit: (c >> b) & 1
        p_ = M.probe(X(z['disc_zh_B'], list(range(8)), l), bits(range(8), fn, nd), np.repeat(np.arange(nd), 8),
                     X(z['conf_zh_B'], list(range(8)), l), bits(range(8), fn, nc), M.CRITERIA_V2,
                     X_te2=X(z['conf_en_B'], list(range(8)), l), y_te2=bits(range(8), fn, nc))
        c5[f] = {'cos_same': same, 'cross': cross, 'transfer': p_['acc_transfer'],
                 'pass': bool(same >= C['C5_cos_min'] and same > cross and p_['acc_transfer'] >= C['C5_transfer_min'])}
    res['C5'] = c5
    verdict['C5'] = bool(c5['q']['pass'] and c5['n']['pass'])
    # C7 / C8 / C9（CF）
    conds = [tuple(c) for c in CF.conditions()]
    ci = {c: j for j, c in enumerate(conds)}
    c7, c8, c9 = {}, {}, {}
    colour_pred = {}
    for lang in ('zh', 'en'):
        allLO = np.concatenate([z[f'disc_{lang}_LO'], z[f'conf_{lang}_LO']])
        cw = float(np.mean(allLO[:, ci['story', 1, 1]] > 0)); pw = float(np.mean(allLO[:, ci['story', 1, 0]] < 0))
        c7[lang] = {'context_win': cw, 'world_prior_win': pw, 'pass': bool(cw >= C['C7_win_min'] and pw >= C['C7_win_min'])}
        data = {}
        for s, n in (('disc', nd), ('conf', nc)):
            LO, PK = z[f'{s}_{lang}_LO'], z[f'{s}_{lang}_PK']
            rows_ = {k_: [] for k_ in ('S', 'src', 'r', 's', 'LO', 'ent')}
            for j, (src, r, Sx) in enumerate(conds):
                if r == 0:
                    continue
                for i in range(n):
                    for k_, v in zip(rows_, (Sx, int(src == 'ency'), r, PK[i, 2], LO[i, j], i)):
                        rows_[k_].append(v)
            data[s] = {k_: np.array(v, float) for k_, v in rows_.items()}
        law = A.law_eval(data['disc'], data['conf'], 's', {**A.CRITERIA, 'n_boot': C['n_boot']})
        gw, gb = law['coef']['s'], law['coef']['s'] + law['coef']['S_s']
        c8[lang] = {'gamma_world': gw, 'gamma_book': gb, 'ci_world': law['ci95']['s'], 'R2_conf': law['R2_conf'],
                    'pass': bool(law['ci95']['s'][1] < 0 and abs(gb) <= C['C8_book_ratio_max'] * abs(gw))}
        D = np.concatenate([z[f'disc_{lang}_DLA'], z[f'conf_{lang}_DLA']])
        b_, w_ = D[:, ci['story', 1, 1]].mean(0), D[:, ci['story', 1, 0]].mean(0)
        att = lambda a: float(a[1::2].sum()); mlp = lambda a: float(a[2::2].sum())
        c9[lang] = {'book_attn': att(b_), 'book_mlp': mlp(b_), 'world_attn': att(w_), 'world_mlp': mlp(w_),
                    'recon_max_abs_err': float(np.abs(D.sum(-1) - allLO).max()),
                    'book_attn_by_layer': b_[1::2].tolist(), 'world_mlp_by_layer': w_[2::2].tolist(),
                    'pass': bool(att(b_) > mlp(b_) and mlp(w_) < att(w_))}
    res.update({'C7': c7, 'C8': c8, 'C9': c9})
    for c_ in ('C7', 'C8', 'C9'):
        verdict[c_] = all(x['pass'] for x in res[c_].values())
    # C11（AM-v1 规则；属性层与 λ 取 C2 的选定值）
    c11, cells = {}, {}
    for lang in ('zh', 'en'):
        c11[lang] = {}
        for attr in ATTRS:
            Yd, Yc = target(z, 'disc', lang, attr), target(z, 'conf', lang, attr)
            l, lam = c2[attr][lang]['layer'], c2[attr][lang]['lambda_rel']
            Y2 = Yd if Yd.ndim == 2 else Yd[:, None]
            _, lam0, _ = E.select_layer_lambda([LD[lang][0]], Y2, E.CRITERIA_V2['lambdas_rel'], 5, 0)
            import g5_assembly_v1 as AM
            atl = AM.ridge_predictions(LD[lang][l], Yd, LC[lang][l], lam)
            h0 = AM.ridge_predictions(LD[lang][0], Yd, LC[lang][0], lam0)
            cm = {c: Yd[dcat == c].mean(0) for c in set(dcat)}
            catd = np.stack([np.mean([Yd[j] for j in range(len(Yd)) if dcat[j] == dcat[i] and j != i], 0) for i in range(len(Yd))])
            catc = np.stack([cm[c] for c in ccat])
            c11[lang][attr] = S.family_scores('cat' if attr == 'color' else 'bin', Yc, {'atlas': atl[1], 'hs0': h0[1], 'category': catc})
            if attr == 'color':
                colour_pred[lang] = {'atlas': atl, 'hs0': h0, 'category': (catd, catc)}
        data = {}
        for s, n in (('disc', nd), ('conf', nc)):
            LO, PK = z[f'{s}_{lang}_LO'], z[f'{s}_{lang}_PK']
            P, K = PK[:, 0].astype(int), PK[:, 1].astype(int)
            si = 0 if s == 'disc' else 1
            st = {m: colour_pred[lang][m][si][np.arange(n), P] - colour_pred[lang][m][si][np.arange(n), K] for m in colour_pred[lang]}
            rows_ = {k_: [] for k_ in ('S', 'src', 'r', 'LO', *st)}
            for j, (src, r, Sx) in enumerate(conds):
                if r == 0:
                    continue
                for i in range(n):
                    rows_['S'].append(Sx); rows_['src'].append(int(src == 'ency')); rows_['r'].append(r); rows_['LO'].append(LO[i, j])
                    for m in st:
                        rows_[m].append(st[m][i])
            data[s] = {k_: np.array(v, float) for k_, v in rows_.items()}
        preds = {}
        for m in ('atlas', 'hs0', 'category'):
            b = A.fit(A.design(data['disc']['S'], data['disc']['src'], data['disc']['r'], data['disc'][m]), data['disc']['LO'])
            preds[m] = A.design(data['conf']['S'], data['conf']['src'], data['conf']['r'], data['conf'][m]) @ b
        c11[lang]['arbitration'] = S.family_scores('bin', data['conf']['LO'], preds)
        gd, gc = z[f'disc_{lang}_gate'], z[f'conf_{lang}_gate']
        stated = lambda rs: [[(r['cat'] if (c & 1) else MV1.false_cat(r)) for c in range(4)] for r in rs]
        sd, scn = stated(drow), stated(crow)
        l1 = c1[lang]['layer']
        cls = {'atlas': AM.centroid_cv(LD[lang][l1], dcat, LC[lang][l1]), 'hs0': AM.centroid_cv(LD[lang][0], dcat, LC[lang][0]), 'category': (dcat, ccat)}
        th = lambda pc, st_: np.array([[float((pc[i] == st_[i][c]) != bool(c >> 1)) for c in range(4)] for i in range(len(st_))])
        preds = {}
        for m, (pd_, pc_) in cls.items():
            Td, Tc = th(pd_, sd), th(pc_, scn)
            P = np.zeros_like(gc)
            for nn in (0, 1):
                cols = [2 * nn, 2 * nn + 1]
                b = np.linalg.lstsq(np.c_[np.ones(Td[:, cols].size), Td[:, cols].ravel()], gd[:, cols].ravel(), rcond=None)[0]
                P[:, cols] = b[0] + b[1] * Tc[:, cols]
            preds[m] = P.ravel()
        c11[lang]['truth'] = S.family_scores('bin', gc.ravel(), preds)
        for f in c11[lang]:
            c11[lang][f]['recovered_atlas'] = S.recovered(c11[lang][f]['atlas'], c11[lang][f]['uniform'])
            cells[f'{f}/{lang}'] = bool(c11[lang][f]['atlas'] < c11[lang][f]['hs0'] and c11[lang][f]['atlas'] < c11[lang][f]['category'])
        c11[lang]['pooled_recovered'] = S.recovered(np.mean([c11[lang][f]['atlas'] for f in AM.FAMILIES]),
                                                    np.mean([c11[lang][f]['uniform'] for f in AM.FAMILIES]))
    res['C11'] = {'kl': c11, 'cells': cells}
    verdict['C11'] = sum(cells.values()) >= C['C11_min_cells']
    rep = save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                                  'precision': 'int8' if int8 else 'bf16', 'n_blocks': L, 'criteria': C, 'results': res, 'verdict': verdict})
    st = rep.get('steer')
    if st:
        verdict['C3'] = sum(x['pass'] for x in st['C3'].values()) >= C['C3_min_items']
        verdict['C6'] = all(any(v['supported_write'] for v in st['C6'][lang].values()) for lang in ('zh', 'en'))
        verdict['C10'] = all(st['C10'][lang].get('layer') is not None and st['C10'][lang]['world']['supported_write']
                             and st['C10'][lang]['world']['swing'] > C['C10_world_over_book'] * st['C10'][lang]['book']['swing'] for lang in ('zh', 'en'))
        rep = save(model_path, int8, {'verdict': verdict})
    p = write_result(f'xm_v1_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=C['seed'],
                     tool='scripts/xm_replicate_v1.py')
    print(json.dumps(verdict, ensure_ascii=False), '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'steer', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model, a.int8), 'steer': lambda: do_steer(a.model, a.int8),
     'analyze': lambda: do_analyze(a.model, a.int8)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
