# -*- coding: utf-8 -*-
"""AM-v3：内容条件化的效应——风格 / 逻辑效应能否写成「加到残差状态上的向量」，且这个向量随事件内容变化。

材料与切分同 AM-v2（FU-v2 的 32 个事件 × 风格 × 逻辑 × 槽位，中 / 英；按事件 sha256 切分发现 16 / 确认 16），
真实下一词分布沿用 AM-v2 已采集的全词表对数概率（同一模型快照、同一批提示与批形状）。本版只新采两种状态（各格末位）：
  h   = 末层块输出（最终归一化的输入）
  key = hs[L // 2]（中层残差状态），只用 (日常, 因果, g) 提示的那一个，作为新事件的「内容键」
新事件只看到 (0,0,g) 格（其 Y00、h00 与 key），预测其余 9 格：
  B0 只用 Y00；B1 对数概率加法 Y00 + s·Δs_g + l·Δl_g（AM-v2 专家乘积）
  C2 状态加法：ĥ = h00 + s·Δh_s,g + l·Δh_l,g，经最终归一化 + unembedding 读出（Δh 为发现集平均）
  C3 内容条件化状态加法：(Δh_s, Δh_l) 由 key 经余弦核岭回归预测（发现集拟合，λ 留一选择，λ 大时退化为 C2）
  对照 C3-shuffle：打乱发现集里 key 与效应的对应 20 次。描述性：C3 用 h00 作键（「效应 = 作用于自身状态的算子」）、
  对数概率空间核岭回归、事件自身效应（含被预测格，泄漏上限，只作参照）。
KL 同 AM-v2：被预测格在该事件 12 个真实格各自前 20 的并集上，各自重新归一化。
  H1  KL(C2) ≤ 0.9 × KL(B1)（每种语言）
  H2  KL(C3) ≤ 0.9 × KL(C2) 且打乱键置换 p ≤ 0.05（每种语言）
  H3  1 − KL(C3) / KL(B0) ≥ 0.6（每种语言）
被测模型（bf16，不使用量化）：Qwen3-4B、Gemma-3-4B-it，分别判定。
用法：python scripts/assembly_v3.py prereg | run --model M | analyze --model M
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import assembly as S  # noqa: E402
import fu_materials_v2 as MAT  # noqa: E402
import fusion_v1 as FU1  # noqa: E402
import assembly_v2 as AM2  # noqa: E402

PREREG = ROOT / 'research/assembly/AM-v3.prereg.json'
NATIVE = AM2.NATIVE
LANGS = ['en', 'zh']
K = 3


def prereg_doc():
    return {'id': 'AM-v3', 'schema': 'assembly_prereg.v3', 'created': '2026-11-02',
            'basis': 'Phase 282（AM-v2）：平均效应迁移英文只恢复 0.43–0.44、中文 0.52–0.60；本版检验效应是否是随内容变化的残差状态向量',
            'models': {'qwen3-4b': 'bf16', 'gemma-3-4b-it': 'bf16'}, 'precision_rule': '不使用任何量化',
            'materials': 'scripts/fu_materials_v2.py（与 FU-v2 / AM-v2 相同的提示）', 'split': AM2.split(),
            'truth': 'AM-v2 已采集的 logp_{lang}.npy（本版用 h 的读出重建核对一致性）',
            'note_reuse': 'AM-v2 确认集只用于其预注册判定；本版的新预测模型 C2 / C3 在看到确认集结果前冻结',
            'power_note': '每种语言 16 个确认事件 × 3 槽位 × 9 格；连续量阈值；H2 另加 20 次打乱键置换',
            'doc': __doc__, 'criteria': S.CRITERIA_V3}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（AM-v4）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path):
    d = ROOT / 'research/assembly' / f'AM-v3-{Path(model_path).name}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, upd):
    p = report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def parts(model):
    """最终归一化模块、输出嵌入、块数（Qwen3 与 Gemma3 条件生成模型通用）。"""
    m = model.model
    lm = getattr(m, 'language_model', m)
    return lm.norm, model.get_output_embeddings(), len(lm.layers)


def do_run(model_path):
    import torch
    snap = FU1.snapshot(model_path)
    od = NATIVE / snap[:16] / 'AM-v3'
    od.mkdir(parents=True, exist_ok=True)
    tok, model = FU1.load(model_path)
    norm, _, L = parts(model)
    kl = int(L * S.CRITERIA_V3['key_layer_frac'])
    grab = {}
    hk = norm.register_forward_pre_hook(lambda mod, a: grab.__setitem__('h', a[0]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    for lang in LANGS:
        H, KY = [], []
        for it in MAT.ITEMS:
            seqs = [tok(MAT.prompt(it, lang, *c)).input_ids for c in MAT.cells()]
            n = max(map(len, seqs))
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.as_tensor(s)
                att[b, :len(s)] = 1
            o = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            ar = torch.arange(len(seqs), device='cuda')
            last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
            H.append(grab['h'][ar, last].float().cpu().numpy())
            KY.append(o.hidden_states[kl][ar, last].float().cpu().numpy())
            del o
        np.save(od / f'h_{lang}.npy', np.stack(H))                       # [n_event, 12, d]，float32
        np.save(od / f'key_{lang}.npy', np.stack(KY))
    hk.remove()
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'bf16', 'n_blocks': L, 'key_layer': kl,
           'seconds': round(time.time() - t0, 1), 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    save(model_path, man)
    print(json.dumps(man, ensure_ascii=False))


def do_analyze(model_path):
    import torch
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(model_path)
    od = NATIVE / snap[:16]
    sp = AM2.split()
    D, C = sp['disc'], sp['conf']
    _, model = FU1.load(model_path)
    norm, emb, _ = parts(model)
    W = emb.weight.float()

    def readout(Hs):
        out = []
        for i in range(0, len(Hs), 64):
            x = norm(torch.as_tensor(np.asarray(Hs[i:i + 64]), dtype=torch.float32, device='cuda'))
            out.append(torch.log_softmax(x.float() @ W.T, -1).cpu().numpy())
        return np.concatenate(out)

    rng = np.random.default_rng(crit['seed'])
    res = {}
    for lang in LANGS:
        A = np.load(od / 'AM-v2' / f'logp_{lang}.npy').astype(np.float32)   # [n, 12, V]
        n, _, V = A.shape
        Y = A.reshape(n, 2, 2, K, V)
        Hh = np.load(od / 'AM-v3' / f'h_{lang}.npy').reshape(n, 2, 2, K, -1)
        Ky = np.load(od / 'AM-v3' / f'key_{lang}.npy').reshape(n, 2, 2, K, -1)
        cand = [np.unique(np.argsort(-A[i], 1)[:, :crit['top_k_cand']]) for i in range(n)]
        recon = [S.kl_restricted(Y[i, 0, 0, g], r, cand[i]) for (i, g), r in
                 zip([(i, g) for i in C for g in range(K)], readout([Hh[i, 0, 0, g] for i in C for g in range(K)]))]
        kls = {}
        lams = {}

        def add(name, i, s_, l_, g, pred):
            kls.setdefault(name, []).append(S.kl_restricted(Y[i, s_, l_, g], pred, cand[i]))

        cells9 = [(s_, l_) for s_ in (0, 1) for l_ in (0, 1) if (s_, l_) != (0, 0)]
        for g in range(K):
            ds, dl, _ = S.slot_effects(Y[D, :, :, g])
            own_h = np.stack([np.concatenate(S.own_effects(Hh[j, :, :, g])) for j in D])        # [16, 2d]
            dh = own_h.mean(0)
            d = Hh.shape[-1]
            mk = S.kr_fit(Ky[D, 0, 0, g], own_h)
            mh = S.kr_fit(Hh[D, 0, 0, g], own_h)
            own_y = np.stack([np.concatenate(S.own_effects(Y[j, :, :, g])) for j in D])
            my = S.kr_fit(Y[D, 0, 0, g], own_y)
            lams[MAT.SLOTS[g]] = {'key': mk['lam'], 'h00': mh['lam'], 'logit': my['lam']}
            shuf = []
            for _ in range(crit['n_shuffle']):
                shuf.append(S.kr_fit(Ky[D, 0, 0, g][rng.permutation(len(D))], own_h))
            for i in C:
                y00, h00 = Y[i, 0, 0, g], Hh[i, 0, 0, g]
                pk = S.kr_predict(mk, Ky[i, 0, 0, g])[0]
                ph = S.kr_predict(mh, h00)[0]
                py = S.kr_predict(my, y00)[0]
                psh = [S.kr_predict(m_, Ky[i, 0, 0, g])[0] for m_ in shuf]
                oh = np.concatenate(S.own_effects(Hh[i, :, :, g]))
                oy = S.own_effects(Y[i, :, :, g])
                states, names = [], []
                for s_, l_ in cells9:
                    for nm, e in [('C2_state_mean', dh), ('C3_state_content', pk), ('C3_key_h00', ph), ('oracle_state_own', oh)] + \
                                 [(f'shuffle{t}', e_) for t, e_ in enumerate(psh)]:
                        states.append(h00 + s_ * e[:d] + l_ * e[d:])
                        names.append((nm, s_, l_))
                lp = readout(states)
                for (nm, s_, l_), r in zip(names, lp):
                    add(nm, i, s_, l_, g, r)
                for s_, l_ in cells9:
                    add('B0_base', i, s_, l_, g, y00)
                    add('B1_logit_mean', i, s_, l_, g, y00 + s_ * ds + l_ * dl)
                    add('C4_logit_content', i, s_, l_, g, y00 + s_ * py[:V] + l_ * py[V:])
                    add('oracle_logit_own', i, s_, l_, g, y00 + s_ * oy[0] + l_ * oy[1])
        m = {k_: float(np.mean(v)) for k_, v in kls.items() if not k_.startswith('shuffle')}
        sh = [float(np.mean(kls[f'shuffle{t}'])) for t in range(crit['n_shuffle'])]
        p_sh = (1 + sum(v <= m['C3_state_content'] for v in sh)) / (1 + len(sh))
        r = {'kl': m, 'shuffle_kl': sh, 'shuffle_p': p_sh, 'recon_kl_h00': float(np.mean(recon)), 'lambda': lams,
             'ratio_C2_B1': m['C2_state_mean'] / m['B1_logit_mean'], 'ratio_C3_C2': m['C3_state_content'] / m['C2_state_mean'],
             'recovered_C3': 1 - m['C3_state_content'] / m['B0_base'], 'recovered_B1': 1 - m['B1_logit_mean'] / m['B0_base'],
             'recovered_C2': 1 - m['C2_state_mean'] / m['B0_base']}
        res[lang] = r
        print(lang, {k_: round(v, 3) for k_, v in m.items()}, 'recon', round(r['recon_kl_h00'], 4), 'p_sh', round(p_sh, 3),
              'rec C3', round(r['recovered_C3'], 3), 'lam', lams, flush=True)
    verdict = {'H1': bool(all(res[l]['ratio_C2_B1'] <= crit['H1_state_ratio_max'] for l in LANGS)),
               'H2': bool(all(res[l]['ratio_C3_C2'] <= crit['H2_content_ratio_max'] and res[l]['shuffle_p'] <= crit['H2_shuffle_p_max'] for l in LANGS)),
               'H3': bool(all(res[l]['recovered_C3'] >= crit['H3_recovered_min'] for l in LANGS))}
    rep = save(model_path, {'criteria': crit, 'split': sp, 'results': res, 'verdict': verdict})
    p = write_result(f'am_v3_{Path(model_path).name}', rep, model_path=model_path, seed=crit['seed'], tool='scripts/assembly_v3.py')
    print(json.dumps(verdict), '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model), 'analyze': lambda: do_analyze(a.model)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
