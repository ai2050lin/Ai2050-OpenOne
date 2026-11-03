# -*- coding: utf-8 -*-
"""WS-v6：拟合单稳态局部积分模型，在留出的混合文本上预测逐位置的模式状态与代码倾向。

接 Phase 288（WS-v5）：模式状态不自持，证据中断后约 20 个 token 内消退；候选式为单稳态局部积分
  m_t ≈ m_0 + η Σ_{j≤t} A_tj e_j，A_tj ∝ exp(−(t−j)/τ + β m_t e_j)。
  证据 e_j：token 的词汇证据 = MF-v1 发现集单风格文本中 log P(token | 代码) − log P(token | 散文)（加 1 平滑；未见 token 为 0）。
  目标：混合文本（md / cmt）逐位置的 m_t = hs[33]·v_33 / Δμ_33（同 WS-v5），以及下一 token 的代码专有词对数质量。
  模型（发现集混合文本 = MF-v1 发现集任务；确认集 = MF-v5 的确认集任务混合文本）：
    M1 本 token：a + b·e_t
    M2 窗口平均：a + b·mean(e_{t−k+1..t})，k 在发现集上选
    M3 指数衰减：a + b·Σ w_tj e_j / Σ w_tj，w = exp(−(t−j)/τ)，τ 在发现集上选
    M4 模式匹配加权：同 M3 但 w = exp(−(t−j)/τ + β m̂_t e_j)，m̂ 先取 M3 的预测、再迭代 2 次；(τ, β) 在发现集上选（β = 0 即 M3）
  H1 确认集 R²：max(M2, M3)（发现集选出的较好者）− M1 ≥ 0.10。
  H2 M4 比 M3 高 ≥ 0.02，且发现集选出的 β > 0。
  H3 M3 选出的半衰期 τ·ln2 ∈ [4, 32] 个 token。
  H4 用 M3 / M4 中发现集较好者的超参数，重新拟合 (a, b) 预测代码专有词质量，确认集 R² ≥ 0.5。
被测模型：Qwen3-4B（bf16，不使用量化）。
用法：python scripts/weight_space_v6.py prereg | run | analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import weight_read as W  # noqa: E402
from shared.dist_protocol import mode_field as F  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402
import weight_space_v1 as WS1  # noqa: E402
import weight_space_v3 as WS3  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v6.prereg.json'
MODEL = WS1.MODEL
STY = F.STYLES
CODE = STY.index('code')
ML = 33
NATIVE = Path('D:/ai2050_native/_weight_space')


def prereg_doc():
    return {'id': 'WS-v6', 'schema': 'weight_space_prereg.v6', 'created': '2026-11-02',
            'basis': 'Phase 288（WS-v5）：模式状态不自持；本版拟合单稳态局部积分模型并做留出预测（AGENTS 5.4 预测保真度）',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'materials': '发现：WS-v3 的发现集混合文本；确认：mf_materials_v1.mixed_corpus（确认集任务）；词汇证据只用 MF-v1 发现集单风格文本',
            'inputs_at_prediction': '只用 token 序列与发现集拟合的参数；不用被预测位置的状态或输出',
            'power_note': '确认集 320 段混合文本的全部位置；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_V6}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v7）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v6-qwen3-4b'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def texts():
    disc = [''.join(s_ for s_, _ in it['pieces']) for it in WS3.mixed_disc()]
    conf = [c['text'] for c in MAT.mixed_corpus()]
    return disc, conf


def do_run():
    import torch
    snap, key = XM.key_of(MODEL, False)
    z = MF1.load_fields(key)
    y, tk, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    X = z['Hs'][D[sh], ML].astype(np.float32)
    v = W.unit(F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], CODE))
    pr = X @ v
    ys = y[sh][D[sh]]
    dmu = float(pr[ys == CODE].mean() - pr[ys != CODE].mean())
    del X, z
    gc.collect()
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    C, ids = MF1.tokenize(tok)
    cc, pc = Counter(), Counter()
    for c, i in zip(C, ids):
        if c['split'] == 'disc':
            (cc if c['style'] == 'code' else pc).update(i)
    ev = W.unigram_logodds(dict(cc), dict(pc), json.loads(PREREG.read_text(encoding='utf-8'))['criteria']['smooth'])
    excl = json.loads((MF1.out_dir(key) / 'vocab.json').read_text(encoding='utf-8'))['excl']
    EXc = torch.as_tensor(excl['code'], device='cuda')
    vt = torch.as_tensor(v, device='cuda', dtype=torch.float32)
    t0 = time.time()
    out = {}
    for split, tx in zip(('disc', 'conf'), texts()):
        rows = []
        for s in tx:
            seq = tok(s, add_special_tokens=False).input_ids
            i_, att = R._batch([seq])
            o = R.model(input_ids=i_, attention_mask=att, output_hidden_states=True)
            m = ((o.hidden_states[ML][0].float() @ vt) / dmu).cpu().numpy()
            mass = torch.logsumexp(torch.log_softmax(o.logits[0].float(), -1)[:, EXc], -1).cpu().numpy()
            rows.append({'e': np.array([ev.get(t, 0.0) for t in seq]), 'm': m, 'mass': mass})
        out[split] = rows
        print(f'@@{split} {len(rows)} {round(time.time() - t0)}s', flush=True)
    od = NATIVE / snap[:16] / 'WS-v6'
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / 'seqs.npz', **{f'{sp_}_{k_}_{i}': r[k_] for sp_, rows in out.items() for i, r in enumerate(rows) for k_ in ('e', 'm', 'mass')})
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16', 'dmu_33': dmu,
          'n_texts': {k_: len(v_) for k_, v_ in out.items()}, 'n_positions': {k_: int(sum(len(r['e']) for r in v_)) for k_, v_ in out.items()},
          'evidence_vocab': len(ev), 'seconds': round(time.time() - t0, 1)})
    print('done', flush=True)


def load_seqs(snap):
    Z = np.load(NATIVE / snap[:16] / 'WS-v6' / 'seqs.npz')
    out = {}
    for sp_ in ('disc', 'conf'):
        n = len([k_ for k_ in Z.files if k_.startswith(f'{sp_}_e_')])
        out[sp_] = [{k_: Z[f'{sp_}_{k_}_{i}'] for k_ in ('e', 'm', 'mass')} for i in range(n)]
    return out


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    S = load_seqs(rep['model_snapshot'])
    cat = lambda rows, f: np.concatenate([f(r) for r in rows])
    yd, yc = cat(S['disc'], lambda r: r['m']), cat(S['conf'], lambda r: r['m'])
    md, mc = cat(S['disc'], lambda r: r['mass']), cat(S['conf'], lambda r: r['mass'])

    def evaluate(feat, target='m'):
        fd, fc = cat(S['disc'], feat), cat(S['conf'], feat)
        td, tc = (yd, yc) if target == 'm' else (md, mc)
        coef = W.fit_linear(fd, td)
        pd_, pc_ = coef[0] + coef[1] * fd, coef[0] + coef[1] * fc
        return W.r2(td, pd_), W.r2(tc, pc_), coef

    res = {'M1': evaluate(lambda r: r['e'])}
    m2 = {k: evaluate(lambda r, k=k: W.window_feature(r['e'], k)) for k in crit['windows']}
    k2 = max(m2, key=lambda k: m2[k][0])
    res['M2'] = m2[k2]
    m3 = {t: evaluate(lambda r, t=t: W.matched_feature(r['e'], t, 0.0, np.zeros(len(r['e'])))) for t in crit['taus']}
    t3 = max(m3, key=lambda t: m3[t][0])
    res['M3'] = m3[t3]
    print('@@M1', res['M1'][:2], 'M2', k2, res['M2'][:2], 'M3', t3, res['M3'][:2], flush=True)

    def m4_feature(tau, beta):
        coef = res['M3'][2]
        mh = {id(r): coef[0] + coef[1] * W.matched_feature(r['e'], t3, 0.0, np.zeros(len(r['e']))) for sp_ in S for r in S[sp_]}
        for _ in range(crit['n_iter']):
            fe = {id(r): W.matched_feature(r['e'], tau, beta, mh[id(r)]) for sp_ in S for r in S[sp_]}
            fd = cat(S['disc'], lambda r: fe[id(r)])
            c = W.fit_linear(fd, yd)
            mh = {k_: c[0] + c[1] * f for k_, f in fe.items()}
        return fe

    m4 = {}
    for tau in crit['taus']:
        for beta in crit['betas']:
            fe = m4_feature(tau, beta) if beta > 0 else {id(r): W.matched_feature(r['e'], tau, 0.0, np.zeros(len(r['e']))) for sp_ in S for r in S[sp_]}
            m4[(tau, beta)] = (evaluate(lambda r: fe[id(r)]), fe)
    best4 = max(m4, key=lambda k_: m4[k_][0][0])
    res['M4'] = m4[best4][0]
    print('@@M4', best4, res['M4'][:2], flush=True)
    better = 'M4' if res['M4'][0] >= res['M3'][0] else 'M3'
    if better == 'M4':
        fe = m4[best4][1]
        mass_fit = evaluate(lambda r: fe[id(r)], 'mass')
    else:
        mass_fit = evaluate(lambda r: W.matched_feature(r['e'], t3, 0.0, np.zeros(len(r['e']))), 'mass')
    mass_own = evaluate(lambda r: r['e'], 'mass')
    integ = res['M2'] if res['M2'][0] >= res['M3'][0] else res['M3']
    h1 = {'conf_r2': {k_: v[1] for k_, v in res.items()}, 'disc_r2': {k_: v[0] for k_, v in res.items()},
          'integration_model': 'M2' if res['M2'][0] >= res['M3'][0] else 'M3', 'gain': integ[1] - res['M1'][1],
          'pass': bool(integ[1] - res['M1'][1] >= crit['H1_context_gain_min'])}
    h2 = {'best_tau_beta': list(best4), 'gain_over_M3': res['M4'][1] - res['M3'][1],
          'pass': bool(best4[1] > 0 and res['M4'][1] - res['M3'][1] >= crit['H2_matched_gain_min'])}
    hl = t3 * np.log(2)
    h3 = {'tau': t3, 'half_life_tokens': hl, 'window_k': k2,
          'pass': bool(crit['H3_halflife_range'][0] <= hl <= crit['H3_halflife_range'][1])}
    h4 = {'model': better, 'mass_conf_r2': mass_fit[1], 'mass_disc_r2': mass_fit[0], 'mass_own_token_conf_r2': mass_own[1],
          'pass': bool(mass_fit[1] >= crit['H4_mass_r2_min'])}
    desc = {'M2_by_k_conf': {str(k): v[1] for k, v in m2.items()}, 'M3_by_tau_conf': {str(t): v[1] for t, v in m3.items()},
            'M4_grid_disc': {f'{t}|{b}': v[0][0] for (t, b), v in m4.items()}, 'coef': {k_: list(v[2]) for k_, v in res.items()}}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass'], 'H4': h4['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4, 'describe': desc, 'verdict': verdict})
    p = write_result('ws_v6_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v6.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\nH4', h4, '\nM3 by tau', desc['M3_by_tau_conf'],
          '\nM2 by k', desc['M2_by_k_conf'], '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
