# -*- coding: utf-8 -*-
"""Goodfire 式原生神经元逆向分析（goodfire_probe.v1）的 GPU 部分：对一个 MLP 中间单元跑 S1–S8 与 S9 最小对。

S1  在普查语料切片（中心 census 作业的 tokens_S*.npz，与神经元普查同一份语料）上逐窗口前向到第 l 层的门控为止，
    记下每个计入位置上单元 j 的 g（gate_proj 输出、SiLU 之前）与残差流 h（post_attention_layernorm 的输入）；
    只保留 g 最大 / 最小的 k + m 个位置的 h（流式合并），另累计 h 的均值 μ、‖h‖ 的均值与 g 的均值 / 标准差。
S2–S5、S8  goodfire_probe.analyze（集中度、有效文本段、劈半、对偶、肩部留出、读 / 写方向）。
S6  Δ̂ 投影到输出侧：U (γ ⊙ Δ̂) 的前 / 后 20 个 token（U 用 get_output_embeddings，兼容不绑定词表的模型）。
S7  steering：在第 l 层解码块的输入上加 α‖h‖·Δ̂（α ∈ alphas，含 0 与正负两侧），记录各提示末位置
    「Δ 正端 20 词平均 logit − 负端 20 词平均 logit」、单元 j 在末位置的 g，以及贪心续写文本。
S9  最小对：每句取单元 j 的 g 最大值，换算为语料 z 分数，A 句预期在正端。

窗口与计入位置的口径与普查 runner 相同（每窗口前 skip_head 个位置与文档分隔符不计入；窗口之间不共享注意力）。
用法：python goodfire_runner.py --model <模型目录> --layer 18 --unit 2047 --job job.json --slices tokens_S0.npz ... \
        --spec spec.json --out result.json [--k 100 --shoulder 100 --batch 8 --device cuda:0]
进度：标准输出 @@progress {...} 行；结果 JSON 另附 delta.npy（Δ 原始向量，供归族比较）。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goodfire_probe as GP  # noqa: E402

RUNNER_VERSION = 'goodfire_runner.v1'
CTX_LEFT, CTX_RIGHT, N_SHOW, N_WORDS = 24, 8, 12, 20


def progress(**kw):
    print('@@progress ' + json.dumps(kw, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--unit', type=int, required=True)
    ap.add_argument('--job', required=True, help='普查作业 job.json（seq_len、skip_head、separator_id）')
    ap.add_argument('--slices', nargs='+', required=True)
    ap.add_argument('--spec', default='', help='{"prompts": [...], "alphas": [...], "gen_tokens": 40, "pairs": [{"a","b"}]}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--shoulder', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    job = json.loads(Path(a.job).read_text(encoding='utf-8'))
    spec = json.loads(Path(a.spec).read_text(encoding='utf-8')) if a.spec else {}
    seq_len, skip, sep = int(job['seq_len']), int(job['skip_head']), int(job['separator_id'])
    Ws = [np.load(p)['windows'] for p in a.slices]
    W = np.concatenate(Ws) if Ws else np.zeros((0, seq_len), np.int32)
    if W.shape[1] != seq_len:
        raise SystemExit('切片窗口长度与 job.json 不符')
    L, J, k, M = a.layer, a.unit, a.k, a.k + a.shoulder
    dev = a.device

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev)
    model.eval()
    dec = list(model.get_decoder().layers)
    if not 0 <= L < len(dec):
        raise SystemExit(f'层号越界：0–{len(dec) - 1}')
    blk = dec[L]
    F = int(blk.mlp.gate_proj.out_features)
    if not 0 <= J < F:
        raise SystemExit(f'单元编号越界：0–{F - 1}')
    env = {'runner': RUNNER_VERSION, 'torch': torch.__version__, 'transformers': transformers.__version__,
           'python': platform.python_version(), 'dtype': 'bf16', 'attn_implementation': getattr(model.config, '_attn_implementation', None),
           'gpu': torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'}

    class _Stop(Exception):
        pass
    ctx = {'h': None, 'g': None, 'stop': True, 'add': None}

    def pre_norm(mod, args):
        ctx['h'] = args[0]

    def gate_hook(mod, args, out):
        ctx['g'] = out[..., J]
        if ctx['stop']:
            raise _Stop()

    def steer_hook(mod, args, kwargs):
        if ctx['add'] is None:
            return None
        if args:
            return (args[0] + ctx['add'],) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs['hidden_states'] = kwargs['hidden_states'] + ctx['add']
        return args, kwargs
    blk.post_attention_layernorm.register_forward_pre_hook(pre_norm)
    blk.mlp.gate_proj.register_forward_hook(gate_hook)
    blk.register_forward_pre_hook(steer_hook, with_kwargs=True)

    # ---------- S1 两极采集 ----------
    d = int(model.config.hidden_size)
    pool = {s: {'g': torch.empty(0, device=dev), 'h': torch.empty(0, d, device=dev), 'w': torch.empty(0, dtype=torch.long, device=dev),
                'p': torch.empty(0, dtype=torch.long, device=dev), 'seg': torch.empty(0, dtype=torch.long, device=dev)} for s in ('top', 'bot')}
    sum_h = torch.zeros(d, dtype=torch.float64, device=dev)
    sum_hn = torch.zeros((), dtype=torch.float64, device=dev)
    sum_g = torch.zeros((), dtype=torch.float64, device=dev)
    sum_g2 = torch.zeros((), dtype=torch.float64, device=dev)
    n_counted, last = 0, 0.0
    pos = torch.arange(seq_len, device=dev)
    t1 = time.time()
    with torch.inference_mode():
        for i in range(0, len(W), a.batch):
            ids = torch.tensor(W[i:i + a.batch], dtype=torch.long, device=dev)
            try:
                model.model(input_ids=ids)
            except _Stop:
                pass
            mask = (pos[None, :] >= skip) & (ids != sep)
            seg = torch.cumsum((ids == sep).long(), dim=1)
            h = ctx['h'].float()[mask]                                   # (n, d)
            g = ctx['g'].float()[mask]
            wi = (torch.arange(ids.shape[0], device=dev)[:, None] + i).expand_as(ids)[mask]
            pi = pos[None, :].expand_as(ids)[mask]
            si = seg[mask]
            sum_h += h.double().sum(0)
            sum_hn += h.norm(dim=1).double().sum()
            sum_g += g.double().sum()
            sum_g2 += (g.double() ** 2).sum()
            n_counted += int(g.numel())
            for side, sign in (('top', 1.0), ('bot', -1.0)):
                P = pool[side]
                cg = torch.cat([P['g'], g])
                take = torch.topk(cg * sign, min(M, len(cg))).indices
                old = len(P['g'])

                def pick(prev, new):
                    both = torch.cat([prev, new])
                    return both[take]
                pool[side] = {'g': cg[take], 'h': pick(P['h'], h), 'w': pick(P['w'], wi), 'p': pick(P['p'], pi), 'seg': pick(P['seg'], si)}
                del old
            if time.time() - last > 3 or i + a.batch >= len(W):
                el = time.time() - t1
                progress(stage='S1', done=min(i + a.batch, len(W)), total=len(W), tokens=n_counted, tok_per_s=round(n_counted / max(el, 1e-6)))
                last = time.time()
    if n_counted < 2 * M:
        raise SystemExit(f'计入位置太少（{n_counted}），至少需要 {2 * M}')
    mu = (sum_h / n_counted).float().cpu().numpy()
    g_mean = float(sum_g / n_counted)
    g_std = float(np.sqrt(max(float(sum_g2 / n_counted) - g_mean ** 2, 0.0)))
    h_norm = float(sum_hn / n_counted)
    T = {s: {kk: v.cpu().numpy() for kk, v in pool[s].items()} for s in pool}

    # ---------- S2–S5、S8 ----------
    with torch.inference_mode():
        gpost = blk.post_attention_layernorm.weight.float()
        read_vec = (blk.mlp.gate_proj.weight[J].float() * gpost).cpu().numpy()
        write_vec = blk.mlp.down_proj.weight[:, J].float().cpu().numpy()
    texts = lambda s, sl: [(int(w), int(sg)) for w, sg in zip(T[s]['w'][sl], T[s]['seg'][sl])]  # noqa: E731
    res = GP.analyze(HP=T['top']['h'][:k], HN=T['bot']['h'][:k], SP=T['top']['h'][k:], SN=T['bot']['h'][k:], mu=mu,
                     textP=texts('top', slice(0, k)), textN=texts('bot', slice(0, k)), read_vec=read_vec, write_vec=write_vec,
                     gP=T['top']['g'][:k], gN=T['bot']['g'][:k], g_mean=g_mean, g_std=g_std)
    delta = res.pop('delta')
    progress(stage='S6')

    # ---------- S6 词表投影；S8 上游 ----------
    with torch.inference_mode():
        U = model.get_output_embeddings().weight
        gfin = model.model.norm.weight
        dh = torch.tensor(GP._unit(delta), device=U.device, dtype=torch.float32)
        s = (U.float() @ (gfin.float() * dh))
        top_ids = torch.topk(s, N_WORDS).indices.tolist()
        bot_ids = torch.topk(-s, N_WORDS).indices.tolist()
        sw = U.float() @ (gfin.float() * torch.tensor(GP._unit(write_vec), device=U.device))
        write_top = torch.topk(sw, 10).indices.tolist()
        write_bot = torch.topk(-sw, 10).indices.tolist()
        upstream = None
        if L > 0:
            Wd = dec[L - 1].mlp.down_proj.weight.float()                        # (d, F)
            r = torch.tensor(GP._unit(read_vec), device=Wd.device)
            c = (Wd.T @ r) / Wd.norm(dim=0).clamp_min(1e-12)
            idx = torch.topk(c.abs(), 5).indices.tolist()
            upstream = {'layer': L - 1, 'top': [{'unit': int(i), 'cos': float(c[i])} for i in idx],
                        'abs_p99': float(torch.quantile(c.abs(), 0.99)),
                        'note': 'cos(上一层 W_down[:, i], γ_post ⊙ W_gate[j])：只看直接通路，不含注意力与更早的层'}
    dec_tok = lambda ids: tok.decode([int(x) for x in ids], skip_special_tokens=False)  # noqa: E731

    def show(side):
        out = []
        for r_ in range(min(N_SHOW, k)):
            w, p = int(T[side]['w'][r_]), int(T[side]['p'][r_])
            row = W[w]
            out.append({'g': float(T[side]['g'][r_]), 'z': (float(T[side]['g'][r_]) - g_mean) / g_std if g_std else None,
                        'left': dec_tok(row[max(0, p - CTX_LEFT):p]), 'hit': dec_tok(row[p:p + 1]), 'right': dec_tok(row[p + 1:p + 1 + CTX_RIGHT])})
        return out

    # ---------- S7 steering ----------
    prompts = spec.get('prompts') or []
    alphas = [float(x) for x in (spec.get('alphas') or [-2, -1, 0, 1, 2])]
    gen_tokens = int(spec.get('gen_tokens') or 40)
    steer_rows, shift = [], []
    ctx['stop'] = False
    with torch.inference_mode():
        dvec = torch.tensor(GP._unit(delta), device=dev, dtype=torch.bfloat16)
        tid, bid = torch.tensor(top_ids, device=dev), torch.tensor(bot_ids, device=dev)
        for pi_, ptxt in enumerate(prompts):
            enc = tok(ptxt, return_tensors='pt').to(dev)
            row = []
            for al in alphas:
                ctx['add'] = (al * h_norm) * dvec if al else None
                lg = model(**enc).logits[0, -1].float()
                gl = float(ctx['g'][0, -1])
                out = model.generate(**enc, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                row.append({'alpha': al, 'shift': float(lg[tid].mean() - lg[bid].mean()), 'g_last': gl,
                            'text': tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)})
                ctx['add'] = None
            steer_rows.append({'prompt': ptxt, 'rows': row})
            shift.append([x['shift'] for x in row])
            progress(stage='S7', done=pi_ + 1, total=len(prompts))
        steer = GP.steering_monotone(np.array(shift), alphas) if prompts else None

        # ---------- S9 最小对 ----------
        za, zb, pair_rows = [], [], []
        for pr in spec.get('pairs') or []:
            zz = []
            for txt in (pr.get('a') or '', pr.get('b') or ''):
                enc = tok(txt, return_tensors='pt').to(dev)
                model.model(**enc)
                gmax = float(ctx['g'][0].float().max())
                zz.append((gmax - g_mean) / g_std if g_std else 0.0)
            za.append(zz[0]); zb.append(zz[1])
            pair_rows.append({'a': pr.get('a'), 'b': pr.get('b'), 'z_a': zz[0], 'z_b': zz[1], 'dz': zz[0] - zz[1]})
        pairs = GP.pairs_check(za, zb) if pair_rows else None

    out = {
        'schema': GP.SCHEMA, 'env': env, 'criteria': GP.CRITERIA,
        'params': {'layer': L, 'unit': J, 'k': k, 'shoulder': a.shoulder, 'batch': a.batch, 'alphas': alphas, 'gen_tokens': gen_tokens},
        'corpus': {'job_id': job.get('job_id'), 'corpus_fp': job.get('corpus_fp'), 'seq_len': seq_len, 'skip_head': skip,
                   'n_windows': int(len(W)), 'n_positions': n_counted, 'slices': [Path(p).name for p in a.slices],
                   'text_unit': '窗口内文档段（窗口 × 分隔符分段）；窗口之间不共享注意力'},
        'S1': {'h_norm_mean': h_norm, 'mu_norm': float(np.linalg.norm(mu)), 'g_mean': g_mean, 'g_std': g_std,
               'hook': 'h = post_attention_layernorm 的输入（norm 之前）；g = gate_proj 输出第 j 维（SiLU 之前）'},
        'reading': res['reading'], 'checks': res['checks'], 'g': res['g'], 'delta_norm': res['delta_norm'],
        'S6': {'pos': [dec_tok([i]) for i in top_ids], 'neg': [dec_tok([i]) for i in bot_ids],
               'write_pos': [dec_tok([i]) for i in write_top], 'write_neg': [dec_tok([i]) for i in write_bot],
               'note': 'U(γ⊙Δ̂)：Δ 的直接读出近似；write_* 为 U(γ⊙W_down[:, j])，对照用'},
        'S8': {'cos_read': res['checks']['S8_read_write']['cos_read'], 'cos_write': res['checks']['S8_read_write']['cos_write'],
               'upstream': upstream},
        'poles': {'pos': show('top'), 'neg': show('bot')},
        'S7': {'summary': steer, 'prompts': steer_rows, 'h_scale': h_norm, 'note': 'α 以 ‖h‖ 的倍数给出，加在第 l 层解码块的输入上（全部位置）'},
        'S9': {'summary': pairs, 'pairs': pair_rows},
        'seconds': {'total': round(time.time() - t0, 1), 'S1': round(time.time() - t1, 1)},
    }
    out['verdict'] = GP.verdict(res['checks'], steer, pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(a.out).with_name('delta.npy'), delta.astype(np.float32))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    progress(stage='done', seconds=out['seconds']['total'])


if __name__ == '__main__':
    for s_ in (sys.stdout, sys.stderr):
        try:
            s_.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    main()
