# -*- coding: utf-8 -*-
"""统一机制评测基准 mech_bench.v1 的 GPU 运行器（判据与材料在 shared/dist_protocol/mech_bench.py）。

三步，分开执行，后一步核对前一步的排序哈希：
  discover  只用 discovery 划分，算出每种方法的组件排序，写 rankings.npz
  eval      在 --split confirm 或 lockbox 上评测；排序不再改动
  report    由逐样本结果生成 report.json

用法：
  python scripts/mech_bench_run.py discover --model D:/develop/model/gpt2-local --tag gpt2
  python scripts/mech_bench_run.py eval --model ... --tag gpt2 --split confirm
  python scripts/mech_bench_run.py report --tag gpt2 --split confirm
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import mech_bench as B  # noqa: E402

OUT = ROOT / 'research' / 'mech_bench' / 'MB-v1'
NATURAL = 'D:/ai2050_native/_census/corpus/2ca5f0a756d36c8a'
TOK_FOR_NATURAL = 'models/hf/qwen3-4b'


class Engine:
    """把 Attention 头、MLP 块和 MLP 神经元做成可替换组件。GPT-2 与 Qwen3 两种布局。"""

    def __init__(self, model, device):
        self.model, self.device = model, device
        cfg = model.config
        if hasattr(model, 'transformer'):
            self.arch, self.blocks = 'gpt2', list(model.transformer.h)
            self.H, self.d = cfg.n_head, cfg.n_embd
            self.dh = self.d // self.H
            self.out_proj = [b.attn.c_proj for b in self.blocks]
            self.mlps = [b.mlp for b in self.blocks]
            self.down = [b.mlp.c_proj for b in self.blocks]
            self.final_norm, self.embed = model.transformer.ln_f, model.transformer.wte
            self.F = self.down[0].weight.shape[0]
        else:
            self.arch, self.blocks = 'qwen3', list(model.model.layers)
            self.H, self.d = cfg.num_attention_heads, cfg.hidden_size
            self.dh = getattr(cfg, 'head_dim', None) or self.d // self.H
            self.out_proj = [b.self_attn.o_proj for b in self.blocks]
            self.mlps = [b.mlp for b in self.blocks]
            self.down = [b.mlp.down_proj for b in self.blocks]
            self.final_norm, self.embed = model.model.norm, model.model.embed_tokens
            self.F = self.down[0].weight.shape[1]
        self.L = len(self.blocks)
        self.n_nodes = self.L * self.H + self.L
        self.n_units = self.L * self.F
        self.grab = None            # None / 'plain' / 'grad'
        self.want_units = False
        self.z, self.m, self.u = {}, {}, {}
        self.rz, self.rm, self.ru = {}, {}, {}      # 层 → (替换掩码, 替换值)
        self.hooks = []
        for l in range(self.L):
            self.hooks.append(self.out_proj[l].register_forward_pre_hook(self._pre(l, self.z, self.rz, self.dh)))
            self.hooks.append(self.down[l].register_forward_pre_hook(self._pre(l, self.u, self.ru, 1, units=True)))
            self.hooks.append(self.mlps[l].register_forward_hook(self._post(l)))

    def _keep(self, store, l, x):
        if self.grab == 'grad':
            if x.requires_grad:
                x.retain_grad()
            store[l] = x
        elif self.grab == 'plain':
            store[l] = x.detach()

    def _pre(self, l, store, repl, width, units=False):
        def hook(mod, inp):
            x = inp[0]
            if l in repl:
                mask, val = repl[l]
                x = torch.where(mask.repeat_interleave(width), val.to(x.dtype), x)
            if not units or self.want_units:
                self._keep(store, l, x)
            return (x,)
        return hook

    def _post(self, l):
        def hook(mod, inp, out):
            if l in self.rm:
                mask, val = self.rm[l]
                if bool(mask):
                    out = val.to(out.dtype).expand_as(out)
            self._keep(self.m, l, out)
            return out
        return hook

    def clear(self):
        for repl in (self.rz, self.rm, self.ru):      # hook 闭包持有这些字典，只能原地清空
            repl.clear()

    def reset_store(self):
        for store in (self.z, self.m, self.u):
            store.clear()

    def forward_logits(self, ids=None, embeds=None, keep=1):
        kw = {'input_ids': ids} if embeds is None else {'inputs_embeds': embeds}
        return self.model(**kw, logits_to_keep=keep, use_cache=False).logits

    @staticmethod
    def _diff(logits, a, b):
        last = logits[:, -1]
        last = last if last.dtype == torch.float64 else last.float()
        return last.gather(1, a[:, None])[:, 0] - last.gather(1, b[:, None])[:, 0]

    @torch.no_grad()
    def ld(self, ids, a, b):
        return self._diff(self.forward_logits(ids), a, b)

    @torch.no_grad()
    def ld_embeds(self, emb, a, b):
        return self._diff(self.forward_logits(embeds=emb), a, b)

    @torch.no_grad()
    def capture(self, ids, neurons=False):
        self.grab, self.want_units = 'plain', neurons
        self.reset_store()
        self.forward_logits(ids)
        self.grab, self.want_units = None, False
        return {'z': dict(self.z), 'm': dict(self.m), 'u': dict(self.u)}

    def node_masks(self, keep):
        """keep: [n_nodes] 布尔。排列：头 l*H+h，其后 MLP L*H+l。"""
        heads = keep[:self.L * self.H].view(self.L, self.H)
        return heads, keep[self.L * self.H:]

    def set_nodes(self, keep, repl):
        keep = keep.to(self.device)
        heads, mlp = self.node_masks(keep)
        self.clear()
        for l in range(self.L):
            if not heads[l].all():
                self.rz[l] = (~heads[l], repl['z'][l])
            if not mlp[l]:
                self.rm[l] = (torch.tensor(True, device=self.device), repl['m'][l])

    def set_units(self, keep, repl):
        keep = keep.to(self.device).view(self.L, self.F)
        self.clear()
        for l in range(self.L):
            if not keep[l].all():
                self.ru[l] = (~keep[l], repl['u'][l])

    def embed_grad_probe(self, emb, pert, a, b):
        x = emb.clone().requires_grad_(True)
        self._diff(self.forward_logits(embeds=x), a, b).sum().backward()
        return ((pert - emb) * x.grad).sum()

    def head_write(self, l, z):
        """z: [..., H*dh] → 每头写入残差的向量 [..., H, d]。"""
        w = self.out_proj[l].weight
        w = w if self.arch == 'gpt2' else w.T                     # [H*dh, d]
        return torch.einsum('...hk,hkd->...hd', z.view(*z.shape[:-1], self.H, self.dh).float(), w.view(self.H, self.dh, -1).float())

    def unit_write_norm(self, l):
        w = self.down[l].weight
        return (w.float().norm(dim=1) if self.arch == 'gpt2' else w.float().norm(dim=0))

    def readout_vec(self, resid_last, a, b):
        """冻结最终归一化尺度后，把差异方向拉回残差空间。"""
        W = self.model.get_output_embeddings().weight.float()
        v = (W[a] - W[b]) * self.final_norm.weight.float()
        x = resid_last.float()
        if self.arch == 'gpt2':
            sd = (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(self.final_norm.eps).sqrt()
            return (v - v.mean(-1, keepdim=True)) / sd
        return v / x.pow(2).mean(-1, keepdim=True).add(self.final_norm.variance_epsilon).sqrt()


def batches(items, size):
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault((it['frame'], len(it['clean_ids'])), []).append(i)
    for key in sorted(groups):
        idx = groups[key]
        for s in range(0, len(idx), size):
            yield idx[s:s + size]


def tensors(items, idx, dev):
    t = lambda k: torch.tensor([items[i][k] for i in idx], device=dev)
    return t('clean_ids'), t('corrupt_ids'), t('a'), t('b')


def load(model_path, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    is_gpt2 = 'gpt2' in str(model_path).lower()
    dtype = torch.float32 if is_gpt2 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation='eager' if is_gpt2 else 'sdpa').to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, str(dtype).replace('torch.', '')


def snapshot(path):
    p = Path(path)
    h = hashlib.sha256()
    for f in sorted(p.glob('*.safetensors')) + [p / 'tokenizer.json']:
        with open(f, 'rb') as fh:
            h.update(f.name.encode())
            h.update(fh.read(1 << 24))           # 每个文件前 16 MiB + 大小，够区分快照
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def materials(tok, split):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    n = B.CRITERIA[f'n_{split}']
    return {t: B.build_items(t, split, n, enc) for t in B.TASKS}


def code_hash():
    return hashlib.sha256((Path(B.__file__).read_bytes() + Path(__file__).read_bytes())).hexdigest()[:16]


# ---------------- discover ----------------

def discover_task(eng, items, batch, dev, progress=None):
    """progress：可选回调 progress(已完成条数)，每批结束调用一次；不传时行为不变（MB-v1 / SP-v1）。"""
    L, H, F = eng.L, eng.H, eng.F
    n = len(items)
    zero = lambda: {'node': torch.zeros(eng.n_nodes, device=dev, dtype=torch.float64), 'unit': torch.zeros(L, F, device=dev, dtype=torch.float64)}
    atp, ig, act, dla = zero(), zero(), zero(), zero()
    exact = torch.zeros(eng.n_nodes, device=dev, dtype=torch.float64)
    mean_z = {l: 0 for l in range(L)}
    mean_m = {l: 0 for l in range(L)}
    mean_u = {l: 0 for l in range(L)}
    tokens = 0
    ld_clean, ld_corrupt = np.zeros(n), np.zeros(n)
    steps = B.CRITERIA['ig_steps']

    def grads_pass(emb_in, a, b):
        eng.grab, eng.want_units = 'grad', True
        eng.reset_store()
        x = emb_in.clone().requires_grad_(True)
        eng._diff(eng.forward_logits(embeds=x), a, b).sum().backward()
        eng.grab, eng.want_units = None, False
        gz = {l: eng.z[l].grad.float() for l in range(L)}
        gm = {l: eng.m[l].grad.float() for l in range(L)}
        gu = {l: eng.u[l].grad.float() for l in range(L)}
        eng.reset_store()
        return gz, gm, gu

    def add_scores(acc, gz, gm, gu, cl, cf, weight=1.0):
        for l in range(L):
            dz = (cf['z'][l].float() - cl['z'][l].float()) * gz[l]
            acc['node'][l * H:(l + 1) * H] += weight * dz.view(*dz.shape[:2], H, eng.dh).sum(dim=(0, 1, 3)).double()
            acc['node'][L * H + l] += weight * ((cf['m'][l].float() - cl['m'][l].float()) * gm[l]).sum().double()
            acc['unit'][l] += weight * ((cf['u'][l].float() - cl['u'][l].float()) * gu[l]).sum(dim=(0, 1)).double()

    for idx in batches(items, batch):
        clean, corrupt, a, b = tensors(items, idx, dev)
        tokens += clean.numel()
        cf = eng.capture(corrupt, neurons=True)
        cl = eng.capture(clean, neurons=True)
        ld_clean[idx] = eng.ld(clean, a, b).cpu().numpy()
        ld_corrupt[idx] = eng.ld(corrupt, a, b).cpu().numpy()
        e_clean, e_cf = eng.embed(clean).detach(), eng.embed(corrupt).detach()
        gz, gm, gu = grads_pass(e_clean, a, b)
        add_scores(atp, gz, gm, gu, cl, cf)
        for s in range(steps):
            alpha = (s + 0.5) / steps
            gz, gm, gu = grads_pass(e_cf + alpha * (e_clean - e_cf), a, b)
            add_scores(ig, gz, gm, gu, cl, cf, 1.0 / steps)
        with torch.no_grad():
            eng.grab = None
            box = {}
            hk = eng.final_norm.register_forward_pre_hook(lambda m, i: box.__setitem__('x', i[0].detach()))
            eng.forward_logits(clean)
            hk.remove()
            v = eng.readout_vec(box['x'][:, -1], a, b)
            for l in range(L):
                wc, wf = eng.head_write(l, cl['z'][l]), eng.head_write(l, cf['z'][l])
                diff = wf - wc
                act['node'][l * H:(l + 1) * H] += diff.norm(dim=-1).sum(dim=(0, 1)).double()
                dla['node'][l * H:(l + 1) * H] += torch.einsum('bhd,bd->h', diff[:, -1], v).double()
                dm = cf['m'][l].float() - cl['m'][l].float()
                act['node'][L * H + l] += dm.norm(dim=-1).sum().double()
                dla['node'][L * H + l] += (dm[:, -1] * v).sum().double()
                du = (cf['u'][l].float() - cl['u'][l].float()).abs().sum(dim=(0, 1))
                act['unit'][l] += (du * eng.unit_write_norm(l)).double()
                mean_z[l] = mean_z[l] + cl['z'][l].double().sum(dim=(0, 1))
                mean_m[l] = mean_m[l] + cl['m'][l].double().sum(dim=(0, 1))
                mean_u[l] = mean_u[l] + cl['u'][l].double().sum(dim=(0, 1))
            base = torch.as_tensor(ld_clean[idx], device=dev)
            for c in range(eng.n_nodes):
                keep = torch.ones(eng.n_nodes, dtype=torch.bool)
                keep[c] = False
                eng.set_nodes(keep, cf)
                exact[c] += (eng.ld(clean, a, b) - base).sum().double()
            eng.clear()
        if progress is not None:
            progress(int(len(idx)))
    scale = 1.0 / n
    return {
        'node': {'act_diff': (act['node'] * scale).cpu().numpy(), 'dla': (dla['node'] * scale).abs().cpu().numpy(),
                 'atp': (atp['node'] * scale).abs().cpu().numpy(), 'atp_ig': (ig['node'] * scale).abs().cpu().numpy(),
                 'exact': (exact * scale).abs().cpu().numpy()},
        'unit': {'act_diff': (act['unit'] * scale).reshape(-1).cpu().numpy(), 'atp_ig': (ig['unit'] * scale).abs().reshape(-1).cpu().numpy(),
                 'atp': (atp['unit'] * scale).abs().reshape(-1).cpu().numpy()},
        'signed_dla': (dla['node'] * scale).cpu().numpy(),
        'means': {'z': {l: (mean_z[l] / tokens).float() for l in range(L)}, 'm': {l: (mean_m[l] / tokens).float() for l in range(L)},
                  'u': {l: (mean_u[l] / tokens).float() for l in range(L)}},
        'gate': B.behavior_gate(ld_clean, ld_corrupt), 'n': n,
    }


def rank(score):
    return np.argsort(-np.nan_to_num(score, nan=-np.inf), kind='stable')


def discover(args):
    dev = 'cuda'
    model, tok, dtype = load(args.model, dev)
    eng = Engine(model, dev)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    mats = materials(tok, 'discovery')
    arrays, meta = {}, {'tasks': {}, 'model': str(args.model), 'snapshot': snapshot(args.model), 'dtype': dtype, 'arch': eng.arch,
                        'L': eng.L, 'H': eng.H, 'F': eng.F, 'design_sha': B.design_sha(code_hash()), 'code': code_hash(),
                        'torch': torch.__version__}
    means = {}
    for task, mat in mats.items():
        t0 = time.time()
        items = mat['items']
        r = discover_task(eng, items, args.batch, dev)
        for m, s in r['node'].items():
            arrays[f'{task}|node|{m}'] = s
        for m, s in r['unit'].items():
            arrays[f'{task}|unit|{m}'] = s.astype(np.float32)
        arrays[f'{task}|signed_dla'] = r['signed_dla']
        means[task] = r['means']
        meta['tasks'][task] = {'n': len(items), 'dropped': mat['dropped'], 'n_families': mat['n_families'], 'gate': r['gate'],
                               'seconds': round(time.time() - t0, 1), 'examples': [{k: i[k] for k in ('clean', 'corrupt', 'answer', 'contrast')} for i in items[:3]]}
        print(task, json.dumps(meta['tasks'][task]['gate']), meta['tasks'][task]['seconds'], flush=True)
    np.savez_compressed(out / 'rankings.npz', **arrays)
    torch.save(means, out / 'means.pt')
    meta['rankings_sha256'] = hashlib.sha256((out / 'rankings.npz').read_bytes()).hexdigest()
    (out / 'discovery.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print('rankings', meta['rankings_sha256'][:16])


# ---------------- eval ----------------

def group_of(eng):
    g = np.concatenate([np.repeat(np.arange(eng.L), eng.H), eng.L + np.arange(eng.L)])
    return g


def keep_from(order, k, n):
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.as_tensor(order[:k].copy())] = True
    return keep


def matched_keep(order, k, groups, rng):
    """与 order 前 k 个分层（层 × 类型）数量相同的随机集合。"""
    chosen = order[:k]
    keep = np.zeros(len(order), dtype=bool)
    for g, c in zip(*np.unique(groups[chosen], return_counts=True)):
        pool = np.flatnonzero(groups == g)
        keep[rng.choice(pool, c, replace=False)] = True
    return torch.as_tensor(keep)


def node_plan(eng, ranks, task, rng):
    """返回 [(key, keep 张量, 消融, 方向)]。"""
    n, plan = eng.n_nodes, []
    ks = dict(zip(B.BUDGETS, B.budget_counts(n, B.BUDGETS)))
    groups = group_of(eng)
    rand_orders = [rng.permutation(n) for _ in range(B.CRITERIA['n_random'])]
    for b, k in ks.items():
        for m in B.METHODS:
            keep = keep_from(ranks[f'{task}|node|{m}'], k, n)
            for abl in B.ABLATIONS:
                plan.append((B.make_key('N', 'keep', abl, m, b), keep, abl))
            plan.append((B.make_key('N', 'remove', 'cf', m, b), ~keep, 'cf'))
        for d, order in enumerate(rand_orders):
            keep = keep_from(order, k, n)
            plan.append((B.make_key('N', 'keep', 'cf', 'random', b, draw=d), keep, 'cf'))
            plan.append((B.make_key('N', 'remove', 'cf', 'random', b, draw=d), ~keep, 'cf'))
            mk = matched_keep(ranks[f'{task}|node|atp_ig'], k, groups, rng)
            plan.append((B.make_key('N', 'keep', 'cf', 'matched', b, draw=d), mk, 'cf'))
    return plan


def cross_plan(eng, ranks, task):
    plan, n = [], eng.n_nodes
    for src in B.TASKS:
        if src == task:
            continue
        for b, k in zip(B.OFF_TARGET_BUDGETS, B.budget_counts(n, B.OFF_TARGET_BUDGETS)):
            for m in B.METHODS:
                plan.append((B.make_key('N', 'remove', 'cf', m, b, src=src), ~keep_from(ranks[f'{src}|node|{m}'], k, n), 'cf'))
    return plan


def unit_plan(eng, ranks, task, rng):
    n, plan = eng.n_units, []
    orders = {m: ranks[f'{task}|unit|{m}'] for m in ('act_diff', 'atp', 'atp_ig')}
    rand_orders = [rng.permutation(n) for _ in range(B.CRITERIA['n_random'])]
    for b, k in zip(B.UNIT_BUDGETS, B.budget_counts(n, B.UNIT_BUDGETS)):
        for m, order in orders.items():
            keep = keep_from(order, k, n)
            plan.append((B.make_key('U', 'keep', 'cf', m, b), keep, 'cf'))
            plan.append((B.make_key('U', 'remove', 'cf', m, b), ~keep, 'cf'))
        for d, order in enumerate(rand_orders):
            plan.append((B.make_key('U', 'keep', 'cf', 'random', b, draw=d), keep_from(order, k, n), 'cf'))
    return plan


def constant_repl(means, eng, kind):
    if kind == 'mean':
        return {'z': {l: means['z'][l].to(eng.device) for l in range(eng.L)}, 'm': {l: means['m'][l].to(eng.device) for l in range(eng.L)}}
    return {'z': {l: torch.zeros(eng.H * eng.dh, device=eng.device) for l in range(eng.L)},
            'm': {l: torch.zeros(eng.d, device=eng.device) for l in range(eng.L)}}


def eval_task(eng, task, items, ranks, means, batch, rng):
    n = len(items)
    plan_n = node_plan(eng, ranks, task, rng) + cross_plan(eng, ranks, task)
    plan_u = unit_plan(eng, ranks, task, rng)
    raw = {k: np.zeros(n, dtype=np.float32) for k, _, _ in plan_n + plan_u}
    extra = ['clean', 'corrupt'] + [f'empty|N|{a}' for a in B.ABLATIONS] + [f'empty_keep|N|{a}' for a in B.ABLATIONS] + ['empty|U|cf', 'empty_keep|U|cf']
    raw.update({k: np.zeros(n, dtype=np.float32) for k in extra})
    consts = {a: constant_repl(means[task], eng, a) for a in ('mean', 'zero')}
    none_n = torch.zeros(eng.n_nodes, dtype=torch.bool)
    none_u = torch.zeros(eng.n_units, dtype=torch.bool)
    for idx in batches(items, batch):
        clean, corrupt, a, b = tensors(items, idx, eng.device)
        cf = eng.capture(corrupt, neurons=True)
        repl = {'cf': cf, **consts}
        raw['clean'][idx] = eng.ld(clean, a, b).cpu().numpy()
        raw['corrupt'][idx] = eng.ld(corrupt, a, b).cpu().numpy()
        for abl in B.ABLATIONS:
            eng.set_nodes(none_n, repl[abl])
            v = eng.ld(clean, a, b).cpu().numpy()
            raw[f'empty|N|{abl}'][idx] = v
            raw[f'empty_keep|N|{abl}'][idx] = v if abl != 'cf' else raw['corrupt'][idx] * 0 + v
        for key, keep, abl in plan_n:
            eng.set_nodes(keep, repl[abl])
            raw[key][idx] = eng.ld(clean, a, b).cpu().numpy()
        eng.set_units(none_u, cf)
        v = eng.ld(clean, a, b).cpu().numpy()
        raw['empty|U|cf'][idx] = v
        raw['empty_keep|U|cf'][idx] = v
        for key, keep, _ in plan_u:
            eng.set_units(keep, cf)
            raw[key][idx] = eng.ld(clean, a, b).cpu().numpy()
        eng.clear()
    return raw


def natural_windows(tok, n_windows, seq):
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(TOK_FOR_NATURAL)
    out = []
    for name in ('wikipedia-en', 'fineweb-edu-en'):
        stream = np.load(f'{NATURAL}/tokens_{name}.npz')['tokens']
        text = src.decode(stream[-40000:].tolist())         # 语料末尾，普查与 transcoder 训练窗口之外
        ids = tok.encode(text, add_special_tokens=False)
        out += [ids[i * seq:(i + 1) * seq] for i in range(n_windows // 2)]
    return torch.tensor(out)


@torch.no_grad()
def natural_kl(eng, tok, ranks, batch=4, n_windows=32, seq=128):
    wins = natural_windows(tok, n_windows, seq).to(eng.device)
    sums = {'z': {l: 0 for l in range(eng.L)}, 'm': {l: 0 for l in range(eng.L)}}
    for s in range(0, len(wins), batch):
        cap = eng.capture(wins[s:s + batch])
        for l in range(eng.L):
            sums['z'][l] = sums['z'][l] + cap['z'][l].double().mean(dim=(0, 1))
            sums['m'][l] = sums['m'][l] + cap['m'][l].double().mean(dim=(0, 1))
    steps = math.ceil(len(wins) / batch)
    means = {k: {l: (v[l] / steps).float() for l in v} for k, v in sums.items()}
    rng = np.random.default_rng(1)
    n = eng.n_nodes
    sets = {}
    for b, k in zip(B.OFF_TARGET_BUDGETS, B.budget_counts(n, B.OFF_TARGET_BUDGETS)):
        for task in B.TASKS:
            for m in B.METHODS:
                sets[f'{task}|{m}|{b:g}'] = ~keep_from(ranks[f'{task}|node|{m}'], k, n)
        for d in range(B.CRITERIA['n_random']):
            sets[f'random|{d}|{b:g}'] = ~keep_from(rng.permutation(n), k, n)
    kl = {k: 0.0 for k in sets}
    for s in range(0, len(wins), batch):
        w = wins[s:s + batch]
        eng.clear()
        base = torch.log_softmax(eng.forward_logits(w, keep=0).float(), -1)
        for key, keep in sets.items():
            eng.set_nodes(keep, means)
            lp = torch.log_softmax(eng.forward_logits(w, keep=0).float(), -1)
            kl[key] += float((base.exp() * (base - lp)).sum(-1).mean())
            del lp
        eng.clear()
    return {k: v / steps for k, v in kl.items()}


def evaluate(args):
    dev = 'cuda'
    out = OUT / args.tag
    disc = json.loads((out / 'discovery.json').read_text(encoding='utf-8'))
    body = (out / 'rankings.npz').read_bytes()
    if hashlib.sha256(body).hexdigest() != disc['rankings_sha256']:
        raise SystemExit('rankings.npz 与 discovery.json 记录的哈希不一致')
    if disc['code'] != code_hash() and not args.allow_code_change:
        raise SystemExit('代码已变，排序须重新生成')
    z = np.load(out / 'rankings.npz')
    ranks = {k: rank(z[k]) for k in z.files if k.count('|') == 2}
    means = torch.load(out / 'means.pt', weights_only=False)
    model, tok, _ = load(args.model, dev)
    eng = Engine(model, dev)
    mats = materials(tok, args.split)
    meta = {'split': args.split, 'tasks': {}, 'rankings_sha256': disc['rankings_sha256']}
    for task, mat in mats.items():
        if not disc['tasks'][task]['gate']['pass'] and not args.include_failed:
            meta['tasks'][task] = {'skipped': 'behavior_gate', 'n': len(mat['items'])}
            print(task, 'skipped: behavior gate', flush=True)
            continue
        t0 = time.time()
        rng = np.random.default_rng(int(hashlib.sha256(f'{task}|{args.split}'.encode()).hexdigest()[:8], 16))
        raw = eval_task(eng, task, mat['items'], ranks, means, args.batch, rng)
        np.savez_compressed(out / f'raw_{args.split}_{task}.npz', **raw, clusters=np.array([i['family'] for i in mat['items']]))
        meta['tasks'][task] = {'n': len(mat['items']), 'dropped': mat['dropped'], 'n_families': mat['n_families'],
                               'gate': B.behavior_gate(raw['clean'], raw['corrupt']), 'seconds': round(time.time() - t0, 1)}
        print(task, meta['tasks'][task], flush=True)
    t0 = time.time()
    meta['natural_kl'] = natural_kl(eng, tok, ranks)
    meta['natural_seconds'] = round(time.time() - t0, 1)
    (out / f'eval_{args.split}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


# ---------------- dict：同层同 k 的字典特征与原生神经元 ----------------

def dict_step(args):
    from shared.dist_protocol import transcoder as T
    dev, layer = 'cuda', args.dict_layer
    out = OUT / args.tag
    model, tok, _ = load(args.model, dev)
    eng = Engine(model, dev)
    tc = T.TopKTranscoder.from_state(np.load(args.transcoder), device=dev)
    box = {}
    eng.mlps[layer].register_forward_pre_hook(lambda m, i: box.__setitem__('x', i[0].detach()))
    Wd = eng.down[layer].weight.float()                       # [d, F]
    Dec = tc.W_dec * tc.y_scale                               # [m, d]

    def states(ids):
        cap = eng.capture(ids, neurons=True)
        return cap['m'][layer].float(), cap['u'][layer].float(), tc.encode(box['x'].float())

    disc = materials(tok, 'discovery')
    held = materials(tok, args.split)
    rep = {'layer': layer, 'split': args.split, 'transcoder': str(args.transcoder), 'tasks': {}}
    for task in B.TASKS:
        items = disc[task]['items']
        sf = torch.zeros(tc.m, device=dev, dtype=torch.float64)
        sn = torch.zeros(eng.F, device=dev, dtype=torch.float64)
        for idx in batches(items, args.batch):
            clean, corrupt, a, b = tensors(items, idx, dev)
            _, u_f, a_f = states(corrupt)
            _, u_c, a_c = states(clean)
            eng.grab = 'grad'
            eng.reset_store()
            x = eng.embed(clean).detach().clone().requires_grad_(True)
            eng._diff(eng.forward_logits(embeds=x), a, b).sum().backward()
            g = eng.m[layer].grad.float()
            eng.grab = None
            eng.reset_store()
            sf += ((a_f - a_c) * (g @ Dec.T)).sum(dim=(0, 1)).double()
            sn += ((u_f - u_c) * (g @ Wd)).sum(dim=(0, 1)).double()
        order = {'feature': rank(sf.abs().cpu().numpy()), 'neuron': rank(sn.abs().cpu().numpy())}
        items = held[task]['items']
        raw = {k: np.zeros(len(items)) for k in ('clean', 'corrupt', 'empty')}
        for kind in order:
            for k in B.DICT_K:
                raw[f'{kind}|{k}'] = np.zeros(len(items))
            raw[f'{kind}|all'] = np.zeros(len(items))
        for idx in batches(items, args.batch):
            clean, corrupt, a, b = tensors(items, idx, dev)
            y_f, u_f, a_f = states(corrupt)
            _, u_c, a_c = states(clean)
            raw['clean'][idx] = eng.ld(clean, a, b).cpu().numpy()
            raw['corrupt'][idx] = eng.ld(corrupt, a, b).cpu().numpy()
            on = torch.tensor(True, device=dev)

            def run(y):
                eng.rm[layer] = (on, y)
                v = eng.ld(clean, a, b).cpu().numpy()
                eng.clear()
                return v
            raw['empty'][idx] = run(y_f)
            for kind, delta, W in (('feature', a_c - a_f, Dec), ('neuron', u_c - u_f, Wd.T)):
                for k in B.DICT_K + ['all']:
                    sel = order[kind] if k == 'all' else order[kind][:k]
                    sel = torch.as_tensor(sel.copy(), device=dev)
                    raw[f'{kind}|{k}'][idx] = run(y_f + delta[..., sel] @ W[sel])
        rep['tasks'][task] = B.dict_compare(raw, B.DICT_K)
        print(task, json.dumps(rep['tasks'][task], ensure_ascii=False), flush=True)
    (out / f'dict_{args.split}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')


# ---------------- report ----------------

def report(args):
    sys.path.insert(0, str(ROOT / 'tools'))
    from power_analysis import check_phase
    from provenance import provenance
    out = OUT / args.tag
    disc = json.loads((out / 'discovery.json').read_text(encoding='utf-8'))
    ev = json.loads((out / f'eval_{args.split}.json').read_text(encoding='utf-8'))
    z = np.load(out / 'rankings.npz')
    rep = {'schema': B.SCHEMA, 'criteria': B.CRITERIA, 'split': args.split, 'model': disc['model'], 'snapshot': disc['snapshot'], 'dtype': disc['dtype'],
           'design_sha': disc['design_sha'], 'rankings_sha256': disc['rankings_sha256'], 'discovery': disc['tasks'], 'tasks': {},
           'natural_kl': ev.get('natural_kl'), 'provenance': provenance(model_path=disc['model'], seed=0, tool='mech_bench_run'),
           'official_mib': False}
    raws = {}
    for task in B.TASKS:
        f = out / f'raw_{args.split}_{task}.npz'
        if f.exists():
            d = np.load(f, allow_pickle=False)
            raws[task] = {k: d[k].astype(np.float64) for k in d.files if k != 'clusters'}, d['clusters'].tolist()
    for task, (raw, clusters) in raws.items():
        meta = ev['tasks'][task]
        score = B.score_task(raw, clusters)
        unit = B.score_task(raw, clusters, methods=['act_diff', 'atp', 'atp_ig'], track='U', budgets=B.UNIT_BUDGETS)
        verdict = B.judge(score, meta['gate'])
        spec = {}
        full, empty = raw['clean'], raw['empty|N|cf']
        b = B.CRITERIA['specific_budget']
        for m in B.METHODS:
            key = B.make_key('N', 'remove', 'cf', m, b)
            target = 1 - B.faithfulness(raw[key], full, empty) if key in raw else None
            offs = []
            for other, (oraw, _) in raws.items():
                if other == task:
                    continue
                k2 = B.make_key('N', 'remove', 'cf', m, b, src=task)
                offs.append(1 - B.faithfulness(oraw[k2], oraw['clean'], oraw['empty|N|cf']) if k2 in oraw else None)
            spec[m] = B.specificity(target, offs if len(offs) == len(B.TASKS) - 1 else [None])
        rep['tasks'][task] = {**meta, 'power': check_phase(B.CRITERIA['expected_delta'], meta['n'], B.CRITERIA['p_d']), 'node': score, 'unit': unit,
                              'verdict': verdict, 'specificity': spec}
    if disc['arch'] == 'gpt2' and 'rs_ioi|node|atp' in z.files:
        L, H = disc['L'], disc['H']
        rep['known_ioi_recall'] = {m: B.known_head_recall(z[f'rs_ioi|node|{m}'][:L * H].reshape(L, H)) for m in B.METHODS}
    (out / f'report_{args.split}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    for task, t in rep['tasks'].items():
        print(task, 'identity', t['node'].get('identity'), 'random', round(t['node'].get('random_auc', float('nan')), 3), 'matched', round(t['node'].get('matched_auc', float('nan')), 3))
        for m, row in t['node']['methods'].items():
            print('  ', m, round(row.get('auc_f', float('nan')), 3), row.get('budget_f80'), t['verdict']['methods'].get(m), t['specificity'].get(m, {}).get('status'))
        print('   unit', {m: round(r.get('auc_f', float('nan')), 3) for m, r in t['unit']['methods'].items()}, 'random', t['unit'].get('random_auc'))
    if 'known_ioi_recall' in rep:
        print({m: v['hit'] for m, v in rep['known_ioi_recall'].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['discover', 'eval', 'dict', 'report'])
    ap.add_argument('--model', default='')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='confirm', choices=['confirm', 'lockbox'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--include-failed', action='store_true')
    ap.add_argument('--dict-layer', type=int, default=30)
    ap.add_argument('--transcoder', default='D:/ai2050_native/_transcoder/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-L30/tc-m16384-k64-s0/transcoder.npz')
    ap.add_argument('--allow-code-change', action='store_true')
    args = ap.parse_args()
    {'discover': discover, 'eval': evaluate, 'dict': dict_step, 'report': report}[args.step](args)


if __name__ == '__main__':
    main()
