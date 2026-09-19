"""Reusable dual-spectrum head-census pipeline (MA / Atlas line).

Faithful extraction of the phase2846 MA2 protocol into a parameterized
library so that Atlas-E (extended-vocab) runs reuse identical machinery.

Causal spectrum (per layer li, head h, word i):
    d_spec_full = raw2[same] - 0.5*(raw2[func] + raw2[null])
    cls_base    = |d_spec_full . cdir| / |d_spec_full|
    clamp head (li,h) at logits (1,1) toward t = 0.5*(A11_func + A11_null)
    cls_c       = same metric under clamp
    drop        = (cls_base - cls_c) / max(cls_base, 1e-30)
Alignment spectrum (per layer, head):
    s0/s1 = spec-contrast of the per-head OV write at pos0 (cond) /
    pos1 (self), projected on cdir; V = v_proj(sain), weights aw[1, :2].

build_vocab reproduces the phase2846 construction order exactly:
tid cache -> single-token filter -> top-N cap -> dW_unit over the FULL
single-token set -> seeded null tids -> func tid('the') LAST.
"""
import numpy as np
import torch


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def single_token_id(tok, t, cache):
    if t not in cache:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, '%s -> %s' % (t, ids)
        cache[t] = int(ids[0])
    return cache[t]


def build_vocab(tok, W_U, cats, cat_words, seed, max_words):
    tc = {}
    all_words = [w for v in cats.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            single_token_id(tok, w, tc)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {}
    for cat in cat_words:
        targets[cat] = [w for w in cats[cat]
                        if w in single_tok][:max_words]
    target_list = [(cat, w) for cat in cat_words for w in targets[cat]]
    n_words = len(target_list)

    Erows = {w: W_U[single_token_id(tok, w, tc)].astype(np.float64)
             for w in single_tok}
    cents = []
    for cat in cat_words:
        ws = [w for w in cats[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / (len(cat_words) - 1.0)
    dW_unit = np.stack([unit(dW[i]) for i in range(len(cat_words))])

    rng = np.random.default_rng(seed)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = single_token_id(tok, 'the', tc)
    return {'targets': targets, 'target_list': target_list,
            'dW_unit': dW_unit, 'Cm': Cm, 'null_tids': null_tids,
            'func_tid': func_tid, 'single_tok': single_tok,
            'tid_map': dict(tc), 'n_words': n_words}


class AtlasCensus:
    """Head census (causal drop + OV alignment), phase2846 protocol."""

    def __init__(self, model, vocab, cats, cat_words, last=35):
        self.model = model
        cfg = model.config
        self.hd = int(cfg.head_dim)
        self.nh = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.group = self.nh // self.n_kv
        self.nl = int(cfg.num_hidden_layers)
        self.last = last
        self.cats = cats
        self.cat_words = cat_words
        self.target_list = vocab['target_list']
        self.targets = vocab['targets']
        self.single_tok = vocab['single_tok']
        self.tmap = vocab['tid_map']
        self.dW_unit = vocab['dW_unit']
        self.null_tids = vocab['null_tids']
        self.func_tid = vocab['func_tid']
        self._cap = {'sain': {}, 'attn': {}, 'mlp': {}}
        self._install_hooks()
        self._install_patches()
        self._cache_ov()

    # ---------- tid ----------
    def tid(self, t):
        return self.tmap[t]

    # ---------- hooks / patched attention ----------
    def _install_hooks(self):
        cap = self._cap
        model = self.model

        def make_out_hook(kind, li):
            def hook(module, args, output):
                o = output[0] if isinstance(output, tuple) else output
                cap[kind].setdefault(li, []).append(
                    o[0].detach().float().cpu().numpy())
            return hook

        def make_in_hook(li):
            def pre_hook(module, args, kwargs):
                x = args[0] if args else kwargs['hidden_states']
                cap['sain'].setdefault(li, []).append(
                    x.detach()[0].float().cpu().numpy())
            return pre_hook

        handles = []
        for li, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_hook(
                make_out_hook('attn', li)))
            handles.append(layer.mlp.register_forward_hook(
                make_out_hook('mlp', li)))
            handles.append(layer.self_attn.register_forward_pre_hook(
                make_in_hook(li), with_kwargs=True))
        self.handles = handles

    def _install_patches(self):
        hd = self.hd
        model = self.model
        import transformers.models.qwen3.modeling_qwen3 as q3

        def make_patched(sa):
            orig = sa.forward
            holder = {'map': {}}
            scaling = sa.scaling

            def forward(hidden_states, position_embeddings,
                        attention_mask=None, past_key_values=None, **kw):
                if not holder['map']:
                    return orig(hidden_states, position_embeddings,
                                attention_mask, past_key_values, **kw)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, hd)
                q = sa.q_norm(sa.q_proj(hidden_states)
                              .view(hidden_shape)).transpose(1, 2)
                k = sa.k_norm(sa.k_proj(hidden_states)
                              .view(hidden_shape)).transpose(1, 2)
                v = sa.v_proj(hidden_states).view(hidden_shape) \
                    .transpose(1, 2)
                cos, sin = position_embeddings
                q, k = q3.apply_rotary_pos_emb(q, k, cos, sin)
                k = q3.repeat_kv(k, sa.num_key_value_groups)
                v = q3.repeat_kv(v, sa.num_key_value_groups)
                aw = torch.matmul(q, k.transpose(2, 3)) * scaling
                L = aw.shape[-1]
                causal = torch.full((L, L), torch.finfo(aw.dtype).min,
                                    device=aw.device,
                                    dtype=aw.dtype).triu(1)
                aw = aw + causal
                for h, blist in holder['map'].items():
                    for (qi, ki, b) in blist:
                        aw[0, h, qi, ki] = aw[0, h, qi, ki] + b
                aw = torch.softmax(aw, dim=-1)
                out = torch.matmul(aw, v).transpose(1, 2) \
                    .reshape(*input_shape, -1)
                out = sa.o_proj(out)
                return out, aw
            return forward, holder

        patches = {}
        for li, layer in enumerate(model.model.layers):
            fwd, holder = make_patched(layer.self_attn)
            layer.self_attn.forward = fwd
            patches[li] = holder
        self.patches = patches

    def _cache_ov(self):
        hd, nh = self.hd, self.nh
        OV = {}
        for li in range(self.nl):
            Wo = self.model.model.layers[li].self_attn.o_proj.weight \
                .detach().float().cpu().numpy().astype(np.float64)
            Wo3 = Wo.reshape(Wo.shape[0], nh, hd)
            for h in range(nh):
                OV[(li, h)] = Wo3[:, h, :]
        self.OV = OV
        self.vproj = {li: self.model.model.layers[li].self_attn.v_proj
                      for li in range(self.nl)}

    def clear_cap(self):
        for d in ('sain', 'attn', 'mlp'):
            for li in self._cap[d]:
                del self._cap[d][li][:]

    def forward_run(self, tokens, pos):
        self.clear_cap()
        with torch.no_grad():
            out = self.model(
                torch.tensor([tokens], device='cuda'),
                output_hidden_states=True, output_attentions=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([self._cap['attn'][li][0][pos]
                         for li in range(self.nl)])
        mlp = np.stack([self._cap['mlp'][li][0][pos]
                        for li in range(self.nl)])
        aw = {li: out.attentions[li][0].float().cpu().numpy()
              for li in range(self.nl)}
        sain = {li: self._cap['sain'][li][0] for li in range(self.nl)}
        return hs, attn, mlp, aw, sain

    # ---------- conditions ----------
    def conds2_for(self, i, cat, w):
        w_tid = self.tid(w)
        same_cat = [x for x in self.targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in self.cats[cat] if x != w
                        and x in self.single_tok][:1]
        return {'same': [self.tid(same_cat[0]), w_tid],
                'func': [self.func_tid, w_tid],
                'null': [self.null_tids[i], w_tid]}

    def bias_for(self, aw2, li, h):
        A11 = float(aw2['same'][li][h][1, 1])
        t_raw = 0.5 * (float(aw2['func'][li][h][1, 1])
                       + float(aw2['null'][li][h][1, 1]))
        t = min(max(t_raw, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0)), t_raw

    # ---------- core measurement ----------
    def measure_layer(self, li, word_rows=None, tag=''):
        """Causal drop + OV alignment for all heads at layer li.

        word_rows: indices into target_list (default all).
        Returns drops (n,nh), s0, s1, clamp_resid list.
        """
        n_kv, hd, group = self.n_kv, self.hd, self.group
        LAST = self.last
        if word_rows is None:
            word_rows = list(range(len(self.target_list)))
        dL = np.zeros((len(word_rows), self.nh))
        s0L = np.zeros((len(word_rows), self.nh))
        s1L = np.zeros((len(word_rows), self.nh))
        clamp_resid = []
        vdt = next(self.vproj[li].parameters()).dtype
        for j, i in enumerate(word_rows):
            cat, w = self.target_list[i]
            cdir = self.dW_unit[self.cat_words.index(cat)]
            w_tid = self.tid(w)
            c2 = self.conds2_for(i, cat, w)
            hs_iso, attn_iso, mlp_iso, _, _ = self.forward_run(
                [w_tid], 0)
            iso0 = hs_iso[LAST] + attn_iso[LAST] + mlp_iso[LAST]
            raw2, aw2, sain2 = {}, {}, {}
            for cn, toks in c2.items():
                hs, attn, mlp, awc, sac = self.forward_run(toks, 1)
                raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
                aw2[cn] = awc
                sain2[cn] = sac
            d_spec_full = raw2['same'] \
                - 0.5 * (raw2['func'] + raw2['null'])
            nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
            cls_base = float(abs(d_spec_full @ cdir)) / nfull

            Vc = {}
            for cn in c2:
                vin = sain2[cn][li][:2]
                Vc[cn] = self.vproj[li](torch.tensor(
                    vin, device='cuda', dtype=vdt)
                ).detach().float().cpu().numpy()
            for h in range(self.nh):
                kv = h // group
                s_dir = {}
                for cn in c2:
                    Vh = Vc[cn].reshape(2, n_kv, hd)[:, kv, :]
                    s_dir[cn] = aw2[cn][li][h][1, :2][:, None] \
                        * (Vh @ self.OV[(li, h)].T)
                ssp = s_dir['same'] \
                    - 0.5 * (s_dir['func'] + s_dir['null'])
                s0L[j, h] = float(ssp[0] @ cdir)
                s1L[j, h] = float(ssp[1] @ cdir)

            for h in range(self.nh):
                b, t = self.bias_for(aw2, li, h)
                self.patches[li]['map'] = {h: [(1, 1, b)]}
                hs, attn, mlp, awc, _ = self.forward_run(c2['same'], 1)
                self.patches[li]['map'] = {}
                clamp_resid.append(
                    abs(float(awc[li][h][1, 1]) - t))
                dsc = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0 \
                    - 0.5 * (raw2['func'] + raw2['null'])
                cls_c = float(abs(dsc @ cdir)) / nfull
                dL[j, h] = (cls_base - cls_c) / max(cls_base, 1e-30)
            if (j + 1) % 10 == 0:
                print('ATLAS L%d%s words [%d/%d]'
                      % (li, tag, j + 1, len(word_rows)), flush=True)
        return dL, s0L, s1L, clamp_resid
