"""Phase 2850 (LPF MA2 cont): content-channel consumer mapping.

From 2849: the FORMER L13 h30 delivers ~82% of its causal damage via
the residual CONTENT channel (its 926x orthogonal write), not via
amplifier gate modulation.  Question: WHO consumes that write?

Arms (80 words, same/func/null matched protocol):
  A  gate-reshaping map: for every downstream head (L14..L35, 32h),
     dA11 = A11(clamp L13h30) - A11(base same).  Gate readers =
     heads whose self-read mass is systematically reshaped.
  B  content-forwarding map: per-head write delta
     dW(l,h) = write_clamp(l,h) - write_base(l,h), aligned with the
     FORMER's original write direction w_out (4096-d, unit).
     Forwarders = heads whose output change carries w_out.
  C  mediation test: per word, restore (bias_to base A11) the top-3
     gate readers (by |dA11|) ON TOP of the L13h30 clamp; if the
     own-direction drop shrinks, readers mediate the damage.

Prereg (frozen before any readout):
  J1  gate_readers_identified iff >= 3 downstream heads have
      frac_words(dA11 < -0.02) >= 0.6 AND mean |dA11| >= 0.02
  J2  content_forwarders_identified iff >= 3 downstream heads have
      mean align(dW, w_out) >= 0.3 AND frac_words(align > 0.2) >= 0.5
  J3  reader_mediation iff mean drop(top3-restored) <= 0.8 * mean
      drop(L13h30 only)
  verdict: consumers_confirmed iff J1 AND (J2 OR J3)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2850' / 'content_consumers'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2850
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32
DL = list(range(14, 36))          # downstream layers
ND = len(DL)

PREREG = {
    'J1': 'gate_readers_identified iff >=3 downstream heads '
          'frac_words(dA11<-0.02)>=0.6 AND mean|dA11|>=0.02',
    'J2': 'content_forwarders_identified iff >=3 downstream heads '
          'mean align(dW,w_out)>=0.3 AND frac_words(align>0.2)>=0.5',
    'J3': 'reader_mediation iff mean drop(top3-restored) <= '
          '0.8 * mean drop(L13h30 only)',
    'verdict': 'consumers_confirmed iff J1 AND (J2 OR J3)',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'L13h30 content consumers: gate map + '
                               'forwarding map + top3 restore mediation, '
                               '80 words'}
        fc.save(execution_path, execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    hd = int(model.config.head_dim)
    nh = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    group = nh // n_kv

    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    cap = {'sain': {}, 'attn': {}, 'mlp': {}}

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

    def clear_cap():
        for d in ('sain', 'attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_run(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True, output_attentions=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(NL)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(NL)])
        aw = {li: out.attentions[li][0].float().cpu().numpy()
              for li in range(NL)}
        sain = {li: cap['sain'][li][0] for li in range(NL)}
        return hs, attn, mlp, aw, sain

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {}
    for cat in CAT_WORDS:
        targets[cat] = [w for w in CATS[cat]
                        if w in single_tok][:MAX_WORDS]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]
    n_words = len(target_list)

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    rng = np.random.default_rng(SEED)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = tid('the')

    OV = {}
    for li in range(NL):
        Wo = model.model.layers[li].self_attn.o_proj.weight \
            .detach().float().cpu().numpy().astype(np.float64)
        Wo3 = Wo.reshape(Wo.shape[0], nh, hd)
        for h in range(nh):
            OV[(li, h)] = Wo3[:, h, :]
    vproj = {li: model.model.layers[li].self_attn.v_proj
             for li in range(NL)}

    def make_patched(sa):
        orig = sa.forward
        holder = {'map': {}}
        scaling = sa.scaling
        import transformers.models.qwen3.modeling_qwen3 as q3

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
                                device=aw.device, dtype=aw.dtype).triu(1)
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

    def set_maps(spec):
        for li in patches:
            patches[li]['map'] = {}
        for (li, h, b) in spec:
            patches[li]['map'][h] = [(1, 1, b)]

    def bias_to(aw2, li, h, target):
        A11 = float(aw2['same'][li][h][1, 1])
        t = min(max(target, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0))

    def bias_for(aw2, li, h):
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        return bias_to(aw2, li, h, t)

    # accumulators
    dA11_map = np.zeros((n_words, ND, NH))       # arm A
    align_wout = np.zeros((n_words, ND, NH))     # arm B
    align_cdir = np.zeros((n_words, ND, NH))
    dwnorm = np.zeros((n_words, ND, NH))
    drop1330 = np.zeros(n_words)
    drop_restore = np.zeros(n_words)
    reader_ids = np.zeros((n_words, 3, 2), dtype=np.int64)
    cdirchg = np.zeros((n_words, NL))
    resid_all = []

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        cdir = dW_unit[ci]
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        c2 = {'same': [tid(same_cat[0]), w_tid],
              'func': [func_tid, w_tid],
              'null': [null_tids[i], w_tid]}

        hs_iso, attn_iso, mlp_iso, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[LAST] + attn_iso[LAST] + mlp_iso[LAST]
        raw2, aw2, sain2 = {}, {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, sac = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
            sain2[cn] = sac
        d_spec_full = raw2['same'] \
            - 0.5 * (raw2['func'] + raw2['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        base_own = abs(float(d_spec_full @ cdir)) / nfull

        # FORMER write direction (same cond, pos1)
        a11_f = float(aw2['same'][13][30][1, 1])
        vin = sain2['same'][13][1:2]
        vdt = next(vproj[13].parameters()).dtype
        v13 = vproj[13](torch.tensor(vin, device='cuda', dtype=vdt)
                        ).detach().float().cpu().numpy().astype(
                            np.float64)
        Vh = v13.reshape(1, n_kv, hd)[0, 13 // group]
        w_out = unit(a11_f * (Vh @ OV[(13, 30)].T))

        # clamp L13 h30
        b13 = bias_for(aw2, 13, 30)
        set_maps([(13, 30, b13)])
        hs_c, attn_c, mlp_c, awc_c, sain_c = forward_run(c2['same'], 1)
        set_maps([])
        resid_all.append(abs(float(awc_c[13][30][1, 1])
                             - 0.5 * (float(aw2['func'][13][30][1, 1])
                                      + float(aw2['null'][13][30][1, 1]))))
        dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
            - 0.5 * (raw2['func'] + raw2['null'])
        cls_c = abs(float(dsc @ cdir)) / nfull
        drop1330[i] = (base_own - cls_c) / max(base_own, 1e-30)

        for l in range(NL):
            cdirchg[i, l] = abs(float(
                (sain_c[l][1] - sain2['same'][l][1]) @ cdir))

        # per-head write deltas + gate deltas (downstream layers)
        vcache = {}
        for q, l in enumerate(DL):
            vin_b = sain2['same'][l][1:2]
            vin_c = sain_c[l][1:2]
            vdt = next(vproj[l].parameters()).dtype
            vb = vproj[l](torch.tensor(vin_b, device='cuda', dtype=vdt)
                          ).detach().float().cpu().numpy() \
                .astype(np.float64)
            vc = vproj[l](torch.tensor(vin_c, device='cuda', dtype=vdt)
                          ).detach().float().cpu().numpy() \
                .astype(np.float64)
            vcache[l] = (vb, vc)
            for h in range(NH):
                kv = h // group
                Vb = vb.reshape(1, n_kv, hd)[0, kv]
                Vc = vc.reshape(1, n_kv, hd)[0, kv]
                a_b = float(aw2['same'][l][h][1, 1])
                a_c = float(awc_c[l][h][1, 1])
                dA11_map[i, q, h] = a_c - a_b
                wB = a_b * (Vb @ OV[(l, h)].T)
                wC = a_c * (Vc @ OV[(l, h)].T)
                dww = wC - wB
                nrm = float(np.linalg.norm(dww))
                dwnorm[i, q, h] = nrm
                if nrm > 1e-12:
                    align_wout[i, q, h] = float(dww @ w_out) / nrm
                    align_cdir[i, q, h] = float(dww @ cdir) / nrm

        # mediation: top-3 gate readers by |dA11|, restore to base
        flat = np.abs(dA11_map[i]).reshape(-1)
        top3 = np.argsort(-flat)[:3]
        spec = []
        for k, t in enumerate(top3):
            q, h = int(t) // NH, int(t) % NH
            l = DL[q]
            reader_ids[i, k] = (l, h)
            spec.append((l, h, bias_to({'same': awc_c}, l, h,
                                       float(aw2['same'][l][h][1, 1]))))
        # deduplicate same-layer heads: patch map supports per-layer
        # multiple heads already (map[h] list); set_maps handles it.
        set_maps([(13, 30, b13)] + spec)
        hs_r, attn_r, mlp_r, _, _ = forward_run(c2['same'], 1)
        set_maps([])
        dsc = (hs_r[LAST] + attn_r[LAST] + mlp_r[LAST]) - iso0 \
            - 0.5 * (raw2['func'] + raw2['null'])
        cls_r = abs(float(dsc @ cdir)) / nfull
        drop_restore[i] = (base_own - cls_r) / max(base_own, 1e-30)

        if (i + 1) % 10 == 0:
            print('P2850 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    # J1: gate readers
    frac_down = (dA11_map < -0.02).mean(0)          # (ND, NH)
    mean_abs = np.abs(dA11_map).mean(0)
    reader_mask = (frac_down >= 0.6) & (mean_abs >= 0.02)
    n_readers = int(reader_mask.sum())

    # J2: content forwarders
    aw_align = align_wout.mean(0)
    frac_al = (align_wout > 0.2).mean(0)
    fwd_mask = (aw_align >= 0.3) & (frac_al >= 0.5)
    n_fwd = int(fwd_mask.sum())

    # J3: mediation
    med_ratio = float(np.mean(drop_restore)) / max(
        float(np.mean(drop1330)), 1e-30)
    j3 = bool(med_ratio <= 0.8)

    top_readers = [(int(l), int(h)) for l, h in reader_ids.reshape(-1, 2)]
    seen = []
    for x in top_readers:
        if x not in seen:
            seen.append(x)

    v = {
        'n_words': n_words,
        'J1_gate_readers': bool(n_readers >= 3),
        'n_gate_readers': n_readers,
        'top_reader_heads': seen[:8],
        'J2_content_forwarders': bool(n_fwd >= 3),
        'n_content_forwarders': n_fwd,
        'top_forwarder_heads': [
            {'layer': int(DL[q]), 'head': int(h),
             'align': round(float(aw_align[q, h]), 3)}
            for q, h in zip(*np.where(fwd_mask))][:8],
        'J3_reader_mediation': j3,
        'med_ratio': round(med_ratio, 4),
        'mean_drop_L13h30': round(float(np.mean(drop1330)), 5),
        'mean_drop_restored': round(float(np.mean(drop_restore)), 5),
        'mean_abs_dA11_amplifiers': {
            'L22h28': round(float(np.abs(dA11_map[:, 8, 28]).mean()), 4),
            'L23h29': round(float(np.abs(dA11_map[:, 9, 29]).mean()), 4),
            'L26h4': round(float(np.abs(dA11_map[:, 12, 4]).mean()), 4),
            'L34h15': round(float(np.abs(dA11_map[:, 20, 15]).mean()), 4),
        },
        'cdirchg_peak_layer': int(np.argmax(cdirchg.mean(0))),
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': 'consumers_confirmed'
        if (n_readers >= 3 and (n_fwd >= 3 or j3))
        else 'partial' if (n_readers >= 3 or n_fwd >= 3 or j3)
        else 'not_confirmed',
    }

    result = {'phase': 2850, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'consumers.npz',
           dA11_map=dA11_map.astype(np.float32),
           align_wout=align_wout.astype(np.float32),
           align_cdir=align_cdir.astype(np.float32),
           dwnorm=dwnorm.astype(np.float32),
           drop1330=drop1330.astype(np.float32),
           drop_restore=drop_restore.astype(np.float32),
           cdirchg=cdirchg.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2850', elapsed)
    print('P2850 VERDICT %s' % json.dumps(v), flush=True)
    print('P2850 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
