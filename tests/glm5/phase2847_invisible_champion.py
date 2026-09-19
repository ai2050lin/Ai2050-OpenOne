"""Phase 2847 (LPF MA2 cont / ATLAS_PLAN frontline 2): invisible-champion
anatomy + dual causal roles.

Motivated by 2846 census inversion:
  early cluster  E = [(13,30), (2,31), (5,25), (5,26), (4,4)]
      tiny direct cdir write (|share| ~ 0.003) but large clamp drop
      (10.1% / 4.9% / 4.8% / 2.9% / 2.3%) -> suspected UPSTREAM
      FORMERS (shape later computation rather than write the answer)
  late cluster   G = [(22,28), (23,29), (26,4), (34,15)]
      large direct write (s1 up to 0.47) but modest drop (3-4%)
      -> suspected READOUT AMPLIFIERS (redundant)

Arms:
  A  L13 h30 anatomy: post-clamp downstream displacement profile
     (which layers carry the causal effect), attention gate profile
     (A[1,:2] same/func/null), total-write magnitude vs cdir write,
     10-direction drop selectivity.
  B  joint clamp ladders: early 5-head rungs by load desc, late
     4-head rungs; additivity ratio joint / sum(singles).
  C  clamp-bias saturation audit (|b| near +-20 clip; descriptive).
  C' 2846 residual-statistics bug: per-layer print used only the
     last word's residuals (clamp_resid[-NH:]) -> confirmed code
     read; global max 0.5708 is a true per-entry max.

Prereg (frozen before any readout):
  F1  upstream_router_profile iff for L13 h30: |mean direct write
      share| < 0.01 AND mean drop >= 0.05 AND post-clamp mean
      downstream displacement over L14..L35 > displacement at L13
      AND the peak-cdir-change layer index > 13
  F2  dual_causal_roles iff early cluster mean |share| < 0.01 with
      mean drop >= 0.03 AND late cluster mean share >= 0.1 with
      mean drop < 0.05
  F3  frontier_joint_additive iff early joint5 / sum(singles) in
      [0.8, 1.2]
  verdict: dual_roles_confirmed iff F1 AND F2
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
OUT = BASE / 'phase2847' / 'invisible_champion'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2847
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32

EARLY = [(13, 30), (2, 31), (5, 25), (5, 26), (4, 4)]
LATE = [(22, 28), (23, 29), (26, 4), (34, 15)]

PREREG = {
    'F1': 'upstream_router_profile iff L13h30 |share|<0.01 AND '
          'drop>=0.05 AND mean disp L14..35 > disp L13 AND peak '
          'cdir-change layer > 13',
    'F2': 'dual_causal_roles iff early mean|share|<0.01 & drop>=0.03 '
          'AND late mean share>=0.1 & drop<0.05',
    'F3': 'frontier_joint_additive iff early joint5/sum(singles) in '
          '[0.8, 1.2]',
    'C': 'saturation audit descriptive: frac champion clamp entries '
         'with |b| >= 19.9',
    'verdict': 'dual_roles_confirmed iff F1 AND F2',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


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
                     'design': 'L13h30 anatomy + dual-cluster joint '
                               'ladders + saturation audit, 80 words'}
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

    def conds2_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

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

    def bias_for(aw2, li, h):
        A11 = float(aw2['same'][li][h][1, 1])
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        t = min(max(t, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0))

    # ---------- accumulators ----------
    # Arm A: per-word profiles for L13 h30
    disp_words = []          # (80, 36) relative displacement per layer
    cdirchg_words = []       # (80, 36) |cdir proj change| per layer
    gate_rows = []           # (80, 6) A[1,:2] same/func/null
    writemag = []            # (80, 2) total write mag / cdir write
    drop10_words = []        # (80, 10) per-direction drop
    drop1330_words = []      # (80,) headline drop
    # Arm B: ladders
    early_single = np.zeros((n_words, len(EARLY)))
    late_single = np.zeros((n_words, len(LATE)))
    early_ladder = np.zeros((n_words, len(EARLY)))
    late_ladder = np.zeros((n_words, len(LATE)))
    # Arm C
    b_abs_all = []           # |b| of every clamp entry this run
    sat_count = 0
    n_clamp_entries = 0
    resid_all = []

    ALL_HEADS = EARLY + LATE

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        cdir = dW_unit[ci]
        w_tid = tid(w)
        c2 = conds2_for(i, cat, w)

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
        cls_base_d = np.array(
            [abs(float(d_spec_full @ dW_unit[d])) / nfull
             for d in range(10)])

        sain_same_ref = {l: sain2['same'][l][1].copy()
                         for l in range(NL)}

        # direct write of L13 h30 (verify census) + total write mag
        li13, h13 = 13, 30
        kv13 = h13 // group
        vdt = next(vproj[li13].parameters()).dtype
        Vc13 = {}
        for cn in c2:
            vin = sain2[cn][li13][:2]
            Vc13[cn] = vproj[li13](torch.tensor(
                vin, device='cuda', dtype=vdt)
            ).detach().float().cpu().numpy()
        sd = {}
        for cn in c2:
            Vh = Vc13[cn].reshape(2, n_kv, hd)[:, kv13, :]
            sd[cn] = aw2[cn][li13][h13][1, :2][:, None] \
                * (Vh @ OV[(li13, h13)].T)
        ssp13 = sd['same'] - 0.5 * (sd['func'] + sd['null'])
        cdir_write = float(ssp13[1] @ cdir) + float(ssp13[0] @ cdir)
        w_tot = float(np.linalg.norm(ssp13[0]) + np.linalg.norm(ssp13[1]))
        writemag.append([w_tot, cdir_write])

        gate = []
        for cn in ('same', 'func', 'null'):
            a = aw2[cn][li13][h13][1, :2]
            gate.extend([float(a[0]), float(a[1])])
        gate_rows.append(gate)

        # ---- clamp runs ----
        def clamp_run(spec):
            set_maps(spec)
            hs, attn, mlp, awc, sac = forward_run(c2['same'], 1)
            set_maps([])
            return hs, attn, mlp, awc, sac

        # L13 h30 alone (arm A)
        b13 = bias_for(aw2, li13, h13)
        n_clamp_entries += 1
        b_abs_all.append(abs(b13))
        if abs(b13) >= 19.9:
            sat_count += 1
        hs_c, attn_c, mlp_c, awc_c, sain_c = clamp_run(
            [(li13, h13, b13)])
        resid_all.append(abs(float(awc_c[li13][h13][1, 1])
                             - 0.5 * (float(aw2['func'][li13][h13][1, 1])
                                      + float(aw2['null'][li13][h13][1, 1]))))
        dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
            - 0.5 * (raw2['func'] + raw2['null'])
        cls_c_d = np.array(
            [abs(float(dsc @ dW_unit[d])) / nfull for d in range(10)])
        drop10_words.append((cls_base_d - cls_c_d)
                            / np.maximum(cls_base_d, 1e-30))
        drop1330_words.append(float(drop10_words[-1][ci]))

        # downstream displacement profile (relative, pos1 input states)
        disp = np.zeros(NL)
        cdirchg = np.zeros(NL)
        for l in range(NL):
            dv = sain_c[l][1] - sain_same_ref[l]
            disp[l] = float(np.linalg.norm(dv)
                            / max(np.linalg.norm(sain_same_ref[l]), 1e-30))
            cdirchg[l] = abs(float(dv @ cdir))
        disp_words.append(disp)
        cdirchg_words.append(cdirchg)

        # singles (arm B, both clusters)
        for j, (li, h) in enumerate(ALL_HEADS):
            b = bias_for(aw2, li, h)
            n_clamp_entries += 1
            b_abs_all.append(abs(b))
            if abs(b) >= 19.9:
                sat_count += 1
            hs_c, attn_c, mlp_c, awc_c, _ = clamp_run([(li, h, b)])
            resid_all.append(
                abs(float(awc_c[li][h][1, 1])
                    - 0.5 * (float(aw2['func'][li][h][1, 1])
                             + float(aw2['null'][li][h][1, 1]))))
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = abs(float(dsc @ cdir)) / nfull
            ddrop = (cls_base_d[ci] - cls_c) / max(cls_base_d[ci], 1e-30)
            if j < len(EARLY):
                early_single[i, j] = ddrop
            else:
                late_single[i, j - len(EARLY)] = ddrop

        # early ladder rungs
        spec_acc = []
        for j, (li, h) in enumerate(EARLY):
            b = bias_for(aw2, li, h)
            spec_acc.append((li, h, b))
            hs_c, attn_c, mlp_c, _, _ = clamp_run(spec_acc)
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = abs(float(dsc @ cdir)) / nfull
            early_ladder[i, j] = (cls_base_d[ci] - cls_c) \
                / max(cls_base_d[ci], 1e-30)
        # late ladder rungs
        spec_acc = []
        for j, (li, h) in enumerate(LATE):
            b = bias_for(aw2, li, h)
            spec_acc.append((li, h, b))
            hs_c, attn_c, mlp_c, _, _ = clamp_run(spec_acc)
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = abs(float(dsc @ cdir)) / nfull
            late_ladder[i, j] = (cls_base_d[ci] - cls_c) \
                / max(cls_base_d[ci], 1e-30)

        if (i + 1) % 10 == 0:
            print('P2847 words [%d/%d]' % (i + 1, n_words), flush=True)

    disp_w = np.stack(disp_words)
    cdirchg_w = np.stack(cdirchg_words)
    drop10_w = np.stack(drop10_words)
    gate_w = np.stack(gate_rows)
    wm = np.stack(writemag)

    # ---------- verdicts ----------
    mean_drop1330 = float(np.mean(drop1330_words))
    share1330 = float(np.mean(wm[:, 1]))
    disp_mean = disp_w.mean(0)
    cdirchg_mean = cdirchg_w.mean(0)
    peak_disp_layer = int(np.argmax(disp_mean))
    peak_cdirchg_layer = int(np.argmax(cdirchg_mean))
    f1 = bool(abs(share1330) < 0.01 and mean_drop1330 >= 0.05
              and float(np.mean(disp_mean[14:])) > float(disp_mean[13])
              and peak_cdirchg_layer > 13)

    # cluster F2 shares/drops come from the registered 2846 census
    # (immutable source):
    cz = np.load(BASE / 'phase2846' / 'fullhead_census'
                 / 'census_full.npz')
    ms0 = cz['mean_s0'].reshape(NL, NH)
    ms1 = cz['mean_s1'].reshape(NL, NH)
    mdr = cz['mean_drop'].reshape(NL, NH)
    early_shares = [float(abs(ms0[l, h] + ms1[l, h]))
                    for (l, h) in EARLY]
    early_drops = [float(mdr[l, h]) for (l, h) in EARLY]
    late_shares = [float(ms0[l, h] + ms1[l, h]) for (l, h) in LATE]
    late_drops = [float(mdr[l, h]) for (l, h) in LATE]
    f2 = bool(np.mean(early_shares) < 0.01
              and np.mean(early_drops) >= 0.03
              and np.mean(late_shares) >= 0.1
              and np.mean(late_drops) < 0.05)

    sum_singles = float(np.mean(early_single.sum(1)))
    joint5 = float(np.mean(early_ladder[:, -1]))
    ratio = joint5 / max(sum_singles, 1e-30)
    f3 = bool(0.8 <= ratio <= 1.2)
    sum_late = float(np.mean(late_single.sum(1)))
    joint4_late = float(np.mean(late_ladder[:, -1]))
    ratio_late = joint4_late / max(sum_late, 1e-30)

    v = {
        'n_words': n_words,
        'F1_upstream_router_profile': f1,
        'L13h30_mean_drop': round(mean_drop1330, 5),
        'L13h30_mean_cdir_write': round(share1330, 5),
        'L13h30_total_write_vs_cdir': round(
            float(np.mean(wm[:, 0])) / max(abs(share1330), 1e-30), 2),
        'L13h30_gate': {
            'A11_same': round(float(np.mean(gate_w[:, 1])), 4),
            'A11_ctrl': round(float(np.mean(
                0.5 * (gate_w[:, 3] + gate_w[:, 5]))), 4),
        },
        'peak_displacement_layer': peak_disp_layer,
        'peak_cdir_change_layer': peak_cdirchg_layer,
        'disp_L13': round(float(disp_mean[13]), 5),
        'disp_L14_35_mean': round(float(np.mean(disp_mean[14:])), 5),
        'F2_dual_causal_roles': f2,
        'early_shares': [round(x, 5) for x in early_shares],
        'early_drops': [round(x, 5) for x in early_drops],
        'late_shares': [round(x, 5) for x in late_shares],
        'late_drops': [round(x, 5) for x in late_drops],
        'F3_frontier_joint_additive': f3,
        'early_joint5': round(joint5, 5),
        'early_sum_singles': round(sum_singles, 5),
        'early_additivity_ratio': round(ratio, 4),
        'late_joint4': round(joint4_late, 5),
        'late_sum_singles': round(sum_late, 5),
        'late_additivity_ratio': round(ratio_late, 4),
        'C_sat_frac': round(sat_count / max(n_clamp_entries, 1), 5),
        'C_max_resid_this_run': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'C_max_b': round(float(np.max(b_abs_all)), 3),
        'L13h30_drop10_by_direction': [
            round(float(x), 4) for x in drop10_w.mean(0)],
        'final_verdict': 'dual_roles_confirmed' if (f1 and f2) else (
            'partial' if (f1 or f2) else 'not_confirmed'),
    }

    result = {'phase': 2847, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'anatomy.npz',
           disp_words=disp_w.astype(np.float32),
           cdirchg_words=cdirchg_w.astype(np.float32),
           drop10_words=drop10_w.astype(np.float32),
           gate_rows=gate_w.astype(np.float64),
           early_single=early_single.astype(np.float32),
           late_single=late_single.astype(np.float32),
           early_ladder=early_ladder.astype(np.float32),
           late_ladder=late_ladder.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2847', elapsed)
    print('P2847 VERDICT %s' % json.dumps(v), flush=True)
    print('P2847 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
