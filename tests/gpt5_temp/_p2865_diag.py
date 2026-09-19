"""2865 v1 failure diagnosis: locate the missing term in the
telescoping identity  h36_diff = sum_{l>=13} (Dattn_l + Dmlp_l).

Steps (single word, 2-token 'same' input):
  S0 determinism: two identical unpatched forwards -> all diffs ~ 0
  S1 tiny patch: L13H30 bias b=0.05 -> telescoping closure vector-norm
  S2 arm profile: per-layer ||Dattn||, ||Dmlp|| and cdir projections
  S3 alternative final point: ln2in[35] only (h35+attn35) vs +mlp35
Writes report to _p2865_diag_report.txt.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import rdc_construction_common as cc  # noqa: E402

ROOT = cc.ROOT
NL = 36
CLAMP_L, CLAMP_H = 13, 30
OUTP = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2865_diag_report.txt'
rep = []


def log(s):
    rep.append(s)
    print(s, flush=True)


def main():
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    hd = int(model.config.head_dim)

    cap = {'ln2in': {}, 'attn': {}, 'mlp': {}, 'hs': []}

    def out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o.detach()[0].float().cpu().numpy())
        return hook

    def in_hook(kind, li):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            cap[kind].setdefault(li, []).append(
                x.detach()[0].float().cpu().numpy())
        return pre_hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.post_attention_layernorm
                       .register_forward_pre_hook(
                           in_hook('ln2in', li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(
            out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            out_hook('mlp', li)))

    def clear():
        for d in ('ln2in', 'attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def run(tokens):
        clear()
        cap['hs'] = []
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_attentions=True,
                        output_hidden_states=True)
        ln2in = {li: cap['ln2in'][li][0][1].astype(np.float64)
                 for li in range(NL)}
        attn = np.stack([cap['attn'][li][0][1].astype(np.float64)
                         for li in range(NL)])
        mlp = np.stack([cap['mlp'][li][0][1].astype(np.float64)
                        for li in range(NL)])
        return ln2in, attn, mlp, out

    # word 'apple' + ' dog' arbitrary single tokens
    t_a = tok(' apple', add_special_tokens=False)['input_ids']
    t_b = tok(' dog', add_special_tokens=False)['input_ids']
    assert len(t_a) == 1 and len(t_b) == 1
    toks = [t_a[0], t_b[0]]

    # patched forward factory (2860 verbatim)
    layers = model.model.layers

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

    # S0 determinism
    A = run(toks)
    B = run(toks)
    dh36 = (A[3].hidden_states[36][0, 1].float().cpu().numpy()
            - B[3].hidden_states[36][0, 1].float().cpu().numpy())
    log('S0 hidden_states[36] diff norm (expect ~0): %.3e'
        % float(np.linalg.norm(dh36.astype(np.float64))))
    dsum0 = np.zeros(2560)
    for l in range(NL):
        dsum0 += (A[1][l] - B[1][l]) + (A[2][l] - B[2][l])
    log('S0 sum(all layers Dattn+Dmlp) norm (expect ~0): %.3e'
        % float(np.linalg.norm(dsum0)))

    # S1 tiny patch
    patches[CLAMP_L]['map'] = {CLAMP_H: [(1, 1, 0.05)]}
    C = run(toks)
    patches[CLAMP_L]['map'] = {}
    A11f = float(A[3].attentions[CLAMP_L][0].float().cpu().numpy()
                 [CLAMP_H][1, 1])
    A11c = float(C[3].attentions[CLAMP_L][0].float().cpu().numpy()
                 [CLAMP_H][1, 1])
    log('S1 A11 full=%.5f clamp=%.5f' % (A11f, A11c))

    h36_diff = (C[3].hidden_states[36][0, 1].float().cpu().numpy()
                - A[3].hidden_states[36][0, 1].float().cpu().numpy()
                ).astype(np.float64)
    log('NOTE hidden_states[36] is POST final_layernorm (see E L35); '
        'h36_diff here is norm-space, telescoping must NOT close to it.')
    # true residual readout: h36 = ln2in[35] + mlp[35] (hook combos)
    res36_diff = (C[0][35] + C[2][35]) - (A[0][35] + A[2][35])
    d_attn = C[1] - A[1]
    d_mlp = C[2] - A[2]
    v13 = d_attn[CLAMP_L]
    log('S1 ||v13||=%.4f  ||res36_diff(hook combo)||=%.4f'
        % (float(np.linalg.norm(v13)), float(np.linalg.norm(res36_diff))))

    tele = d_attn[CLAMP_L:36].sum(0) + d_mlp[CLAMP_L:36].sum(0)
    log('S1 telescoping [attn 13-35 + mlp 13-35] vs res36_diff: '
        'residual=%.4f' % float(np.linalg.norm(res36_diff - tele)))
    tele2 = d_attn[CLAMP_L:35].sum(0) + d_mlp[CLAMP_L:35].sum(0)
    log('S1 telescoping [13-34 both] vs ln2in35_diff: residual=%.4f'
        % float(np.linalg.norm((C[0][35] - A[0][35]) - tele2)))

    # S2 per-layer arm norms (top 8 by contribution to residual)
    prof = []
    for l in range(13, 36):
        prof.append((l, float(np.linalg.norm(d_attn[l])),
                     float(np.linalg.norm(d_mlp[l]))))
    prof.sort(key=lambda t: -(t[1] + t[2]))
    for l, na, nm in prof[:8]:
        log('S2 L%d ||Dattn||=%.4f ||Dmlp||=%.4f' % (l, na, nm))

    # E: per-state layer recursion check (capture vs hidden_states)
    hs = [h[0, 1].float().cpu().numpy().astype(np.float64)
          for h in C[3].hidden_states]
    hsf = [h[0, 1].float().cpu().numpy().astype(np.float64)
           for h in A[3].hidden_states]
    worst_a = worst_c = 0.0
    for l in range(0, 36):
        err_a = float(np.linalg.norm(
            hsf[l] + A[1][l] + A[2][l] - hsf[l + 1]))
        err_c = float(np.linalg.norm(
            hs[l] + C[1][l] + C[2][l] - hs[l + 1]))
        worst_a = max(worst_a, err_a)
        worst_c = max(worst_c, err_c)
        if err_a > 0.1 or err_c > 0.1:
            log('E layer %d: A-recursion err=%.4f  C-recursion err=%.4f'
                % (l, err_a, err_c))
    log('E max recursion error: A=%.4f C=%.4f' % (worst_a, worst_c))

    open(OUTP, 'w', encoding='utf-8').write('\n'.join(rep))
    print('WROTE %s' % OUTP, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
