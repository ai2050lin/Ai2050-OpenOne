"""Phase 2776: carrier-ablation replication on Qwen3-14B (fourth scale).

Restores the scale point blocked in Phase 2774 (system-memory degradation
crashed every 14B load path; RAM now freed to ~24 GiB available, GPU 14.6
GiB free).  Reuses the loader proven in Phase 2774b on DS-7B: meta model,
manual per-parameter assignment from the original shards with ONE
persistent safe_open handle per shard, meta-buffer fix (inv_freq), then
dispatch.  Non-quantized, eager attention, bf16.

Placement (frozen before any forward): embed_tokens + layers 0..19 on CPU,
layers 20..39 + lm_head on GPU.  Module layers M={35..39} (last 5 of 40)
therefore run on GPU.  Per-layer ~0.51 GiB -> GPU ~12.0 GiB, CPU ~11.7 GiB.

Replicates the Phase 2769/2774b protocol: native wrong rows of the 320
controlled_relation diagnostic, per-row rival direction, top-4
positive-gain carrier heads in M, targeted o_proj-input ablation at the
final position, random-head null, union-set collateral.

Prompts re-tokenised: qwen4 ids -> text (qwen4 decode) -> qwen14 ids.
Targets from row['target_text'] (first 14B-model token).

Preregistered (frozen before any 14B forward; identical wording to 2774b):
  E1: targeted flips >= 4 AND targeted total > null q95 AND
      null mean <= targeted/3.
  E2: union carrier-head set breaks <= 2/20 native-correct rows.
  verdict: carrier_replicated_14b iff E1 and E2.
Gates: G0 decode roundtrip; G1 determinism (first 4 rows, exact argmax);
  G2 no meta params/buffers remain; G3 target id stability recorded.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2776' / 'qwen14b_replication'
M_LAYERS = [35, 36, 37, 38, 39]  # last 5 of 40
TOP_HEADS = 4
N_NULL_ROWS = 20
N_NULL_DRAWS = 10
NULL_SEED = 27760
N_COLL_ROWS = 20
N_CPU_LAYERS = 20  # layers 0..19 on CPU (embed too); 20..39 on GPU

PREREG = {
    'E1': 'targeted flips >= 4 AND targeted total > null q95 AND '
          'null mean <= targeted/3',
    'E2': 'union carrier-head set breaks <= 2/20 native-correct rows',
    'verdict': 'carrier_replicated_14b iff E1 and E2',
}


def build_model_manual():
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import dispatch_model

    path = str(ROOT / 'models' / 'hf' / 'Qwen3-14B')
    cfg = AutoConfig.from_pretrained(path, local_files_only=True,
                                     trust_remote_code=True)
    assert not getattr(cfg, 'tie_word_embeddings', False), 'tied weights'
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            cfg, attn_implementation='eager', dtype=torch.bfloat16)

    plan = {'model.embed_tokens': 'cpu', 'lm_head': 0, 'model.norm': 0}
    for i in range(cfg.num_hidden_layers):
        plan['model.layers.%d' % i] = 'cpu' if i < N_CPU_LAYERS else 0

    idx = json.load(open(path + r'\model.safetensors.index.json',
                         encoding='utf-8'))
    wm = idx['weight_map']

    def device_for(name):
        best = None
        for k in plan:
            if name == k or name.startswith(k + '.'):
                if best is None or len(k) > len(best):
                    best = k
        if best is None:
            return 'cpu'  # root-level aux modules (model.rotary_emb)
        return plan[best]

    # ONE persistent safe_open handle per shard: rapid open/unmap cycles of
    # multi-GB views fragment the address space on this machine (the
    # Phase 2774 segfault path).
    shard_handles = {}
    for sh in sorted(set(wm.values())):
        shard_handles[sh] = safe_open(path + '\\' + sh, framework='pt',
                                      device='cpu')

    n_assigned = 0
    for mod_name, mod in model.named_modules():
        for pname, p in list(mod.named_parameters(recurse=False)):
            if p is None:
                continue
            name = (mod_name + '.' + pname) if mod_name else pname
            assert p.is_meta, name
            t = shard_handles[wm[name]].get_tensor(name)
            dev = device_for(name)
            # swap_tensors avoids nn.Parameter subclass construction on a
            # live CUDA context (access violation path); the meta parameter
            # becomes the real tensor in place.
            torch.utils.swap_tensors(p, t.to(dev))
            del t
            # swap_tensors swaps __class__ too: p is now a plain Tensor.
            # Re-wrap as Parameter (primitive proven safe at 7B) so
            # dispatch_model sees proper _parameters entries.
            if not isinstance(p, torch.nn.Parameter):
                mod._parameters[pname] = torch.nn.Parameter(
                    p, requires_grad=False)
            n_assigned += 1
            if n_assigned % 25 == 0:
                free, _ = torch.cuda.mem_get_info()
                print('BUILD %d assigned, gpu_free=%.2f GiB (%s)'
                      % (n_assigned, free / 2**30, name), flush=True)

    # fix meta buffers (rotary inv_freq)
    n_buf_fix = 0
    for name, b in model.named_buffers():
        if not b.is_meta:
            continue
        assert name.endswith('inv_freq'), name
        hd = (getattr(cfg, 'head_dim', None) or
              cfg.hidden_size // cfg.num_attention_heads)
        theta = float(getattr(cfg, 'rope_theta', 1000000.0))
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2,
                                            dtype=torch.float32) / hd))
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        setattr(parent, name.rsplit('.', 1)[1], inv.to(device_for(name)))
        n_buf_fix += 1

    for name, p in model.named_parameters():
        assert not p.is_meta, ('meta param left', name)
    for name, b in model.named_buffers():
        assert not b.is_meta, ('meta buffer left', name)

    model.eval()
    W_O = {}
    for l in M_LAYERS:
        W_O[l] = model.model.layers[l].self_attn.o_proj.weight.            detach().float().cpu().numpy()
    model = dispatch_model(model, device_map=plan)
    return model, cfg, plan, n_assigned, n_buf_fix, W_O


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)

    tok4 = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    tok14 = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'Qwen3-14B'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    id_list = []
    tgt_ids = np.empty(len(rows), dtype=np.int64)
    n_multi = 0
    for r_i, r in enumerate(rows):
        text = tok4.decode(r['prompt_ids'])
        ids = tok14(text, add_special_tokens=False)['input_ids']
        id_list.append(ids)
        tt = tok14(r['target_text'], add_special_tokens=False)['input_ids']
        if len(tt) > 1:
            n_multi += 1
        tgt_ids[r_i] = tt[0]
    g0_ok = all(tok14.decode(id_list[i]) == tok4.decode(rows[i]['prompt_ids'])
                for i in range(8))
    assert g0_ok, 'G0 decode roundtrip'
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)

    model, cfg, plan, n_assigned, n_buf_fix, W_O = build_model_manual()
    assert cfg.num_hidden_layers == 40
    device = torch.device('cuda')
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    group = n_heads // n_kv

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'n_multi_token_targets': n_multi,
                 'loader': 'manual meta+shard assign (persistent shard handles), dispatch CPU20(embed+L0-19)/GPU20(L20-39+lm_head)',
                 'n_params_assigned': n_assigned,
                 'n_buffers_fixed': n_buf_fix}
    fc.save(OUT / 'execution.json', execution)

    cap = {}
    handles = []
    abl_state = {'pairs': frozenset()}

    def oproj_pre(l):
        def hook(module, args):
            cap.setdefault('oproj', {})[l] = args[0].detach()
        return hook

    def abl_pre(l):
        def hook(module, args):
            hs_ = [h for (ll, h) in abl_state['pairs'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return hook

    for l in M_LAYERS:
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(oproj_pre(l)))
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(abl_pre(l)))

    def forward(i, want_att=False):
        with torch.inference_mode():
            t = torch.tensor([id_list[i]], device=device)
            o = model(t, output_attentions=want_att)
            return o

    a0 = [int(forward(i).logits[0, -1].float().argmax())
          for i in range(4)]
    a1 = [int(forward(i).logits[0, -1].float().argmax())
          for i in range(4)]
    assert a0 == a1, 'G1 determinism'
    print('P2776 G1_OK', flush=True)

    arg_native = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        arg_native[i] = int(forward(i).logits[0, -1].float().argmax())
        if (i + 1) % 40 == 0:
            print('P2776 NATIVE %d/320' % (i + 1), flush=True)
    wrong_all = arg_native != tgt_ids
    fam_wrong = {f: int(wrong_all[fam_arr == f].sum())
                 for f in sorted(set(fam_arr))}
    wrong_idx = [int(i) for i in np.where(wrong_all)[0]]
    print('P2776 NATIVE_DONE wrong_total=%d %s'
          % (len(wrong_idx), json.dumps(fam_wrong)), flush=True)

    W_U = model.get_submodule('lm_head').weight  # on cuda:0

    carrier = {}
    targeted = {}
    for j, i in enumerate(wrong_idx):
        cap.clear()
        o = forward(i)
        zi = o.logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        u = (W_U[rival] - W_U[tgt_ids[i]]).float()
        u = (u / u.norm()).cpu().numpy()
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = W_O[l]
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        ch = [(l, h) for (l, h), g in
              sorted(gains.items(), key=lambda x: -x[1]) if g > 0][:TOP_HEADS]
        carrier[i] = ch
        abl_state['pairs'] = frozenset(ch)
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
        targeted[i] = bool(m == tgt_ids[i])
        if (j + 1) % 10 == 0:
            print('P2776 CARRIER %d/%d flips=%d'
                  % (j + 1, len(wrong_idx), sum(targeted.values())),
                  flush=True)

    rng = np.random.default_rng(NULL_SEED)
    null_rows = list(rng.choice(wrong_idx, size=min(N_NULL_ROWS,
                                                    len(wrong_idx)),
                                replace=False))
    null_totals = []
    for i in null_rows:
        tot = 0
        for _ in range(N_NULL_DRAWS):
            pairs = [(int(l), int(h))
                     for l in rng.choice(M_LAYERS, size=TOP_HEADS)
                     for h in [int(rng.integers(0, n_heads))]]
            abl_state['pairs'] = frozenset(pairs)
            try:
                m = int(forward(i).logits[0, -1].float().argmax())
            finally:
                abl_state['pairs'] = frozenset()
            tot += int(m == tgt_ids[i])
        null_totals.append(tot)
    null_totals = np.array(null_totals)

    union_pairs = sorted({p for i in wrong_idx for p in carrier[i]})
    correct_idx = [int(i) for i in np.where(~wrong_all)[0]]
    coll_rows = list(rng.choice(correct_idx,
                                size=min(N_COLL_ROWS, len(correct_idx)),
                                replace=False))
    coll_breaks = 0
    for i in coll_rows:
        abl_state['pairs'] = frozenset(union_pairs)
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
        coll_breaks += int(m != arg_native[i])

    n_flips = int(sum(targeted.values()))
    null_q95 = float(np.percentile(null_totals, 95)) if len(null_totals) \
        else 0.0
    e1_pass = bool(n_flips >= 4 and n_flips > null_q95 and
                   float(null_totals.mean()) <= n_flips / 3.0)
    e2_pass = bool(coll_breaks <= 2)
    verdict = {'E1_pass': e1_pass, 'E2_pass': e2_pass,
               'carrier_replicated_14b': bool(e1_pass and e2_pass),
               'n_wrong': len(wrong_idx), 'targeted_flips': n_flips,
               'null_mean': float(null_totals.mean()),
               'null_q95': null_q95, 'coll_breaks': coll_breaks,
               'n_union_heads': len(union_pairs),
               'fam_wrong': fam_wrong}

    result = {'phase': '2776', 'prereg': PREREG, 'verdict': verdict,
              'carrier_heads': {str(i): [list(p) for p in carrier[i]]
                                for i in wrong_idx},
              'targeted': {str(i): targeted[i] for i in wrong_idx},
              'null_totals': null_totals.tolist(),
              'null_rows': [int(x) for x in null_rows],
              'coll_rows': [int(x) for x in coll_rows]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'carrier_stats_14b.npz', wrong_idx=np.array(wrong_idx),
           arg_native=arg_native, tgt_ids=tgt_ids,
           null_totals=null_totals)
    for h in handles:
        h.remove()
    print('P2776 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
