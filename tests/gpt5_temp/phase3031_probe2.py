# -*- coding: utf-8 -*-
"""3031 预注册前探针 2: 3014/3015 npz 键 + 3022 份额公式 + 锚可行性."""
import numpy as np
import io
import os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
out = []

for ph, name in [('phase3014', 'omega_p2i_kv_dose_routing_qwen'),
                 ('phase3015', 'omega_p2j_k_consumption_qwen'),
                 ('phase3016', 'omega_p2k_amplification_trace_qwen')]:
    p = os.path.join(BASE, ph, name, name + '.npz')
    if not os.path.exists(p):
        out.append('MISSING %s' % p)
        continue
    z = np.load(p, allow_pickle=True)
    out.append('=== %s %s ===' % (ph, name))
    for k in z.files:
        a = z[k]
        try:
            out.append('  %-40s %-18s %s' % (k, str(a.shape), str(a.dtype)))
        except Exception as e:
            out.append('  %s ERR %s' % (k, e))

# --- 锚可行性: 3029.js_alpha2 vs 3028 js_dose[:, -1] (alpha=2) ---
z28 = np.load(os.path.join(BASE, 'phase3028', 'omega_p2v_dose_symmetry_qwen', 'omega_p2v_dose_symmetry_qwen.npz'), allow_pickle=True)
z29 = np.load(os.path.join(BASE, 'phase3029', 'omega_p2w_recruitment_decomp_qwen', 'omega_p2w_recruitment_decomp_qwen.npz'), allow_pickle=True)
a28 = z28['alphas']
out.append('3028 alphas = %s' % str(a28))
js2_28 = z28['js_dose'][:, -1] if a28[-1] == 2.0 else None
js2_29 = z29['js_alpha2']
d = np.abs(js2_28 - js2_29).max()
out.append('anchor_feasible |3028 js2 - 3029 js2| max = %.3e' % d)

# rel_dev 重算恒等
js0 = z28['js_dose'][:, 0] if a28[0] == 0.0 else None
pred2 = 2.0 * z28['js_erase'] - js0
rel = (js2_28 - pred2) / pred2
out.append('rel_dev recompute max abs diff = %.3e' % np.abs(rel - z28['rel_dev']).max())

# tags 一致性
tags = [str(t) for t in z28['tags']]
for zz, nm in [(z29, '3029')]:
    out.append('tags %s identical: %s' % (nm, str([str(t) for t in zz['tags']]) == tags))

# --- 3020 traj 末列 vs js_final_logic (key 对齐) ---
z20 = np.load(os.path.join(BASE, 'phase3020', 'omega_p2n_readout_specificity_qwen', 'omega_p2n_readout_specificity_qwen.npz'), allow_pickle=True)
keys20 = [str(t) for t in z20['traj_keys_logic']]
tags20 = [str(t) for t in z20['tags']]
idx = [keys20.index(t) for t in tags20]
traj_aligned = z20['traj_logic'][idx, :]
jfl = z20['js_final_logic']
dd = np.abs(traj_aligned[:, -1] - jfl).max()
out.append('anchor_feasible 3020 traj last col vs js_final_logic (aligned) = %.3e' % dd)
out.append('3020 traj_keys_logic = %s' % str(keys20))

# --- 3022 s_relay top32 share per tag (探查量级) ---
z22 = np.load(os.path.join(BASE, 'phase3022', 'omega_p2p_l3_relay_neurons_qwen', 'omega_p2p_l3_relay_neurons_qwen.npz'), allow_pickle=True)
s = z22['s_relay'].astype(np.float64)
sp = s.sum(axis=0)
top32 = np.argpartition(sp, -32)[-32:]
for i in range(3):
    share = s[i][top32].sum() / np.abs(s[i]).sum()
    out.append('tag %s top32 pos-share %.4f' % (tags[i], share))

# --- tokenizer 位置特征可行性 ---
tok_dir = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    prompts = [str(p) for p in z28['prompts']]
    posinfo = []
    for i, t in enumerate(tags):
        pr = prompts[i]
        toks = tok.encode(pr)
        # 找 tag 的 token 位置: 用 offset 或逐 token decode 匹配
        found = -1
        ids = tok.encode(t, add_special_tokens=False)
        for p0 in range(len(toks) - len(ids) + 1):
            if toks[p0:p0 + len(ids)] == ids:
                found = p0
                break
        posinfo.append((t, found, len(toks), round(found / max(len(toks) - 1, 1), 4)))
    out.append('tokenizer positions:')
    for x in posinfo:
        out.append('  %s' % str(x))
except Exception as e:
    out.append('TOKENIZER ERR %r' % e)

rep = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3031_probe2.txt'
io.open(rep, 'w', encoding='utf-8').write('\n'.join(out))
print('WROTE')
