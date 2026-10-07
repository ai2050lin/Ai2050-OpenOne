# -*- coding: utf-8 -*-
"""Phase 3017 patch2: add TRUE residual-stream capture
(decoder-layer pre-hook) - cap['ai'] is post-LN and
must stay for the a2 dirs_word anchor; run_chain now
returns the true residual for the recursion identity."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3017_omega_p2k_deep_absorption_qwen.py')
t = io.open(P, encoding='utf-8').read()
miss = []

# 1. state + dict
a = """    state_c = {'on': False}
    ao = {}
    mo = {}
    handles = []"""
b = """    state_c = {'on': False}
    ao = {}
    mo = {}
    state_r = {'on': False}
    rs = {}
    handles = []"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('state_r')

# 2. hook definition (insert before cap_attn)
a = """    def cap_attn(li):"""
b = """    def pre_layer(li):
        def h(module, args, kwargs):
            if state_r['on']:
                x = args[0] if args \\
                    else kwargs.get('hidden_states')
                if x is not None and x.dim() >= 2:
                    rs.setdefault(li, []).append(
                        x[:, -1, :].detach().float()
                        .cpu().numpy().copy())
            return None
        return h

    def cap_attn(li):"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('pre_layer-def')

# 3. registration
a = """        handles.append(layers[li].mlp
                       .register_forward_hook(
                           cap_mlp(li)))"""
b = """        handles.append(layers[li].mlp
                       .register_forward_hook(
                           cap_mlp(li)))
        handles.append(layers[li]
                       .register_forward_pre_hook(
                           pre_layer(li),
                           with_kwargs=True))"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('pre_layer-reg')

# 4. run_chain: clear rs, toggle state_r, use rs
a = """        clear_cap()
        ao.clear()
        mo.clear()
        state_c['on'] = True"""
b = """        clear_cap()
        ao.clear()
        mo.clear()
        rs.clear()
        state_c['on'] = True
        state_r['on'] = True"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('runchain-on')

a = """                use_cache=False)
        state_c['on'] = False
        lg = out2.logits[0, -1].detach() \\"""
b = """                use_cache=False)
        state_c['on'] = False
        state_r['on'] = False
        lg = out2.logits[0, -1].detach() \\"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('runchain-off')

a = """        res36 = np.stack(
            [cap['ai'][li][-1][0, 0, :].astype(
                np.float64) for li in range(NL)])"""
b = """        res36 = np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('res36-rs')

io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
chk = {
    'state_r': "state_r = {'on': False}" in t2,
    'pre_layer': "def pre_layer(li):" in t2,
    'reg': "pre_layer(li),\n                           with_kwargs=True))" in t2,
    'on': "state_r['on'] = True" in t2,
    'off': "state_r['on'] = False" in t2,
    'rs-res36': "[rs[li][-1][0].astype(np.float64)" in t2,
    'ai-untouched': "cap['ai'][li][-1][0, 0, :]" in t2,
}
# ai-untouched should now be False in run_chain
# (replaced) - but cap['ai'] capture itself remains
n_ai = t2.count("cap['ai']")
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p17b.txt', 'w',
        encoding='utf-8').write(
    'miss=%s chk=%s n_cap_ai=%s'
    % (miss, chk, n_ai))
print('ok')
