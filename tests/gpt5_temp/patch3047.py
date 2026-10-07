# -*- coding: utf-8 -*-
import io
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5' \
    r'\phase3047_omega_p44_kv_joint_replay_qwen.py'
s = io.open(P, encoding='utf-8').read()

START = 'SL = slice(KV_HEAD * HDIM'
END = "# ---------- chain sources ----------"
i0 = s.index(START)
i1 = s.index(END)
assert i0 < i1

NEW = '''SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)
stateK = {li: {'on': False, 'pos': -1,
               'delta': None} for li in range(NL)}
stateV = {li: {'on': False, 'pos': -1,
               'delta': None} for li in range(NL)}
capK = {li: {'rec': False, 'pos': -1,
             'orig': None, 'mod': None}
        for li in range(NL)}
capV = {li: {'rec': False, 'pos': -1,
             'orig': None, 'mod': None}
        for li in range(NL)}


def make_hook(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0, cp['pos'], SL] \\
                .detach().clone()
        if st['on']:
            out[0, st['pos'], SL] += st['delta']
        if cp['rec']:
            cp['mod'] = out[0, cp['pos'], SL] \\
                .detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_proj \\
        .register_forward_hook(make_hook(stateK[li],
                                         capK[li]))
    layers[li].self_attn.v_proj \\
        .register_forward_hook(make_hook(stateV[li],
                                         capV[li]))


def reset_all():
    for li in range(NL):
        stateK[li]['on'] = False
        stateV[li]['on'] = False
        capK[li]['rec'] = False
        capV[li]['rec'] = False


def clear_caps():
    for li in range(NL):
        capK[li]['orig'] = None
        capK[li]['mod'] = None
        capV[li]['orig'] = None
        capV[li]['mod'] = None


def forward_cap(ids, pos):
    reset_all()
    clear_caps()
    for li in range(NL):
        capK[li].update(rec=True, pos=pos)
        capV[li].update(rec=True, pos=pos)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    kp = np.stack([capK[li]['orig'].double().cpu()
                   .numpy() for li in range(NL)])
    vp = np.stack([capV[li]['orig'].double().cpu()
                   .numpy() for li in range(NL)])
    reset_all()
    return lg, kp, vp


def past_numpy(past):
    pk = {}
    pv = {}
    for li in range(NL):
        pk[li] = past.layers[li].keys[0] \\
            .detach().double().cpu().numpy()
        pv[li] = past.layers[li].values[0] \\
            .detach().double().cpu().numpy()
    return pk, pv


def forward_past(ids):
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    pk, pv = past_numpy(out.past_key_values)
    return lg, pk, pv


def forward_inj(ids, pos, dK, dV, force_on=False,
                integ=False, want_past=False):
    reset_all()
    clear_caps()
    for li in range(NL):
        onk = force_on or float(
            np.max(np.abs(dK[li]))) > 0.0
        onv = force_on or float(
            np.max(np.abs(dV[li]))) > 0.0
        if onk:
            stateK[li].update(
                on=True, pos=pos,
                delta=torch.tensor(
                    np.ascontiguousarray(dK[li]),
                    dtype=torch.float32,
                    device='cuda'))
        if onv:
            stateV[li].update(
                on=True, pos=pos,
                delta=torch.tensor(
                    np.ascontiguousarray(dV[li]),
                    dtype=torch.float32,
                    device='cuda'))
        if integ:
            capK[li].update(rec=True, pos=pos)
            capV[li].update(rec=True, pos=pos)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    res = {'lg': lg}
    if integ:
        ik = np.stack([(capK[li]['mod']
                        - capK[li]['orig'])
                       .double().cpu().numpy()
                       for li in range(NL)])
        iv = np.stack([(capV[li]['mod']
                        - capV[li]['orig'])
                       .double().cpu().numpy()
                       for li in range(NL)])
        res['integK'] = ik
        res['integV'] = iv
    reset_all()
    if want_past:
        res['pk'], res['pv'] = past_numpy(
            out.past_key_values)
    return res


'''

s = s[:i0] + NEW + s[i1:]
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

# verify on disk
s2 = io.open(P, encoding='utf-8').read()
checks = {
    'clear_caps def': s2.count('def clear_caps():') == 1,
    'reset before read in cap':
        s2.index('clear_caps()') < s2.index('def forward_past'),
    'cap reads before reset':
        s2.index("kp = np.stack") < s2.index("    reset_all()\n    return lg, kp, vp"),
    'inj reads integ before reset':
        s2.index("res['integK'] = ik") < s2.index("    reset_all()\n    if want_past:"),
    'no old buggy order':
        '    reset_all()\n    lg = out.logits' not in s2,
}
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3047_result.txt', 'w',
        encoding='utf-8').write(
    '\n'.join('%s=%s' % kv for kv in checks.items())
    + '\ncompile ok\n')
print('ok')
