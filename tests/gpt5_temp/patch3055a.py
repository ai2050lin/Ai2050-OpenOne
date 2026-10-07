# -*- coding: utf-8 -*-
import io
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3055_omega_p52_gamma_prealign_qwen.py'
s = io.open(p, encoding='utf-8').read()

# 1) add forward_plain after forward_run def
anchor1 = """    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    reset_all()
    return lg


# ---------- chain: load capture banks ----------"""
assert s.count(anchor1) == 1, s.count(anchor1)
new1 = """    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    reset_all()
    return lg


def forward_plain(ids):
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids],
                    device='cuda'), use_cache=True)
    lg = out.logits[0, -1].detach().double() \\
        .cpu().numpy()
    reset_all()
    return lg


# ---------- chain: load capture banks ----------"""
s = s.replace(anchor1, new1)

# 2) fix the a173 baseline
anchor2 = """    lg_base = forward_run(ids0,
                          np.zeros_like(KPpost[
                              base_i0, :, :n0, :]),
                          np.zeros(n0, dtype=bool),
                          np.zeros_like(VP[base_i0,
                                       :, :n0, :]),
                          np.zeros(n0, dtype=bool))"""
assert s.count(anchor2) == 1, s.count(anchor2)
s = s.replace(anchor2,
              '    lg_base = forward_plain(ids0)')

# 3) register the run1 crash in corrections
anchor3 = """                             'pre-alignment, not a '
                             'change of target; '
                             'verdict in one branch',
}"""
assert s.count(anchor3) == 1, s.count(anchor3)
new3 = """                             'pre-alignment, not a '
                             'change of target; '
                             'verdict in one branch',
    'corrections': 'run1 (55s) crashed pre-'
                   'verdict at the a173 baseline: '
                   'the no-replacement baseline '
                   'was built as zeros-repl + '
                   'all-False mask, but the hook '
                   'still enters the assignment '
                   'branch when repl is not None '
                   'and the boolean index selects '
                   '0 rows vs repl 6 rows '
                   '(broadcast error); fix: '
                   'dedicated forward_plain path '
                   'with no replacement state. '
                   'Statistics unobserved at '
                   'crash time (T3 perturbed-gamma '
                   'arms not started; a169/a170/'
                   'a172 bit 0.0 already passed '
                   'and T2 numbers printed are '
                   'valid but re-measured in '
                   'run2). run2 authoritative if '
                   'anchors pass.',
}"""
s = s.replace(anchor3, new3)

io.open(p, 'w', encoding='utf-8').write(s)
print('patched')
