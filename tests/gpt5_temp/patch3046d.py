# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = ("    lg = out.logits[0, -1].detach().double() \\\n"
        "        .cpu().numpy()\n"
        "    return lg, out.attentions\n\n\n"
        "def forward_inj_attn(")
new1 = ("    lg = out.logits[0, -1].detach().double() \\\n"
        "        .cpu().numpy()\n"
        "    attns = [a.detach().double().cpu()\n"
        "           .numpy() for a in out.attentions]\n"
        "    return lg, attns\n\n\n"
        "def forward_inj_attn(")
assert s.count(old1) == 1, ('old1', s.count(old1))
s = s.replace(old1, new1)

old2 = ("    state[li]['on'] = False\n"
        "    lg = out.logits[0, -1].detach().double() \\\n"
        "        .cpu().numpy()\n"
        "    return lg, out.attentions")
new2 = ("    state[li]['on'] = False\n"
        "    lg = out.logits[0, -1].detach().double() \\\n"
        "        .cpu().numpy()\n"
        "    attns = [a.detach().double().cpu()\n"
        "           .numpy() for a in out.attentions]\n"
        "    return lg, attns")
assert s.count(old2) == 1, ('old2', s.count(old2))
s = s.replace(old2, new2)

old3 = ("    'corrections': 'run1 crashed pre-anchor "
        "at prompt '")
new3 = ("    'corrections': 'run2 crashed mid-T4: "
        "attention '\n                   'tensors were left on cuda "
        "(numpy '\n                   'conversion failed); T1/T2/T3 "
        "were '\n                   'observed and printed and "
        "reproduce '\n                   'under frozen seeds; fixed "
        "by '\n                   'detaching the attention tuple "
        "to '\n                   'numpy inside "
        "forward_attn/'\n                   'forward_inj_attn; "
        "run3 authoritative '\n                   'candidate; run1 "
        "crashed pre-anchor '\n                   'at prompt "
        "assembly: the NEW-bodies '\n                   'loop "
        "else-branch used BODIES[bi] '\n                   '"
        "(transcription slip; 3045 line 409 '\n                   '"
        "correctly reads NEW_BODIES[bi]); '\n                   'NEW "
        "b3 cond0 target although had '\n                   'count 0 "
        "in the wrong sentence; '\n                   'fixed to "
        "NEW_BODIES[bi]; no '\n                   'anchor or "
        "statistic observed '\n                   'before the "
        "run1 crash',")
n_cor = s.count(old3)
assert n_cor == 1, ('old3', n_cor)
s = s.replace(old3, new3)

old4 = "    'run': 'run2 authoritative (fp32;"
new4 = "    'run': 'run3 authoritative (fp32;"
assert s.count(old4) == 1, ('old4', s.count(old4))
s = s.replace(old4, new4)
io.open(P, 'w', encoding='utf-8').write(s)

py_compile.compile(P, doraise=True)
print('ok')
