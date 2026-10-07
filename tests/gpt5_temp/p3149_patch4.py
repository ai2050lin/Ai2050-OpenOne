# -*- coding: utf-8 -*-
"""p3149 patch4: chunked fp32 unembed
matmul (OOM guard). Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()


def rep(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, (tag, n)
    s = s.replace(old, new)
    print('OK', tag)


# 1) replace WUG_T with chunked helper
old1 = ("log('== PART T2: carrier dlogit '\n"
        "        \"==')\n"
        "WUG_T = WUG.detach().float()")
if s.count(old1) != 1:
    # fallback simpler anchor
    old1 = ("log('== PART T2: carrier "
            "dlogit ==')\n"
            "WUG_T = WUG.detach().float()")
new1 = ("log('== PART T2: carrier "
        "dlogit ==')\n"
        "\n"
        "\n"
        "def _dl_vec(dh):\n"
        "    \"\"\"dlogit vector via chunked\n"
        "    fp32 matmul against WUG (GPU\n"
        "    mem-safe, 268MB peak per\n"
        "    chunk).\"\"\"\n"
        "    ots = []\n"
        "    for c0 in range(0, VOCAB,\n"
        "                    16384):\n"
        "        Wc = WUG[c0:c0 + 16384] \\\n"
        "            .float()\n"
        "        ots.append(dh @ Wc.T)\n"
        "        del Wc\n"
        "    return torch.cat(ots)")
rep(old1, new1, 'wug helper')

# 2) T2 matmul (no line-end bkslash in
# anchor)
rep("dl = ((hi - hb) @ WUG_T.T)",
    "dl = _dl_vec(hi - hb)",
    't2 matmul')

# 3) L: w131 def after wdn_np
rep("wdn_np = w_dn_g.astype(np.float64)",
    "wdn_np = w_dn_g.astype(np.float64)\n"
    "    w131 = WUG[SPEC_TOKS[0]] \\\n"
    "        .float()",
    'w131 def')

# 4) L d131 single-row dot
rep("((hi - hb)\n"
    "                 @ WUG_T.T)[0,",
    "((hi - hb)\n"
    "                 @ w131)[0,",
    'L d131')

assert 'WUG_T' not in s, 'WUG_T residual'
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH4_DONE')
