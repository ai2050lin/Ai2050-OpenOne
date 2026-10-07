# -*- coding: utf-8 -*-
"""Phase 3013 patch2: kv_replace view fix (Edit phantom
workaround - apply directly on disk)."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3013_omega_p2g_kv_content_decomposition_'
     r'qwen.py')
t = io.open(P, encoding='utf-8').read()

old = (
    "    def kv_replace(past, p, k_vec, v_vec, "
    "li=L3_GATED):\n"
    "        L = past.layers[li]\n"
    "        L.keys[:, :, p, :] = k_vec\n"
    "        L.values[:, :, p, :] = v_vec\n")
new = (
    "    def kv_replace(past, p, k_vec, v_vec, "
    "li=L3_GATED):\n"
    "        L = past.layers[li]\n"
    "        L.keys[:, :, p, :] = k_vec.view(\n"
    "            L.keys[:, :, p, :].shape)\n"
    "        L.values[:, :, p, :] = v_vec.view(\n"
    "            L.values[:, :, p, :].shape)\n")
assert old in t, 'OLD NOT FOUND'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
assert 'k_vec.view(' in t2, 'patch not on disk'
assert old not in t2, 'old still there'
print('patched ok')

# append run3 note to correction_note
old2 = ("            'deviation from PREREG); run2: '\n"
        "            'authoritative',")
new2 = ("            'deviation from PREREG); run2: '\n"
        "            'crashed at the pool_mean arm - the '\n"
        "            'kv_replace view fix was a phantom '\n"
        "            'edit (reported success, absent on '\n"
        "            'disk), 1-D flat pool vector hit '\n"
        "            'expanded-size mismatch [1,8,128] vs '\n"
        "            '[1024]; patch reapplied via direct '\n"
        "            'disk write; no design change; '\n"
        "            'run3: authoritative',")
assert old2 in t2, 'NOTE NOT FOUND'
t2 = t2.replace(old2, new2, 1)
io.open(P, 'w', encoding='utf-8').write(t2)
t3 = io.open(P, encoding='utf-8').read()
assert 'run3: authoritative' in t3
print('note ok')
