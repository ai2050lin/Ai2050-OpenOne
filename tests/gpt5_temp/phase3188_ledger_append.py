# -*- coding: utf-8 -*-
"""Phase 3188 ledger append: measurement #340 (G5-C1), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
PRE = 'da29d7d6'

raw = open(P, 'rb').read()
got = hashlib.sha256(raw).hexdigest()[:8]
assert got == PRE, ('pre sha', got, PRE)
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 339
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m340 = {
    "phase": 3188, "name": "g5c1_gap4_dv_attribution", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5c1_gap4_dv_attribution|dv_direction_decoupled_from_oov_collapse|KS_0.3344_0.1003_0.1913|p_5e-5_0.0105_5e-5|medDiff_+0.0621_-0.0031_-0.0104|sign_inconsistent_x3|FTR24_counter_evidence|registry_v115",
    "evidence_level": "counter_evidence_registered",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-C1", "superseded_by": None,
    "detail": ("G5-C1 GAP-4 residual 1.8x Delta-v directional attribution (C-line "
               "opener, zero GPU; sealed 3169 collect x3 + 3175 collect_out x3 + "
               "3157 collect x3 + 3169/3175/3183 results). 3183-verbatim Delta-v "
               "direction projected on 3169 B-protocol per-cell collapse residual "
               "directions (q = unit(Y - XW) at readout; OOV rows 876 all-3-seed "
               "mean vs seen S1-test row mean). Device gates all 0.0 drift: 3157 "
               "tc anchors, 3183 dv replication, 3169 per-seed E replication, "
               "3175 out-arm k=8 anchor (appendix, 780 rows). Primary gate "
               "(pinned): COUPLED iff KS>=0.15 & perm_p<0.01 in >=2/3 models AND "
               "median-diff sign consistent >=2/3. Observed: KS 0.3344/0.1003/"
               "0.1913, p 5e-5/0.0105/5e-5 (4b+glm4 pass, 14b fail), median diff "
               "+0.0621/-0.0031/-0.0104 -> 4b vs glm4 SIGN OPPOSITION => "
               "DECOUPLED. Direction-share (report-only): OOV 0.0088/0.0091/0.0060 "
               "vs seen 0.0100/0.0142/0.0193, control ~1/D (3.9e-4/2.0e-4/2.4e-4) "
               "- share tiny everywhere and OOV<=seen x3, consistent with "
               "decoupling. GAP-4 residual 1.8x is decoupled from the relation "
               "offset direction at direction level; counter_evidence appended to "
               "FTR-24 (statement/anchors byte-preserved). registry v1.14->v1.15 "
               "(0a55d2e7, features 24, failures 19), atlas_v1_16.html (384d0834, "
               "734 fields 0/0/0/0), final res 8c1d94a3/seal ad0444ad, design "
               "977bb327."),
    "chain": {"design_sha8": "977bb327", "res_sha8": "8c1d94a3", "seal_sha8": "ad0444ad",
              "registry_v115": "0a55d2e7", "atlas_v1_16": "384d0834"},
}
d['measurements'] = ms + [m340]
assert len(d['measurements']) == 340
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob[:-1].startswith(old_blob[:-1].rstrip(',') ) or \
    new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3188')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 340
assert json.dumps(d2['measurements'][:339], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #340 appended: pre %s post %s' % (PRE, post))
