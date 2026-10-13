# -*- coding: utf-8 -*-
"""Phase 3192 ledger append: measurement #344 (G5-C5), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 343
assert ms[-1]['phase'] == 3191
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m344 = {
    "phase": 3192, "name": "g5c5_gap4_amplitude_attribution", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5c5_gap4_amplitude_attribution|ustar_amplitude_carrier_x3|amp_ratio_10.5984_7.1100_8.8909|dir_ratio_6.23_4.75_5.38|ks_amp_0.998_0.948_0.972|gap_v15_to_v16_registry_v118_unchanged",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-C5", "superseded_by": None,
    "detail": ("G5-C5 GAP-4 amplitude-level attribution (C line, zero GPU; "
               "sealed 3169 collect x3 + 3169/3189/3190/3191 results). Residual "
               "chain = 3191 verbatim; device anchors: 3189/3190/3191 sealed "
               "scalars drift == 0 bitwise x3, 3169 per-seed E replication 0 "
               "drift - chain identical across 3189..3192. u* = Vt[0] of Qn "
               "(feature space, 3190 verbatim). Gate (pinned pre-observation): "
               "USTAR_AMPLITUDE_CARRIER iff amp_ratio = median(|Rbar @ u*|, 876 "
               "UNnormalized OOV mean-residual rows) / median(|proj|, seen-panel "
               "per-(row,seed) residuals) >= 1.3 in >= 2/3 models. Observed 3/3: "
               "amp_ratio 10.5984/7.1100/8.8909 (med 8.97e1 vs 8.46e0 / 2.53e2 "
               "vs 3.55e1 / 5.00e1 vs 5.63e0) - u* is a SUPER-proportional "
               "carrier: amplitude ratio exceeds the overall collapse ratio_B "
               "2.5388 x3-4x, so the spectral-out 2.54x collapse and the 1.8x "
               "residual are carried PRIMARILY on the shared direction, not "
               "spread across the orthogonal complement. Report-only: "
               "unit-direction projection ratio 6.23/4.75/5.38 (direction "
               "concentration on u* is itself large - consistent with 3190 KS "
               "0.97-0.99), KS(|proj| distributions) = 0.9977/0.9477/0.9715. "
               "CARRIER branch: GAP-4 mechanism_note append "
               "(u*_amplitude_carrier) + evidence append, gap ledger v1.5->v1.6 "
               "(0f6423e0, append-only; other gaps/appendix byte-preserved), "
               "registry v1.18 UNCHANGED (no feature-level change), "
               "atlas_v1_20.html (88300285, 754 fields 0/0/0/0). SMOKE/FULL arm "
               "payload bitwise identical. registry v1.18 c08cbcc2 unchanged, "
               "design efc062be, arm res 6221fe00/seal 58c4f183, final res "
               "7f35fca4/seal b147d3d2. C line 3188-3192 five-phase chain CLOSED: "
               "GAP-4 residual mechanism = shared direction u* carrying "
               "super-proportional OOV amplitude, held-out predictive (E2)."),
    "chain": {"design_sha8": "efc062be", "res_sha8": "7f35fca4", "seal_sha8": "b147d3d2",
              "arm_res_sha8": "6221fe00", "arm_seal_sha8": "58c4f183",
              "gap_v16": "0f6423e0", "registry_v118_unchanged": "c08cbcc2",
              "atlas_v1_20": "88300285"},
}
d['measurements'] = ms + [m344]
assert len(d['measurements']) == 344
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3192')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 344
assert json.dumps(d2['measurements'][:343], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #344 appended: post %s' % post)
