# -*- coding: utf-8 -*-
"""Phase 3189 ledger append: measurement #341 (G5-C2), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 340
assert ms[-1]['phase'] == 3188
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m341 = {
    "phase": 3189, "name": "g5c2_residual_structure", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5c2_residual_structure|shared_oov_collapse_direction|top1_0.3065_0.2936_0.3008|cross_med_0.5441_0.5663_0.4449|within_below_cross_x3|FTR25_E1_repeatable|registry_v116",
    "evidence_level": "E1_repeatable",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-C2", "superseded_by": None,
    "detail": ("G5-C2 class-level/shared-direction structure of the 3169 OOV "
               "collapse residuals (C line, zero GPU; sealed 3169 collect x3 + "
               "3157 collect x3 for dvu re-derivation + 3169/3183/3188 results). "
               "Residual chain = 3188 primary verbatim; row-level directions "
               "(unit of 3-seed mean, 876 rows). Device: 3157 tc anchors 0 drift, "
               "3183 dv replication 0 drift, 3169 per-seed E replication 0 drift, "
               "3188 sealed scalars (share_oov/share_seen/ks/med_oov/med_seen) "
               "drift == 0 bitwise. Gate (pinned): SHARED iff top1>=0.2 & "
               "cross-class median cos>=0.2 x>=2/3. Observed 3/3: top1 "
               "0.3065/0.2936/0.3008, cross_med 0.5441/0.5663/0.4449; within-class "
               "mean cos BELOW cross-class x3 (0.46/0.42/0.50 vs pooled cross) - "
               "collapse direction is a cross-class SHARED direction, not "
               "class-specific. r80 37/47/29, r90 93/120/79 (of 876). GAP-4 "
               "residual 1.8x gets its first structural mechanism clue; FTR-25 "
               "candidate E1_repeatable registered (features 24->25 append-only). "
               "Honest log: SMOKE-1 log-format patch (drift69 % formatting); "
               "FULL-1 crash after atlas write on FTR-25.scope_limits list-vs-"
               "string field convention (pre final_result write; arm res/seal "
               "bitwise identical across reruns 3a26f541/fb8c1f65). registry "
               "v1.15->v1.16 (114aa114), atlas_v1_17.html (4c779d24, 746 fields "
               "0/0/0/0), final res 66a8a8c2/seal cb4c92bb, design 10339511."),
    "chain": {"design_sha8": "10339511", "res_sha8": "66a8a8c2", "seal_sha8": "cb4c92bb",
              "registry_v116": "114aa114", "atlas_v1_17": "4c779d24"},
}
d['measurements'] = ms + [m341]
assert len(d['measurements']) == 341
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3189')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 341
assert json.dumps(d2['measurements'][:340], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #341 appended: post %s' % post)
