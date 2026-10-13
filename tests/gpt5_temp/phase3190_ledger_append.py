# -*- coding: utf-8 -*-
"""Phase 3190 ledger append: measurement #342 (G5-C3), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 341
assert ms[-1]['phase'] == 3189
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m342 = {
    "phase": 3190, "name": "g5c3_shared_direction_robustness", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5c3_shared_direction_robustness|loo_robust_x3|loo_max_delta_0.0342_0.0214_0.0332|ks_ustar_oov_vs_seen_0.9932_0.9761_0.9715|abs_cos_ustar_vc_0.72_0.85|FTR25_anchors_append_E1_unchanged|registry_v117",
    "evidence_level": "E1_repeatable",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-C3", "superseded_by": None,
    "detail": ("G5-C3 robustness + content attribution of the shared OOV collapse "
               "direction (C line, zero GPU; sealed 3169 collect x3 + 3189 result). "
               "Residual chain = 3189 verbatim; device anchors: 3189 sealed scalars "
               "(top1/top2/cross_med/within/r80/r90) drift == 0 bitwise x3, 3169 "
               "per-seed E replication 0 drift x3. u* = top-1 singular direction of "
               "the 876 OOV row directions in FEATURE space (Vt[0]; SMOKE-1 fix: "
               "U[:,0] is sample-space 876-dim and cannot be projected against Qn "
               "rows or v_c - honest log). Gate (pinned pre-observation): ROBUST iff "
               "LOO 4-fold (drop one class, 219 rows) top1 max |delta| < 0.05 in "
               ">= 2/3 models. Observed 3/3 ROBUST: LOO top1 4b "
               "0.3246/0.3408/0.3118/0.3217 max_delta 0.0342; 14b "
               "0.3136/0.3147/0.2937/0.3149 max_delta 0.0214; glm4 "
               "0.3181/0.3305/0.3140/0.3340 max_delta 0.0332 - shared direction is "
               "NOT single-class carried. Report-only attributions: (a) KS(u* "
               "projections; 876 OOV vs seen-panel baseline directions) = "
               "0.9932/0.9761/0.9715 - the shared collapse direction is essentially "
               "ABSENT from the seen-panel baseline residual directions (OOV-specific "
               "geometry, consistent with GAP-4 spectral-out collapse 2.54x); "
               "(b) preregistered downgrade cos(u*, v_c) per class = 4b "
               "-0.806/-0.762/-0.836/-0.819, 14b -0.807/-0.797/-0.852/-0.808, glm4 "
               "-0.778/-0.746/-0.802/-0.724 (sign arbitrary for singular vectors; "
               "|cos| 0.72-0.85) - u* strongly aligned with ALL four class-mean "
               "directions, i.e. the shared direction is the common core of the "
               "class-level collapse geometry. ROBUST branch executed: FTR-25 "
               "anchors append (role=loo_robustness) + statement append; E level "
               "UNCHANGED E1_repeatable (stability != predictivity). registry "
               "v1.16->v1.17 (73923c83, append-only), atlas_v1_18.html (580af247, "
               "746 fields 0/0/0/0), design bfe79091, arm res 24461856/seal "
               "0c975f5b (SMOKE arm payload bitwise identical to FULL; diffs only "
               "mode/res_sha8/seal_sha8), final res fc8be7db/seal f7d8571a."),
    "chain": {"design_sha8": "bfe79091", "res_sha8": "fc8be7db", "seal_sha8": "f7d8571a",
              "arm_res_sha8": "24461856", "arm_seal_sha8": "0c975f5b",
              "registry_v117": "73923c83", "atlas_v1_18": "580af247"},
}
d['measurements'] = ms + [m342]
assert len(d['measurements']) == 342
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3190')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 342
assert json.dumps(d2['measurements'][:341], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #342 appended: post %s' % post)
