# -*- coding: utf-8 -*-
"""Phase 3197 ledger append: measurement #349 (G5-D5), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 348
assert ms[-1]['phase'] == 3196
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m349 = {
    "phase": 3197, "name": "g5d5_signed_ls_shares", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d5_signed_ls_shares|SIGNED_CLASS_CONTRAST|3of3|ls_resid_0.0181_0.0174_0.0181|vc_l1_share_0.9159_0.8905_0.9244|vc_weights_all_negative_-0.273_to_-0.361|ustar_anti_class_mean_contrast_axis|D_line_qualitative_closing|registry_v120_untouched_report_only",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D5", "superseded_by": None,
    "detail": ("G5-D5 u* SIGNED composition shares arm (unconstrained ridge-"
               "stabilized LS lam=1e-12 over the 22-direction candidate set; "
               "zero GPU; sealed 3169/3183/3189/3190/3191/3192/3193/3194/"
               "3195/3196 results). Residual chain = 3196 verbatim; device "
               "anchors drift69/89/90/91/93/94 all == 0 bitwise x3, "
               "dv_drift == 0; control anchors re-derived drift == 0 "
               "bitwise (3195 QR ratio 0.01806072230473137/"
               "0.017358575654316224/0.01814552246882951; 3196 nnls_resid "
               "4dp 0.8184/0.8125/0.8236). Gate (pinned): "
               "SIGNED_CLASS_CONTRAST iff LS resid < 0.05 AND |v_c x4| l1 "
               "share >= 0.5 x3/3. Observed: SIGNED_CLASS_CONTRAST 3/3 - "
               "ls_resid 0.0181/0.0174/0.0181 (matches QR ratio: near-"
               "complete signed fit); vc_l1_share 0.9159/0.8905/0.9244; "
               "ALL FOUR v_c weights NEGATIVE in every model (4b: v_c[8] "
               "-0.336, v_c[6] -0.320, v_c[9] -0.303, v_c[7] -0.273; 14b: "
               "-0.355/-0.289/-0.281/-0.276; glm4: -0.361/-0.320/-0.314/"
               "-0.302); next-largest |w| <= 0.024; n_negative 18/13/13 of "
               "22. D-LINE QUALITATIVE CLOSING: u* ~= -(w8*v_c[8]+w6*"
               "v_c[6]+w9*v_c[9]+w7*v_c[7]) - an ANTI-CLASS-MEAN CONTRAST "
               "AXIS (deviation from all four OOV class means "
               "simultaneously). D line five-link chain now closed: 3193 "
               "NOVEL (single-direction level, restricted gate set) -> 3194 "
               "E2_held_out (cross-class predictability) -> 3195 "
               "PARTIALLY_DECOMPOSED (subspace level, v_c included) -> 3196 "
               "non-negative cone FAIL (nnls_resid 0.82 >> ratio 0.018) -> "
               "3197 signed resolution (negative class-mean sum). "
               "Consistent with origin: u* = SVD top-1 of OOV-specific "
               "residuals (must point AWAY from every class mean). "
               "REPORT-ONLY: registry v1.20 byte-UNTOUCHED (recheck "
               "2aac7e5e), FTR-26 stays E2_held_out. Honest log: SMOKE-1 "
               "sha-anchor placeholder f086e6a3 -> measured f3f86f47 "
               "(p3196_result, measured probe before SMOKE - no "
               "observation contamination); SMOKE/FULL payload bitwise "
               "identical. design c3a64857, res dd2af224/seal 4311d796, "
               "final f8a39985/2d361f0d."),
    "chain": {"design_sha8": "c3a64857", "res_sha8": "dd2af224", "seal_sha8": "4311d796",
              "final_res_sha8": "f8a39985", "final_seal_sha8": "2d361f0d",
              "registry_v120": "2aac7e5e"},
}
d['measurements'] = ms + [m349]
assert len(d['measurements']) == 349
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3197')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 349
assert json.dumps(d2['measurements'][:348], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #349 appended: post %s' % post)
