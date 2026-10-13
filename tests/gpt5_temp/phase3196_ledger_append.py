# -*- coding: utf-8 -*-
"""Phase 3196 ledger append: measurement #348 (G5-D4), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 347
assert ms[-1]['phase'] == 3195
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m348 = {
    "phase": 3196, "name": "g5d4_ustar_nnls_shares", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d4_ustar_nnls_shares|MIXED_0of3|vc_share_0.0000_x3|known_share_0.1999_0.3998_0.1139|nnls_resid_0.82_gt_ratio_0.018_nonneg_cone_fail|signed_mixture_required|registry_v120_untouched_report_only",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D4", "superseded_by": None,
    "detail": ("G5-D4 u* non-negative composition shares arm (NNLS via "
               "scipy.optimize.nnls over the 22-direction candidate set of "
               "G5-D3 {dvu, v_s x6, u_seen, v_c x4, KOUT top-10 PCs}; zero "
               "GPU; sealed 3169/3183/3189/3190/3191/3192/3193/3194/3195 "
               "results). Residual chain = 3195 verbatim; device anchors "
               "drift69/89/90/91/93/94 all == 0 bitwise x3, dv_drift == 0; "
               "3195 QR-projection ratio control anchor re-derived drift == "
               "0 bitwise (0.01806072230473137/0.017358575654316224/"
               "0.01814552246882951). Gate (pinned): CLASS_MEAN_DOMINATED "
               "iff v_c x4 share sum >= 0.8 AND known-struct share sum < 0.1 "
               "x3/3. Observed: MIXED 0/3 - vc_share = 0.0000 in ALL 3 "
               "models (NNLS assigns zero weight to v_c); known_share "
               "0.1999/0.3998/0.1139; NNLS share spectrum led by kout_pc "
               "(0.216-0.343 per model) + u_seen (0.074-0.343) + small "
               "dvu/v_s. KEY NEGATIVE RESULT: nnls_resid 0.8184/0.8125/"
               "0.8236 >> QR-projection ratio 0.018 - although u* lies "
               "almost entirely in the candidate SPAN (3195 PARTIALLY_"
               "DECOMPOSED, proj 0.9998), NO non-negative coefficient "
               "combination of the candidates fits u* (cone membership "
               "FAILS); the cos-level v_c dominance (0.72-0.85) can only "
               "hold via NEGATIVE-coefficient cancellation. Interpretation: "
               "u* is a SIGNED contrast mixture - consistent with its "
               "origin as OOV-specific residual direction (must contain "
               "'subtract the seen baseline' negative components); cos-"
               "level reads without sign constraint overstate class-mean "
               "composition. Caveat registered: NNLS over collinear "
               "candidates is non-unique, but resid 0.82 is not a "
               "non-uniqueness artifact (no non-negative combo fits at "
               "all). REPORT-ONLY: registry v1.20 byte-UNTOUCHED (recheck "
               "2aac7e5e), FTR-26 stays E2_held_out. Honest log: SMOKE-1 "
               "leftover return (decomposed -> dominated, mechanical); "
               "SMOKE/FULL payload bitwise identical. design 94bbdd58, "
               "res e69ab1e7/seal c0cdd97f, final 7313de4d/f57cc8b7."),
    "chain": {"design_sha8": "6723c371", "res_sha8": "e69ab1e7", "seal_sha8": "c0cdd97f",
              "final_res_sha8": "7313de4d", "final_seal_sha8": "f57cc8b7",
              "registry_v120": "2aac7e5e"},
}
d['measurements'] = ms + [m348]
assert len(d['measurements']) == 348
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3196')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 348
assert json.dumps(d2['measurements'][:347], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #348 appended: post %s' % post)
