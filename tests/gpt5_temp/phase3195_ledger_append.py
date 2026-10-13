# -*- coding: utf-8 -*-
"""Phase 3195 ledger append: measurement #347 (G5-D3), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 346
assert ms[-1]['phase'] == 3194
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m347 = {
    "phase": 3195, "name": "g5d3_ustar_attribution", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d3_ustar_attribution|PARTIALLY_DECOMPOSED|3of3|residual_ratio_0.0181_0.0174_0.0181|carriers_vc_x4_plus_kout_pc|registry_v120_untouched_report_only",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D3", "superseded_by": None,
    "detail": ("G5-D3 u* content attribution deepening (zero GPU; sealed "
               "3157/3169 collect + 3169/3183/3189/3190/3191/3192/3193/3194 "
               "results). Residual chain = 3194 verbatim; device anchors "
               "drift69/89/90/91/93/94 all == 0 bitwise x3, dv_drift == 0. "
               "Control anchor: 3193 gate set re-derived bitwise (gate_max "
               "0.2130/0.3303/0.2457, |cos(u*,dvu)| 0.1064/0.0405/0.0240, "
               "cos_vs x6, cos_useen, cos_vc_report x4 0.72-0.85). Candidate "
               "span C = 22 directions {dvu, v_s x6, u_seen, v_c x4, KOUT "
               "top-10 PCs of mean-centered raw 876-row Y}; metric = "
               "residual ratio ||u* - P_C u*||/||u*|| (QR-orthonormalized). "
               "Gate (pinned): PARTIALLY_DECOMPOSED iff ratio < 0.5 x3/3. "
               "Observed: PARTIALLY_DECOMPOSED 3/3 - ratio 0.0181/0.0174/"
               "0.0181 (proj 0.9998 x3): u* lies almost entirely in the "
               "candidate span. Top carriers: v_c x4 dominate (|cos| "
               "0.724-0.852 across models), KOUT pc0/pc2 next (0.38-0.43); "
               "dvu/v_s/u_seen single-direction contributions stay low - "
               "3193 NOVEL (no single known-structure direction in the "
               "restricted gate set) and 3195 decomposable (subspace level, "
               "v_c included) are consistent: u* is a class-mean-dominated "
               "mixture of the OOV residual geometry, NOT a new primitive "
               "direction. REPORT-ONLY: registry v1.20 byte-UNTOUCHED (file "
               "sha recheck 2aac7e5e), no atlas HTML, FTR-26 stays E2_held_"
               "out; attribution note recorded here + MEMO. Honest log: "
               "SMOKE-1 candidate-count assert 17 -> 22 (prereg arithmetic "
               "slip 1+6+1+4+10=22; design refrozen c23f7ada); "
               "SMOKE/FULL payload bitwise identical. design c23f7ada, "
               "res 21aee353/seal f9fe19e0, final b527537a/551e8b40."),
    "chain": {"design_sha8": "c23f7ada", "res_sha8": "21aee353", "seal_sha8": "f9fe19e0",
              "final_res_sha8": "b527537a", "final_seal_sha8": "551e8b40",
              "registry_v120": "2aac7e5e"},
}
d['measurements'] = ms + [m347]
assert len(d['measurements']) == 347
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3195')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 347
assert json.dumps(d2['measurements'][:346], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #347 appended: post %s' % post)
