# -*- coding: utf-8 -*-
"""Phase 3194 ledger append: measurement #346 (G5-D2), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 345
assert ms[-1]['phase'] == 3193
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m346 = {
    "phase": 3194, "name": "g5d2_ftr26_heldout_upgrade", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d2_ftr26_heldout_upgrade|HOLDOUT_PREDICTIVE_USTAR|3of3|min_med_cos_held_ustar_0.4910_0.5198_0.5221|FTR26_E1_to_E2_held_out|upgrade_log_8|registry_v119_to_v120",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D2", "superseded_by": None,
    "detail": ("G5-D2 FTR-26 held_out upgrade arm (E1->E2 channel, zero GPU; "
               "sealed 3157/3169 collect + 3169/3183/3189/3190/3191/3192/3193 "
               "results). Residual chain = 3193 verbatim (incl. corrected "
               "build_dvu); device anchors drift69/89/90/91/93 all == 0 "
               "bitwise x3, dv_drift == 0 - chain identical across "
               "3189..3194. Dual predictors per model x 4 folds: P1 = u* "
               "(all-876-row SVD top-1, label-free non-leaking direction "
               "summary), P2 = v_hat_held (3191 protocol verbatim control "
               "anchor - med_abs_cos_held / min_med_abs_cos replicated "
               "bitwise: 3191 min values 0.3906/0.4174/0.3717 re-derived "
               "exactly). Gate (pinned pre-observation): median |cos(row, "
               "u*)| >= 0.3 in ALL 4 folds x >= 2/3 models. Observed: "
               "HOLDOUT_PREDICTIVE_USTAR 3/3 - 4b folds 0.5632/0.4910/0.5996/"
               "0.5657 (min 0.4910); 14b 0.5264/0.5303/0.6035/0.5198 (min "
               "0.5198); glm4 0.5757/0.5254/0.5875/0.5221 (min 0.5221); all "
               "12 fold medians exceed the FTR-25 same-protocol control "
               "minima - u* generalizes across held-out classes MORE strongly "
               "than the FTR-25 shared direction. FTR-26 upgraded "
               "E1_repeatable -> E2_held_out (19th E2, upgrade_log #8): "
               "evidence_level change + anchors append role=held_out + "
               "statement append (25 legacy features byte-preserved), "
               "registry v1.19->v1.20 (2aac7e5e, 26 features / E2x19 / E3x3 "
               "/ E1x3 / failures 19 / upgrades 8), atlas_v1_22.html "
               "(b991709b, 771 fields 0/0/0/0). Honest log: SMOKE-1 "
               "NameError NPZ57_KEY dict omitted (SRC p3157 keys present, "
               "dict literal missing) - one-line fix, no observation "
               "contamination; SMOKE/FULL per_model payload bitwise "
               "identical. registry v1.20 2aac7e5e, design a733e0b6, "
               "res f4836e97/seal be9ce95b, final fca2ccbb/90bb8a9c."),
    "chain": {"design_sha8": "a733e0b6", "res_sha8": "f4836e97", "seal_sha8": "be9ce95b",
              "registry_v120": "2aac7e5e", "atlas_v1_22": "b991709b",
              "final_res_sha8": "fca2ccbb", "final_seal_sha8": "90bb8a9c"},
}
d['measurements'] = ms + [m346]
assert len(d['measurements']) == 346
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3194')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 346
assert json.dumps(d2['measurements'][:345], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #346 appended: post %s' % post)
