# -*- coding: utf-8 -*-
"""Phase 3198 ledger append: measurement #350 (G5-D6), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 349
assert ms[-1]['phase'] == 3197
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m350 = {
    "phase": 3198, "name": "g5d6_dline_registration", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d6_dline_registration|REGISTERED|3of3|control_anchor_3197_drift==0_x3|ftr26_statement_append_signed_closing|registry_v1.20_2aac7e5e_to_v1.21_47cf01e0|atlas_v1_23_9488777c_775_fields_clean|evidence_level_E2_held_out_unchanged|upgrade_log_unchanged_8|D_line_five_link_registered",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D6", "superseded_by": None,
    "detail": ("G5-D6 D-line closing registration arm (zero GPU; sealed "
               "registry v1.20 2aac7e5e + sealed p3197 result ee91f8c7 + gap "
               "ledger v1.5). Gate (pinned pre-observation): REGISTERED iff "
               "(a) statement append verified (startswith + anchors prefix + "
               "25 legacy features byte-equal + evidence_level/upgrade_log/"
               "failures/other top-level keys unchanged), (b) atlas field "
               "check clean, (c) control anchor: sealed 3197 signed_weights "
               "v_c x4 + ls_resid (4dp, measured from sealed p3197 result "
               "BEFORE freeze) re-read drift == 0 x3/3. Observed: REGISTERED "
               "3/3 - control drift==0 x3/3 (ls_resid 0.0181/0.0174/0.0181, "
               "v_c x4 all negative, matching pinned ANCH_3197 exactly); "
               "FTR-26 statement append: SIGNED_CLASS_CONTRAST closing + "
               "D-line five-link chain 3193 NOVEL -> 3194 E2_held_out -> "
               "3195 PARTIALLY_DECOMPOSED -> 3196 non-negative cone FAIL -> "
               "3197 signed resolution + methodology note (subspace-"
               "projection / NNLS / signed-LS three-layer readings must be "
               "reported separately; unsigned cos-level readings "
               "systematically overstate class-mean composition); anchors "
               "append role=signed_closing (asserts vc_l1_share 0.9159/"
               "0.8905/0.9244, control_3197_drift 0.0); evidence_level stays "
               "E2_held_out, upgrade_log stays 8, failures F1-F19 unchanged. "
               "Registry v1.20 2aac7e5e -> v1.21 47cf01e0 (26 features, "
               "25 legacy byte-preserved); atlas_v1_23.html 9488777c (775/"
               "775 fields, 0 missing/extra/mismatch/dups). Honest log: "
               "DESIGN re-frozen at patch time (prereg text refined per "
               "MEMO 3197 tail before any observation; SMOKE and FULL share "
               "design 1d6b40ce, payloads identical). design 1d6b40ce, res "
               "a1167bb8/seal 591185ae, final b1e81f1b/fbd85fbb."),
    "chain": {"design_sha8": "1d6b40ce", "res_sha8": "a1167bb8", "seal_sha8": "591185ae",
              "final_res_sha8": "b1e81f1b", "final_seal_sha8": "fbd85fbb",
              "registry_v121": "47cf01e0", "registry_v120": "2aac7e5e",
              "atlas_v1_23": "9488777c"},
}
d['measurements'] = ms + [m350]
assert len(d['measurements']) == 350
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3198')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 350
assert json.dumps(d2['measurements'][:349], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #350 appended: post %s' % post)
