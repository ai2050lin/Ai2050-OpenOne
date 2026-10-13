# -*- coding: utf-8 -*-
"""Phase 3193 ledger append: measurement #345 (G5-D1), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 344
assert ms[-1]['phase'] == 3192
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m345 = {
    "phase": 3193, "name": "g5d1_ustar_decomposition", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5d1_ustar_decomposition|novel_x2of3|gate_max_0.2130_0.3303_0.2457|cos_dvu_0.1064_0.0405_0.0240|FTR26_novel_direction_candidate|registry_v119_features_26",
    "evidence_level": "E1_repeatable",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-D1", "superseded_by": None,
    "detail": ("G5-D1 u* subspace decomposition (D line opening, zero GPU; "
               "sealed 3157/3169 collect + 3169/3183/3189/3190/3191/3192 "
               "results). Residual chain = 3192 verbatim; device anchors: "
               "3189/3190/3191 sealed scalars drift == 0 bitwise x3, 3157 tc "
               "anchor + 3183 dv replication (cos_rl/norm_dv), 3169 per-seed E "
               "0 drift - chain identical across 3189..3193; report-only "
               "|cos(u*,v_c)| bitwise == 3190 values. PREREG DOWNGRADE (freeze-"
               "pinned, honest): 3165/3166 sealed results hold scalars/singular "
               "values only, no direction vectors; gate set downgraded to "
               "{dvu (3157/3183 verbatim), v_s x6 (seen-class mean residual "
               "directions from all 738 seen rows), u_seen (seen residual "
               "top-1 SVD)}; report-only v_c x4. Gate (pinned): DECOMPOSED iff "
               "max |cos| >= 0.5 x>=2/3; NOVEL iff max < 0.3 x>=2/3. Observed: "
               "NOVEL 2/3 (4b 0.2130, glm4 0.2457 < 0.3; 14b 0.3303 "
               "intermediate); |cos(u*,dvu)| = 0.1064/0.0405/0.0240 (u* is NOT "
               "the relation-offset direction - consistent with 3188 DECOUPLED "
               "at direction level); max|cos(u*,v_s)| 0.21/0.22/0.20; "
               "|cos(u*,u_seen)| 0.15/0.33/0.25. NOVEL branch: FTR-26 "
               "u*_novel_direction candidate registered (family=context "
               "E1_repeatable, features 25->26 append-only, 25 legacy features "
               "byte-preserved), registry v1.18->v1.19 (4d64a153), "
               "atlas_v1_21.html (5a6d219f, 764 fields 0/0/0/0). Honest log: "
               "SMOKE crashes x3 (ALL_SEEN_ROWS filter missed entity constraint "
               "i<41; 3183 path dir name g5b5_lowrank_check -> g5b8_dv_lowrank; "
               "3183 per_model key glm4 -> glm4-9b) + isa/hasa row grouping fix "
               "(rl==0 at i%4 in {0,1}, not first 32 rows; caught by 3183 dv "
               "anchor) + FULL-1 SHA_ANCHOR missing p3192_result key (design "
               "freeze refrozen bd6dbbc6 -> 7af34fe6, refreeze honest); "
               "SMOKE/FULL per_model payload bitwise identical. registry v1.19 "
               "4d64a153, design 65609e8b, arm res 5465dd47/seal ce1d8c8d, "
               "final res e75c6e34/seal 9164b1ec."),
    "chain": {"design_sha8": "7af34fe6", "res_sha8": "e75c6e34", "seal_sha8": "9164b1ec",
              "arm_res_sha8": "5465dd47", "arm_seal_sha8": "ce1d8c8d",
              "registry_v119": "4d64a153", "atlas_v1_21": "5a6d219f"},
}
d['measurements'] = ms + [m345]
assert len(d['measurements']) == 345
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3193')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 345
assert json.dumps(d2['measurements'][:344], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #345 appended: post %s' % post)
