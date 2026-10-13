# -*- coding: utf-8 -*-
"""Phase 3202 ledger append: measurement #354 (G5-F1 FTR-04 recheck), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 353
assert ms[-1]['phase'] == 3201
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m354 = {
    "phase": 3202, "name": "g5f1_ftr04_recheck", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5f1_ftr04_recheck|REGISTERED|R1_4dp_drift0_x3|R2_anchor_bitwise|R3_loeo_gap_0.1181_ge_0.10|margin_ratio_1.9661_ge_1.5|geometry_band_ok|ftr04_weak_specificity=readout_inconsistency_not_geometry|scope_limits_append|registry_v1.24_0d74b628_to_v1.25_3bb33a24|atlas_v1_27_739faeb7_784_fields_clean",
    "evidence_level": "E1_repeatable",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-F1", "superseded_by": None,
    "detail": ("G5-F1 FTR-04 glm4 weak-specificity unit-level decomposition "
               "recheck arm (registration arm on sealed 3181 scalars; zero GPU; "
               "default prereg arm (a) confirmed viable by pre-freeze probe). "
               "R1: sealed 3181 per_model scalars re-read vs pinned 4dp constants "
               "(loeo 0.6019/0.5981/0.4800, margin 2.6614/1.4164/5.2326, kappa "
               "48118411.9811/46016538.1628/44953766.4873, pr 8.7763/8.9016/"
               "8.9220, proj 0.0763/0.0328/0.0294, e90 8) drift == 0 x3. R2: "
               "FTR-04 p3181 anchor asserts (kappa/margin/proj glm4 + res/seal "
               "sha) bitwise-consistent with sealed scalars. R3 decomposition "
               "bars (pinned post-probe, honest): loeo_gap = min(qwen loeo) - "
               "glm4 loeo = 0.1181 >= 0.10; margin_ratio = glm4 margin / max "
               "qwen margin = 1.9661 >= 1.5; geometry band healthy (kappa ratio "
               "0.934, pr ratio 1.017, e90 equal). Interpretation: glm4-9b weak "
               "specificity = READOUT INCONSISTENCY (highest class-pair margin "
               "with LOWEST leave-one-entity-out readout accuracy => margin "
               "distribution scattered across class pairs), NOT a geometry "
               "deficit - unit-level support for the 3181 noise_level verdict "
               "(weak specificity not attributable to geometry or readout "
               "mechanism absence; E level unchanged E1_repeatable, no upgrade). "
               "Registration: FTR-04 scope_limits append ONLY; 25 other features "
               "+ upgrade_log + failures F1-F19 byte-preserved. Registry v1.24 "
               "0d74b628 -> v1.25 3bb33a24 (append-only); atlas_v1_27.html "
               "739faeb7 (784/784 fields, 0/0/0/0). Control anchors: 3181 "
               "content double lock (res 1e95a19d / seal 462fd221) + 3178/3183 "
               "double locks + 3157 tc + 3183 dv bitwise x3. Honest log: "
               "generation pre-FULL fixes - 3181 result has top-level per_model "
               "(no arm wrapper) and flatten meta.p3181_result key added after "
               "HTML extra=1; both fixed in create script + generated script. "
               "design 822996d4, res af81e9f4/seal 480bd271, final f75b3a6d/"
               "dbcaf643."),
    "chain": {"design_sha8": "822996d4", "res_sha8": "af81e9f4", "seal_sha8": "480bd271",
              "final_res_sha8": "f75b3a6d", "final_seal_sha8": "dbcaf643",
              "registry_v125": "3bb33a24", "registry_v124": "0d74b628",
              "atlas_v1_27": "739faeb7"},
}
d['measurements'] = ms + [m354]
assert len(d['measurements']) == 354
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3202')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 354
assert json.dumps(d2['measurements'][:353], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #354 appended: post %s' % post)
