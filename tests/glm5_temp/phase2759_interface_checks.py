"""Source-interface and saved-summary checks; no model forward or GPU allocation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_six_puzzles_audit_20260923"
R = lambda n: json.loads((OUT / f"source_result_{n}.json").read_text(encoding="utf-8"))
paths = [
    ".venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py",
    ".venv/Lib/site-packages/transformers/utils/output_capturing.py",
    "tests/glm5/phase3105_omega_p103_incontext_truth_consistency.py",
    "tests/glm5/phase3107_omega_p105_writehead_mapping.py",
    "tests/glm5/phase3116_omega_p114_full_mlp_sweep_decouple.py",
    "tests/glm5/phase3118_omega_p116_autoregressive_margin_trajectory.py",
    "tests/glm5/phase3122_omega_p120_write_content_readout_sentence_causal_dist_recon.py",
    "tests/glm5/phase3123_omega_p121_dirfit_anchor_l35loc_syntax_trace.py",
]
evidence = []
for p in paths:
    f = ROOT / p
    t = f.read_text(encoding="utf-8")
    excerpts = []
    needles = ("hidden_states = self.norm", "logits = self.lm_head",
               "def capture_outputs", "collected_outputs[key].append(outputs.last_hidden_state)",
               "norm_mod(hs[NL]", "out.hidden_states[NL]", "h_all = norm(hs)",
               "w5n =", "wdnn =", "cos_5 =", "sd = X[tr].std", "Z = (X - mu)",
               "for L in range(20", "s1'][k1", "sub = sub[:L]")
    for i,line in enumerate(t.splitlines(), 1):
        if any(q in line for q in needles):
            excerpts.append({"line":i, "text":line.strip()})
    evidence.append({"path":p,"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"excerpts":excerpts})

curves = R(3123)["part_c"]["curves"]
thresholds = {"P":.05,"A1":.025}
emergence = {}
for d in ("P","A1"):
    emergence[d] = {}
    for name in ("syn","cont"):
        key = f"E_{name}_{d}"
        arr = np.array(curves[key])
        ids = np.flatnonzero(arr <= -thresholds[d])
        restricted = ids[ids >= 20]
        emergence[d][name] = {"first_negative_threshold_any_layer":int(ids[0]) if len(ids) else None,
                              "first_negative_threshold_from_20":int(restricted[0]) if len(restricted) else None,
                              "interpretation":"Threshold crossing of a logit-lens contrast, not a causal gate localization."}

effect = R(3122)["part_b"]["effects"]
comparisons = {}
for d,v in effect.items():
    dm = v["D_mean"]
    comparisons[d] = {"coherent_minus_original":dm["s1"],"shuffle_minus_original":dm["s2"],
                      "dots_minus_original":dm["s3"],"coherent_minus_dots":v["E_cont"],
                      "coherent_minus_shuffle":v["E_syn"]}

result = {
    "source_evidence":evidence,
    "double_norm": {
        "status":"Confirmed source-interface mismatch with the installed Qwen3/Transformers implementation; original GPU jobs not rerun.",
        "native":"hidden_states[-1] is final-normalized; native logits = lm_head(hidden_states[-1])",
        "historical":"norm(hidden_states[-1]) introduces an extra normalization before the research readout",
        "impact":"The captured final slot and many reported margins are custom twice-normalized observables, not native residuals/logits. Native out.logits generation remains a separate valid observation.",
        "limits":"Historical runtime versions were not independently reconstructed. Native-logit numerical effects require forward checks. GPU inspection showed 15798/16303 MiB used; no competing model was loaded or process interrupted.",
    },
    "probe_coordinate_mismatch":{
        "code":"fit w in Z=(X-mu)/sd, then compare directly to raw unembedding direction",
        "correct_mapping":"w_raw = w_standardized / sd; compare at the same normalization boundary",
        "limits":"No corrected angle estimated here; the old angle alone does not establish raw-space orthogonality."},
    "layer_thresholds":emergence,
    "sentence_comparisons":comparisons,
    "sentence_material_limits":R(3122)["part_b"]["n_pad_info"],
}
(OUT/"interface_checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"layer_thresholds":emergence,"sentence_comparisons":comparisons},ensure_ascii=False,indent=2))
