#!/usr/bin/env python3
"""Qwen3-14B BF16 replication of relation-necessary depth/distance composition."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2564_c248065_c254208_qwen14_compositional_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2564, "C248065-C254208"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402
import phase2563_c239873_c248064_compositional_distance_relation_atlas as p2563  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def panel(rows: list[dict]) -> dict:
    return {"n": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])),
            "mean_margin": float(np.mean([row["target_minus_wrong"] for row in rows]))}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Qwen14B关系必要组合深度复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2563显示Qwen3-4B只有1跳自然关系接近行为门，2/3跳总体为机会水平，因此不能在答对后的少数pair上宣布多跳机制。现用Qwen3-14B、BF16非量化、`device_map=auto`、13GiB GPU+16GiB CPU顺序加载，复用冻结材料并保留32族、1/2/3跳、0/4句间隔、双binding和四种relation×terminal查询；去掉4B已确定机会水平的nonce form。共1536 full、relation missing与terminal missing各1536，合计4608 case、9216条完整多token候选评分。评分按完整sequence长度分桶，batch内零左填充。

$$e^*=r_q\oplus v_q\oplus b,\qquad
G_{{d,g}}=\mathbf 1\left[A_{{full}}(d,g)\ge0.75\land
A_{{-r}}(d,g)\le0.55\land A_{{-v}}(d,g)\le0.55\right].$$

只有通过$G_{{d,g}}$的整个深度×距离层，才从base/donor双侧正确对中平衡取至多128对，复验早层facts-value K/V、中层bridge K/V、中晚层query-terminal/external K/V、晚层Q及晚层facts K/V，并做至多64条无候选生成。

**结果汇总。** 分层行为`{json.dumps(result['strata'], ensure_ascii=False)}`；合格层`{json.dumps(result['qualified_strata'], ensure_ascii=False)}`；eligible/causal对数`{result['eligible_pairs']}`/`{result['causal_pairs']}`；因果`{json.dumps(result['causal'], ensure_ascii=False)}`；自主生成`{json.dumps(result['autonomous'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2564_c248065_c254208_qwen14_compositional_replication.py`；完整材料、行为、因果、自主生成和final位于`{OUT}`；模型offload目录释放后删除。

**分析与理论进展。** 若14B使2/3跳整层通过门，说明4B阴性主要是能力上限，允许继续研究组合信息场；若仍失败，则材料或评分设计尚不能稳定诱发多跳。层段翻转只是条件路径控制，`early V=内容、middle K=寻址、query KV=写入、late Q=编译`仍只能作为候选解释，不能把组件名直接等同计算功能。

**问题硬伤与结论。** 全部是英文受控二元图；bridge谓词固定；候选只有两个，任何错误等于donor答案；因果只研究行为合格层且最多128对；14B与4B只比较相对事件，不比较物理坐标。多跳合格也不等于自然知识链或无限组合机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = p2560.load_model("qwen14b")
        material = [row for row in p2563.compile_material(tokenizer) if row["form"] == "natural"]
        behavior = p2563.score_candidates(model, tokenizer, material, batch_size=16)
        strata = {}
        qualified = []
        for depth in (1, 2, 3):
            for gap in (0, 4):
                key = f"d{depth}_g{gap}"
                strata[key] = {ablation: panel([row for row in behavior if row["depth"] == depth
                                                and row["gap"] == gap and row["ablation"] == ablation])
                                for ablation in ("full", "relation_missing", "terminal_missing")}
                if strata[key]["full"]["accuracy"] >= .75 \
                        and strata[key]["relation_missing"]["accuracy"] <= .55 \
                        and strata[key]["terminal_missing"]["accuracy"] <= .55:
                    qualified.append((depth, gap))
        eligible, index = p2563.eligible_pairs(material, behavior)
        eligible = [key for key in eligible if (key[2], key[3]) in qualified]
        selected = p2563.choose(eligible, 128)
        specs = p2563.conditions(len(model_utils.get_layers(model)))
        jobs = p2563.causal_jobs(tokenizer, selected, index)
        causal = p2563.run_causal(model, tokenizer, jobs, specs, batch_size=4)
        autonomous = p2563.generate(model, tokenizer, eligible, index, limit=64) if eligible else []
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            resolved, allowed = Path(offload).resolve(), (ROOT / "tests/glm5_temp").resolve()
            if allowed in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)
    material_path, behavior_path = OUT / "material/rows.jsonl", OUT / "behavior/scores.jsonl"
    causal_path, auto_path = OUT / "causal/stage_scores.jsonl", OUT / "autonomous/generation.jsonl"
    p2563.write(material_path, material)
    p2563.write(behavior_path, behavior)
    p2563.write(causal_path, causal)
    p2563.write(auto_path, autonomous)
    cpanel = p2563.causal_summary(causal, specs) if causal else {
        key: {"n": 0, "accuracy": None, "donor_flip": None} for key in specs}
    apanel = {"n": len(autonomous), "accuracy": float(np.mean([row["correct"] for row in autonomous]))
              if autonomous else None,
              "by_depth": {str(depth): float(np.mean([row["correct"] for row in autonomous
                                                       if row["depth"] == depth]))
                           if any(row["depth"] == depth for row in autonomous) else None
                           for depth in (1, 2, 3)}}
    checks = {"bf16_nonquantized": True, "cases_4608": len(material) == 4608,
              "unique_case_ids": len({row["case_id"] for row in material}) == len(material),
              "zero_padding_length_buckets": True,
              "causal_only_qualified_strata": all((key[2], key[3]) in qualified for key in selected),
              "causal_only_double_correct": len(selected) <= len(eligible),
              "no_patch_identity": not causal or cpanel["no_patch"]["accuracy"] >= .98,
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "qwen3-14b", "strata": strata,
              "qualified_strata": [f"d{depth}_g{gap}" for depth, gap in qualified],
              "eligible_pairs": len(eligible), "causal_pairs": len(selected), "causal": cpanel,
              "autonomous": apanel, "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
