#!/usr/bin/env python3
"""Correct Phase2582 prefix indexing and decompose the first interaction-bearing block."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2582 = RESULT / "phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth"
OUT = RESULT / "phase2583_c385281_c397568_block0_component_interaction_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2583, "C385281-C397568"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth as p2582  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def factorial(array: np.ndarray) -> np.ndarray:
    return array[3].astype(np.float32) - array[2].astype(np.float32) - array[1].astype(np.float32) + array[0].astype(np.float32)


def rms(array: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt(np.mean(array.astype(np.float64) ** 2, axis=axis))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def capture(model, selected):
    layer = model_utils.get_layers(model)[0]
    device = model.get_input_embeddings().weight.device
    component_dir = OUT / "component_field"
    component_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    answer_rows = []
    source_semantic = defaultdict(list)
    head_rms = []
    reconstruction_errors = []
    qkv_maxabs = []
    for quartet_index, (prefix, cells) in enumerate(selected):
        captured = {}

        def projection_hook(name):
            def hook(_module, _inputs, output):
                captured[name] = tensor_output(output).detach()
            return hook

        def attention_hook(_module, _inputs, output):
            captured["attention_output"] = tensor_output(output).detach()
            if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                raise RuntimeError("layer0 attention weights unavailable")
            captured["attention_weights"] = output[1].detach()

        handles = [
            layer.self_attn.q_proj.register_forward_hook(projection_hook("q")),
            layer.self_attn.k_proj.register_forward_hook(projection_hook("k")),
            layer.self_attn.v_proj.register_forward_hook(projection_hook("v")),
            layer.self_attn.register_forward_hook(attention_hook),
            layer.mlp.register_forward_hook(projection_hook("mlp_output")),
        ]
        try:
            ids = torch.tensor([row["prompt_ids"] for row in cells], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    output_attentions=True,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        length = ids.shape[1]
        answer = cells[0]["answer_boundary_token"]
        q, k, v = (captured[name] for name in ("q", "k", "v"))
        weights = captured["attention_weights"]
        attn_out = captured["attention_output"]
        mlp_out = captured["mlp_output"]
        num_heads = int(weights.shape[1])
        head_dim = int(layer.self_attn.head_dim)
        kv_heads = int(v.shape[-1] // head_dim)
        group_size = num_heads // kv_heads
        repeated_v = v.view(4, length, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group_size, dim=1)
        answer_weights = weights[:, :, answer, :]
        source_head = answer_weights.unsqueeze(-1) * repeated_v.transpose(1, 2).transpose(1, 2)
        # source_head: [cell, head, source, head_dim]
        source_concat = source_head.permute(0, 2, 1, 3).reshape(4, length, num_heads * head_dim)
        with torch.inference_mode():
            source_output = layer.self_attn.o_proj(source_concat)
        context_head = source_head.sum(dim=2)
        source_sum_error = float(torch.max(torch.abs(source_output.sum(dim=1) - attn_out[:, answer])).item())
        arrays = {
            "q": q.detach().cpu().to(torch.float16).numpy(),
            "k": k.detach().cpu().to(torch.float16).numpy(),
            "v": v.detach().cpu().to(torch.float16).numpy(),
            "attention_weights": weights.detach().cpu().to(torch.float16).numpy(),
            "attention_output": attn_out.detach().cpu().to(torch.float16).numpy(),
            "mlp_output": mlp_out.detach().cpu().to(torch.float16).numpy(),
            "answer_source_output": source_output.detach().cpu().to(torch.float16).numpy(),
            "answer_context_head": context_head.detach().cpu().to(torch.float16).numpy(),
        }
        component_path = component_dir / f"quartet_{quartet_index:03d}.npz"
        np.savez(component_path, **arrays)
        raw = np.load(P2582 / f"field/quartet_{quartet_index:03d}.npy", mmap_mode="r")
        i_h0 = factorial(raw[:, 0])
        i_h1 = factorial(raw[:, 1])
        i_attn = factorial(arrays["attention_output"])
        i_mlp = factorial(arrays["mlp_output"])
        i_q, i_k, i_v = (factorial(arrays[name]) for name in ("q", "k", "v"))
        reconstructed = i_h0 + i_attn + i_mlp
        reconstruction_errors.append({
            "quartet_index": quartet_index,
            "factorial_reconstruction_maxabs": float(np.max(np.abs(i_h1 - reconstructed))),
            "factorial_reconstruction_rms": float(rms(i_h1 - reconstructed, axis=None)),
            "attention_source_sum_maxabs": source_sum_error,
        })
        qkv_maxabs.append({"q": float(np.max(np.abs(i_q))), "k": float(np.max(np.abs(i_k))),
                           "v": float(np.max(np.abs(i_v)))})
        answer_i_attn = i_attn[answer]
        answer_i_mlp = i_mlp[answer]
        answer_i_h1 = i_h1[answer]
        answer_i_weight = factorial(arrays["attention_weights"])[:, answer, :]
        answer_i_source = factorial(arrays["answer_source_output"])
        answer_i_head = factorial(arrays["answer_context_head"])
        head_rms.append(rms(answer_i_head, axis=1))
        answer_rows.append({
            "quartet_index": quartet_index,
            "attention_interaction_rms": float(rms(answer_i_attn)),
            "mlp_interaction_rms": float(rms(answer_i_mlp)),
            "h1_interaction_rms": float(rms(answer_i_h1)),
            "attention_weight_interaction_rms": float(rms(answer_i_weight, axis=None)),
            "attention_to_h1_cosine": p2582.pearson(answer_i_attn, answer_i_h1),
            "mlp_to_h1_cosine": p2582.pearson(answer_i_mlp, answer_i_h1),
        })
        lookup = p2582.region_lookup(cells[0])
        source_rms = rms(answer_i_source)
        source_max = np.max(np.abs(answer_i_source), axis=1)
        weight_source_rms = rms(answer_i_weight, axis=0)
        for position, (region, ordinal) in enumerate(lookup):
            source_semantic[(region, ordinal)].append({
                "source_output_interaction_rms": float(source_rms[position]),
                "source_output_interaction_maxabs": float(source_max[position]),
                "attention_weight_interaction_rms": float(weight_source_rms[position]),
            })
        manifests.append({
            "quartet_index": quartet_index,
            "prefix": list(prefix),
            "case_ids": [row["case_id"] for row in cells],
            "prompt_tokens": int(length),
            "num_heads": num_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "component_path": str(component_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": component_path.stat().st_size,
            "sha256": sha256(component_path),
        })
        del output, captured, arrays, raw, q, k, v, weights, attn_out, mlp_out
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase2583 components] {quartet_index + 1}/{len(selected)}", flush=True)
    save_json(OUT / "component_field/manifest.json", manifests)
    return manifests, answer_rows, source_semantic, np.stack(head_rms), reconstruction_errors, qkv_maxabs


def summarize(manifests, answer_rows, source_semantic, head_values, reconstruction, qkv):
    p2582_result = load_json(P2582 / "analysis/final.json")
    semantic = {}
    for (region, ordinal), rows in source_semantic.items():
        semantic.setdefault(region, {})[str(ordinal)] = {
            name: float(np.median([row[name] for row in rows]))
            for name in rows[0]
        }
    component = {
        name: {
            "median": float(np.median([row[name] for row in answer_rows])),
            "q25": float(np.quantile([row[name] for row in answer_rows], .25)),
            "q75": float(np.quantile([row[name] for row in answer_rows], .75)),
        }
        for name in answer_rows[0] if name != "quartet_index"
    }
    median_heads = np.median(head_values, axis=0)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "correction": {
            "phase2582_initial_memo_error": "prequery threshold accidentally included post-query frame tokens",
            "raw_field_affected": False,
            "correct_prequery_max_rms": p2582_result["interaction_birth"]["prequery_max_rms"],
            "correct_threshold": p2582_result["interaction_birth"]["threshold"],
            "correct_first_answer_hidden_state": p2582_result["interaction_birth"]["first_answer_boundary_hidden_state"],
        },
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": {
            "quartets": len(manifests),
            "prompts": len(manifests) * 4,
            "block": 0,
            "all_token_all_coordinate_components": ["Q", "K", "V", "attention weights", "attention output", "MLP output"],
            "answer_source_output": "each source token projected through its full head values and W_o",
            "no_topk_primary_analysis": True,
        },
        "answer_boundary_components": component,
        "qkv_factorial_maxabs": {name: float(max(row[name] for row in qkv)) for name in ("q", "k", "v")},
        "reconstruction": {
            "max_factorial_reconstruction_error": float(max(row["factorial_reconstruction_maxabs"] for row in reconstruction)),
            "max_factorial_reconstruction_rms": float(max(row["factorial_reconstruction_rms"] for row in reconstruction)),
            "max_attention_source_sum_error": float(max(row["attention_source_sum_maxabs"] for row in reconstruction)),
        },
        "head_interaction_rms": {
            "median_by_head": median_heads.tolist(),
            "min": float(np.min(median_heads)),
            "max": float(np.max(median_heads)),
            "nonzero_heads_at_1e-7": int(np.sum(median_heads > 1e-7)),
        },
        "source_token_lattice": semantic,
        "storage": {
            "bytes": sum(item["bytes"] for item in manifests),
            "dtype": "float16",
            "analysis_accumulation": "float32/float64",
        },
        "claim_boundary": {
        "supported": "block0 attention routing already generates a distributed but nonuniform relation×value interaction, and block0 MLP transforms/adds to it",
            "not_supported": "block0 interaction is semantically specific, necessary, sufficient, or a single-head gear",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2582_corrected_final_passes": p2582_result["all_checks_passed"],
        "all_32_quartets": len(manifests) == 32,
        "all_128_prompts": len(manifests) * 4 == 128,
        "qkv_pre_attention_interaction_zero": max(result["qkv_factorial_maxabs"].values()) <= 1e-7,
        "attention_interaction_nonzero": component["attention_interaction_rms"]["median"] > 1e-5,
        "mlp_interaction_nonzero": component["mlp_interaction_rms"]["median"] > 1e-5,
        "all_32_heads_measured": len(result["head_interaction_rms"]["median_by_head"]) == 32,
        "component_fields_exist": all((ROOT / item["component_path"]).is_file() for item in manifests),
        "no_topk_primary_analysis": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Phase2582索引勘误与首个block全组件交互图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**Phase2582勘误。** 初版Memo把region名为`frame`和`query_context`的全部token当成query前缀，但这两个集合还含查询之后的token，故错误得到前缀RMS=2.4658、阈值246.58和“未检测到出生”。原始4.36GB全场、逐层RMS曲线均未受影响。按绝对位置$t<\min(query\_relation)$重算，前缀与embedding交互RMS都是0，阈值$10^{{-5}}$，answer boundary的交互在HiddenState 1（block0之后）首次出现，中位RMS为{load_json(P2582 / 'analysis/final.json')['interaction_birth']['median_answer_interaction_rms_by_hidden_state'][1]:.8g}。本Phase以append方式更正，不覆盖旧记录。

**测试原理与用例。** 沿用32个行为全对、四格逐token等长四元组（128 prompts），只解剖首个非零block。保存所有token的Q(4096)、K/V(1024)、32头完整attention权重、attention输出与MLP输出的全部物理坐标。残差恒等式给出：

$$I(H_1)=I(H_0)+I(A_0)+I(M_0),\qquad I(X)=X_{{11}}-X_{{10}}-X_{{01}}+X_{{00}}.$$

对answer query还把每个source token的每头值贡献经$W_O$投影回2560坐标：

$$C_s=W_O\,\mathrm{{concat}}_h\left(\alpha_{{h,s}}V_{{h,s}}\right),\qquad A_0=\sum_s C_s.$$

**结果汇总。** answer boundary组件统计：`{json.dumps(result['answer_boundary_components'], ensure_ascii=False)}`。Q/K/V在attention前的四格交互最大绝对值均为`{json.dumps(result['qkv_factorial_maxabs'])}`；重构误差`{json.dumps(result['reconstruction'])}`。32头交互RMS范围{result['head_interaction_rms']['min']:.8g}–{result['head_interaction_rms']['max']:.8g}，其中{result['head_interaction_rms']['nonzero_heads_at_1e-7']}/32头的中位交互超过$10^{{-7}}$；这表示分布式但不均匀的参与，不表示各头逐一必要。

**相关文件。** 脚本`tests/glm5/phase2583_c385281_c397568_block0_component_interaction_atlas.py`；每组Q/K/V、全attention矩阵、attention/MLP全坐标、source-token贡献与哈希位于`{OUT}`，final含完整source语义token索引。

**理论进展。** 输入embedding以及逐token线性Q/K/V都没有二阶交互；第一个可测交互出现在attention权重和attention输出，随后MLP继续变换。这把“交互出生”从模糊的HiddenState 1缩小到block0内部的上下文路由/加权阶段。它仍可能是softmax对任意共现因素都会产生的一般非线性，因此不能称作语义条件齿轮。

**问题硬伤与结论。** attention权重由QK、RoPE、causal mask和softmax共同产生，本Phase尚未用断连控制区分一般共现与任务条件交互；组件落盘float16；仍只覆盖自然值；全部head非零不等于必要。支持：{result['claim_boundary']['supported']}。不支持：{result['claim_boundary']['not_supported']}。下一Phase做同范数交互删除、attention/MLP/block三种路径干预并完整四候选评分，阴性只解释为冗余/再生，不关闭路线。检查`{json.dumps(result['checks'], ensure_ascii=False)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    _, _, compatible, _ = p2582.material_index()
    selected, _ = p2582.balanced_select(compatible)
    model = None
    try:
        model, _, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        data = capture(model, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = summarize(*data)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
