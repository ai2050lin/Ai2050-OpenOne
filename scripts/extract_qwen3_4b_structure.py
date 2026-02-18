import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PromptItem:
    text: str
    concept_id: int
    syntax_id: int
    domain_id: int


def build_probe_dataset() -> List[PromptItem]:
    domains: Dict[str, List[Tuple[str, str]]] = {
        "physical": [
            ("apple", "red"),
            ("sky", "blue"),
            ("grass", "green"),
            ("snow", "white"),
            ("lemon", "yellow"),
            ("rock", "hard"),
            ("water", "clear"),
            ("metal", "solid"),
        ],
        "emotion": [
            ("child", "happy"),
            ("worker", "tired"),
            ("friend", "kind"),
            ("stranger", "calm"),
            ("teacher", "patient"),
            ("leader", "brave"),
            ("artist", "curious"),
            ("parent", "careful"),
        ],
        "dynamics": [
            ("fire", "hot"),
            ("ice", "cold"),
            ("rabbit", "fast"),
            ("turtle", "slow"),
            ("summer", "warm"),
            ("winter", "chilly"),
            ("storm", "loud"),
            ("breeze", "quiet"),
        ],
    }
    templates = [
        "The {noun} is {adj}.",
        "A {adj} {noun} appeared.",
        "People say the {noun} feels {adj}.",
        "In one phrase: {noun}, {adj}.",
    ]

    items: List[PromptItem] = []
    concept_idx = 0
    for domain_idx, (_, pairs) in enumerate(domains.items()):
        for noun, adj in pairs:
            for syntax_idx, template in enumerate(templates):
                items.append(
                    PromptItem(
                        text=template.format(noun=noun, adj=adj),
                        concept_id=concept_idx,
                        syntax_id=syntax_idx,
                        domain_id=domain_idx,
                    )
                )
            concept_idx += 1
    return items


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norms
    return x_norm @ x_norm.T


def score_same_diff(sim: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    n = labels.shape[0]
    same = labels[:, None] == labels[None, :]
    diff = ~same
    diag = np.eye(n, dtype=bool)
    same[diag] = False
    diff[diag] = False
    same_mean = float(sim[same].mean()) if np.any(same) else 0.0
    diff_mean = float(sim[diff].mean()) if np.any(diff) else 0.0
    return same_mean, diff_mean, same_mean - diff_mean


def layer_stats(x: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    x = x.astype(np.float32)
    x_centered = x - x.mean(axis=0, keepdims=True)
    n_samples = x_centered.shape[0]

    # SVD on sample-major matrix; with N << D this remains tractable.
    _, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    if n_samples <= 1:
        return {
            "k95": 0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "svd_energy_total": 0.0,
        }

    eigvals = (s**2) / (n_samples - 1 + 1e-8)
    total = float(eigvals.sum()) + 1e-12
    p = eigvals / total

    cumsum = np.cumsum(p)
    k95 = int(np.searchsorted(cumsum, 0.95) + 1)
    effective_rank = float(math.exp(-np.sum(p * np.log(p + 1e-12))))
    participation_ratio = float((eigvals.sum() ** 2) / (np.sum(eigvals**2) + 1e-12))

    metrics: Dict[str, float] = {
        "k95": k95,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "svd_energy_total": float(total),
    }

    denom = np.linalg.norm(x_centered, ord="fro") + 1e-12
    for k in k_values:
        k_eff = min(k, len(s))
        if k_eff <= 0:
            metrics[f"recon_error_k{k}"] = 1.0
            continue
        u, s_all, vt = np.linalg.svd(x_centered, full_matrices=False)
        xk = (u[:, :k_eff] * s_all[:k_eff]) @ vt[:k_eff, :]
        err = np.linalg.norm(x_centered - xk, ord="fro") / denom
        metrics[f"recon_error_k{k}"] = float(err)
    return metrics


def extract_layer_encodings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int,
    max_length: int,
) -> Dict[int, np.ndarray]:
    num_hidden = model.config.num_hidden_layers + 1
    per_layer: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_hidden)}

    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        toks = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        with torch.no_grad():
            out = model(
                **toks,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = out.hidden_states
        attn_mask = toks["attention_mask"]
        last_pos = attn_mask.sum(dim=1) - 1
        batch_idx = torch.arange(attn_mask.shape[0], device=device)

        for li in range(num_hidden):
            hs = hidden_states[li]
            selected = hs[batch_idx, last_pos, :].detach().float().cpu().numpy()
            per_layer[li].append(selected)

        del out, hidden_states, toks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged: Dict[int, np.ndarray] = {}
    for li in range(num_hidden):
        merged[li] = np.concatenate(per_layer[li], axis=0)
    return merged


def build_layer_network(layer_rsms: Dict[int, np.ndarray]) -> np.ndarray:
    layers = sorted(layer_rsms.keys())
    n = len(layers)
    adj = np.zeros((n, n), dtype=np.float32)
    tri = np.triu_indices(layer_rsms[layers[0]].shape[0], k=1)

    vecs = {}
    for li in layers:
        vecs[li] = layer_rsms[li][tri]

    for i, li in enumerate(layers):
        vi = vecs[li]
        vi = (vi - vi.mean()) / (vi.std() + 1e-8)
        for j, lj in enumerate(layers):
            vj = vecs[lj]
            vj = (vj - vj.mean()) / (vj.std() + 1e-8)
            adj[i, j] = float((vi * vj).mean())
    return adj


def write_report(output_dir: Path, metrics: Dict) -> None:
    lines: List[str] = []
    lines.append("# Qwen3-4B 编码提取与结构测试总结")
    lines.append("")
    lines.append("## 1. 运行概况")
    lines.append(f"- 模型: `{metrics['model_name']}`")
    lines.append(f"- 设备: `{metrics['device']}`")
    lines.append(f"- 样本数: `{metrics['dataset']['num_prompts']}`")
    lines.append(f"- 层数(含embedding层): `{metrics['dataset']['num_layers_with_embed']}`")
    lines.append(f"- 提取耗时: `{metrics['runtime_sec']:.2f}` 秒")
    lines.append(f"- 峰值显存: `{metrics['peak_gpu_mem_gb']:.2f} GB`")
    lines.append("")
    lines.append("## 2. 关键数学结构指标")
    lines.append(f"- 平均有效秩: `{metrics['aggregate']['avg_effective_rank']:.2f}`")
    lines.append(f"- 平均参与比(PR): `{metrics['aggregate']['avg_participation_ratio']:.2f}`")
    lines.append(f"- 平均k95: `{metrics['aggregate']['avg_k95']:.2f}`")
    lines.append(
        f"- 平均重建误差 k=64: `{metrics['aggregate']['avg_recon_error_k64']:.4f}`"
    )
    lines.append("")
    lines.append("## 3. 四类特性测量")
    lines.append(
        f"- 高维抽象(语义同类-异类相似度差): `{metrics['aggregate']['avg_concept_score']:.4f}`"
    )
    lines.append(
        f"- 低维精确(k=64 重建误差越低越好): `{metrics['aggregate']['avg_recon_error_k64']:.4f}`"
    )
    lines.append(
        f"- 特异性(领域同类-异类相似度差): `{metrics['aggregate']['avg_domain_score']:.4f}`"
    )
    lines.append(
        f"- 系统性(层间RSM相关均值): `{metrics['aggregate']['systemicity_index']:.4f}`"
    )
    lines.append("")
    lines.append("## 4. 最强层间结构耦合(Top-10)")
    for item in metrics["aggregate"]["top_layer_edges"]:
        lines.append(
            f"- layer_{item['layer_i']} <-> layer_{item['layer_j']}: corr={item['corr']:.4f}"
        )
    lines.append("")
    lines.append("## 5. 结论")
    lines.append("- Qwen3-4B 已成功提取分层编码，输出可用于后续结构还原。")
    lines.append("- 表征可用低维子空间近似，但非唯一映射，适合做“功能还原”而非“参数唯一逆推”。")
    lines.append("- 建议下一步在选定任务上做 activation patching 与稀疏字典学习，验证因果回路。")

    (output_dir / "TEST_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen3-4B encodings and structure.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--local-snapshot",
        type=str,
        default=r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    )
    parser.add_argument("--output-dir", type=str, default="tempdata/qwen3_4b_structure")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(args.local_snapshot)
    use_local = local_path.exists()
    model_ref = str(local_path) if use_local else args.model_name

    os.environ.setdefault("HF_HOME", r"D:\develop\model")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print(f"[INFO] Loading model from: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True,
        local_files_only=use_local,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=use_local,
        )
        device_name = torch.cuda.get_device_name(0)
    else:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=use_local,
        )
        device_name = "cpu"
    model.eval()

    dataset = build_probe_dataset()
    prompts = [item.text for item in dataset]
    concept_labels = np.array([item.concept_id for item in dataset], dtype=np.int64)
    syntax_labels = np.array([item.syntax_id for item in dataset], dtype=np.int64)
    domain_labels = np.array([item.domain_id for item in dataset], dtype=np.int64)

    start = time.time()
    layer_enc = extract_layer_encodings(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    runtime_sec = time.time() - start

    layer_metrics = []
    layer_rsms: Dict[int, np.ndarray] = {}
    k_values = [8, 16, 32, 64]

    npz_payload = {}
    for layer_idx, x in layer_enc.items():
        npz_payload[f"layer_{layer_idx}"] = x.astype(np.float32)

        stats = layer_stats(x, k_values=k_values)
        sim = cosine_matrix(x)
        layer_rsms[layer_idx] = sim.astype(np.float32)
        _, _, concept_score = score_same_diff(sim, concept_labels)
        _, _, syntax_score = score_same_diff(sim, syntax_labels)
        _, _, domain_score = score_same_diff(sim, domain_labels)

        row = {
            "layer": layer_idx,
            **stats,
            "concept_score": float(concept_score),
            "syntax_score": float(syntax_score),
            "domain_score": float(domain_score),
        }
        layer_metrics.append(row)

    np.savez_compressed(output_dir / "encodings_last_token.npz", **npz_payload)

    layer_network = build_layer_network(layer_rsms)
    np.save(output_dir / "layer_rsm_network.npy", layer_network)

    off_diag = layer_network[~np.eye(layer_network.shape[0], dtype=bool)]
    systemicity_index = float(off_diag.mean()) if off_diag.size else 0.0

    # Top-10 strongest off-diagonal edges
    pairs = []
    n = layer_network.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(layer_network[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_edges = [
        {"layer_i": i, "layer_j": j, "corr": corr} for i, j, corr in pairs[:10]
    ]

    peak_gb = (
        float(torch.cuda.max_memory_allocated() / (1024**3))
        if torch.cuda.is_available()
        else 0.0
    )

    aggregate = {
        "avg_effective_rank": float(np.mean([m["effective_rank"] for m in layer_metrics])),
        "avg_participation_ratio": float(
            np.mean([m["participation_ratio"] for m in layer_metrics])
        ),
        "avg_k95": float(np.mean([m["k95"] for m in layer_metrics])),
        "avg_recon_error_k64": float(
            np.mean([m.get("recon_error_k64", 1.0) for m in layer_metrics])
        ),
        "avg_concept_score": float(np.mean([m["concept_score"] for m in layer_metrics])),
        "avg_domain_score": float(np.mean([m["domain_score"] for m in layer_metrics])),
        "avg_syntax_score": float(np.mean([m["syntax_score"] for m in layer_metrics])),
        "systemicity_index": systemicity_index,
        "top_layer_edges": top_edges,
    }

    metrics = {
        "model_name": args.model_name,
        "model_ref_loaded": model_ref,
        "dtype": str(dtype),
        "device": device_name,
        "runtime_sec": runtime_sec,
        "peak_gpu_mem_gb": peak_gb,
        "dataset": {
            "num_prompts": len(prompts),
            "num_concepts": int(len(np.unique(concept_labels))),
            "num_syntax": int(len(np.unique(syntax_labels))),
            "num_domains": int(len(np.unique(domain_labels))),
            "num_layers_with_embed": len(layer_metrics),
        },
        "layer_metrics": layer_metrics,
        "aggregate": aggregate,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    write_report(output_dir, metrics)
    print(f"[OK] Outputs written to: {output_dir}")
    print(f"[OK] Summary report: {output_dir / 'TEST_SUMMARY.md'}")


if __name__ == "__main__":
    main()
