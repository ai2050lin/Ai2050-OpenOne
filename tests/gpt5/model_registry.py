from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "hf"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_dir: Path
    load_strategy: str
    trust_remote_code: bool = True
    attn_implementation: str = "eager"


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3": ModelSpec(
        key="qwen3",
        repo_id="Qwen/Qwen3-4B",
        local_dir=MODEL_ROOT / "qwen3-4b",
        load_strategy="cuda",
    ),
    "glm4": ModelSpec(
        key="glm4",
        repo_id="zai-org/glm-4-9b-chat-hf",
        local_dir=MODEL_ROOT / "glm4-9b-chat-hf",
        load_strategy="cuda",
    ),
    "deepseek7b": ModelSpec(
        key="deepseek7b",
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        local_dir=MODEL_ROOT / "deepseek-r1-distill-qwen-7b",
        load_strategy="cuda",
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise SystemExit(f"Unknown model '{model_key}'. Valid values: {valid}") from exc


def all_model_keys() -> list[str]:
    return list(MODEL_SPECS.keys())
