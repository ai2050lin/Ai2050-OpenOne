from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "hf" / "pythia-1.4b-deduped" / "step143000"
CASE_PATH = ROOT / "tests" / "glm5" / "result" / "phase1121_wordnet_adjective_double_orthogonal" / "protocol" / "cases.pythia.jsonl"


class Adapter(nn.Module):
    def __init__(self, hidden_size: int = 2048, rank: int = 16) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(torch.nn.functional.gelu(self.down(hidden.float()))).to(hidden.dtype)


def main() -> None:
    torch.manual_seed(1125)
    torch.cuda.reset_peak_memory_stats()
    row = json.loads(CASE_PATH.read_text(encoding="utf-8").splitlines()[0])
    input_ids = torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = Adapter().cuda().train()
    layer = model.gpt_neox.layers[12]

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> object:
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + adapter(hidden), *output[1:])
        return output + adapter(output)

    handle = layer.register_forward_hook(hook)
    output = model(input_ids=input_ids, use_cache=False)
    true_id = int(row["candidate_first_token_ids"]["true"][0])
    false_id = int(row["candidate_first_token_ids"]["false"][0])
    candidate_logits = output.logits[0, -1, [false_id, true_id]].float().unsqueeze(0)
    target = torch.tensor([1 if row["truth"] else 0], device="cuda")
    loss = torch.nn.functional.cross_entropy(candidate_logits, target)
    loss.backward()
    gradient_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()
    )
    print(json.dumps({
        "loss": float(loss.detach().cpu()),
        "gradient_finite": bool(gradient_finite),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
    }, indent=2, sort_keys=True))
    handle.remove()


if __name__ == "__main__":
    main()
