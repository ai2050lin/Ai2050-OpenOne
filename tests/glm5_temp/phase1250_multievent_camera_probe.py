#!/usr/bin/env python3
"""Development-only probe for a source-only versus multi-event response camera."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


BOS, DIRECT, CODE, REC, SEP, MAP, QUERY, ANSWER = range(8)
ENTITY_START, LABEL_START, CODE_START, SHIFT_START = 8, 10, 14, 18
VOCAB, LENGTH = 22, 23
SOURCE_POSITIONS = (4, 8)
SHIFT_POSITION = 11
ANSWER_POSITION = 22


def sequence(rep: int, codes: np.ndarray, shift: int, order: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping = (np.arange(4) + shift) % 4
    source = mapping[codes] + LABEL_START if rep == 0 else codes + CODE_START
    values = [BOS, DIRECT if rep == 0 else CODE, REC, ENTITY_START, int(source[0]), SEP,
              REC, ENTITY_START + 1, int(source[1]), SEP, MAP, SHIFT_START + shift]
    positions = {}
    for code in order:
        values.extend([CODE_START + int(code), LABEL_START + int(mapping[code])])
        positions[int(code)] = len(values) - 1
    values.extend([QUERY, ENTITY_START, ANSWER])
    return np.asarray(values, dtype=np.int64), positions


def batch(count: int, seed: int, forced_rep: int | None = None):
    rng = np.random.default_rng(seed)
    x, y, reps = [], [], []
    for _ in range(count):
        rep = int(rng.integers(2)) if forced_rep is None else forced_rep
        shift = int(rng.integers(4))
        codes = rng.choice(4, 2, replace=False)
        row, _ = sequence(rep, codes, shift, rng.permutation(4))
        x.append(row)
        y.append((int(codes[0]) + shift) % 4)
        reps.append(rep)
    return torch.tensor(np.stack(x)), torch.tensor(y), torch.tensor(reps)


def worlds(groups: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        codes = rng.choice(4, 2, replace=False)
        target = int(rng.choice([v for v in range(4) if v not in codes]))
        null = int(rng.choice([v for v in range(4) if v != codes[1]]))
        order = rng.permutation(4)
        for shift in range(4):
            for rep in range(2):
                receiver, positions = sequence(rep, codes, shift, order)
                tcodes = codes.copy(); tcodes[0] = target
                ncodes = codes.copy(); ncodes[1] = null
                target_ids, _ = sequence(rep, tcodes, shift, order)
                null_ids, _ = sequence(rep, ncodes, shift, order)
                rows.append({"group": group, "rep": rep, "shift": shift, "receiver": receiver,
                             "target": target_ids, "null": null_ids, "codes": codes.copy(),
                             "target_code": target, "null_code": null, "positions": positions})
    return rows


def projection(width: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(width, dim)).astype(np.float32) / np.sqrt(dim)


def center(logits: torch.Tensor):
    x = logits[:, -1, LABEL_START:LABEL_START + 4].float()
    return x - x.mean(-1, keepdim=True)


def fit(x, y, ridge=1e-2):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    z = np.concatenate([z, np.ones((len(z), 1))], 1)
    eye = np.eye(z.shape[1]); eye[-1, -1] = 0
    w = np.linalg.solve(z.T @ z + ridge * eye, z.T @ y)
    return mean, scale, w


def predict(x, model):
    mean, scale, w = model
    z = (x - mean) / scale
    return np.concatenate([z, np.ones((len(z), 1))], 1) @ w


def metrics(p, y):
    dot = np.sum(p * y, 1)
    cosine = dot / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1), 1e-8)
    return float(cosine.mean()), float((cosine > 0).mean()), float(np.mean(np.linalg.norm(p-y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-6)))


def train(seed=1250001):
    torch.manual_seed(seed)
    architecture = os.environ.get("PHASE1250_ARCH", "compact")
    configs = {
        "compact": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
        "wide": ModelConfig(layers=4, width=128, heads=4, mlp_width=256, max_length=LENGTH, vocab_size=VOCAB),
        "deep": ModelConfig(layers=6, width=128, heads=4, mlp_width=256, max_length=LENGTH, vocab_size=VOCAB),
    }
    config = configs[architecture]
    model = TinyCausalTransformer(config).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for step in range(6500):
        if step == 3000:
            opt.param_groups[0]["lr"] = 2e-4
        if step < 3000:
            x, y, _ = batch(512, seed + step + 1000, 1)
        else:
            cx, cy, _ = batch(384, seed + step + 1000, 1)
            dx, dy, _ = batch(128, seed + step + 2001000, 0)
            x, y = torch.cat([cx, dx]), torch.cat([cy, dy])
        loss = F.cross_entropy(model(x.cuda())[:, -1, LABEL_START:LABEL_START+4].float(), y.cuda())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if step >= 3000 and step % 100 == 99:
            tx, ty, tr = batch(8192, seed + 999999)
            with torch.inference_mode():
                pred = model(tx.cuda())[:, -1, LABEL_START:LABEL_START+4].argmax(-1).cpu()
            scores = [float((pred == ty).float().mean())] + [float((pred[tr == r] == ty[tr == r]).float().mean()) for r in range(2)]
            print("behavior_probe", architecture, step + 1, scores, flush=True)
            if min(scores) >= .995:
                print("behavior", step+1, scores, flush=True)
                return model.eval()
    raise RuntimeError("behavior failed")


@torch.no_grad()
def collect(model, rows, alpha):
    width = model.config.width
    event_dims = {"source": 80, "shift": 16, "map_receiver": 16, "map_donor": 16, "boundary": 16}
    projectors = {k: projection(width, d, 1250900+i) for i, (k, d) in enumerate(event_dims.items())}
    out = []
    for start in range(0, len(rows), 128):
        chunk = rows[start:start+128]
        receiver = torch.tensor(np.stack([r["receiver"] for r in chunk]), device="cuda")
        rlog, rs = model(receiver, return_states=True)
        rbase = center(rlog)
        for donor_name in ("target", "null"):
            donor = torch.tensor(np.stack([r[donor_name] for r in chunk]), device="cuda")
            _dlog, ds = model(donor, return_states=True)
            entity = 0 if donor_name == "target" else 1
            source_pos = SOURCE_POSITIONS[entity]
            hidden = rs[0].clone()
            hidden[:, source_pos] += alpha * (ds[0][:, source_pos] - rs[0][:, source_pos])
            patched_states = [hidden]
            for block in model.blocks:
                hidden = block(hidden)
                patched_states.append(hidden)
            patched_logits = model.lm_head(model.final_norm(hidden))
            actual = (center(patched_logits) - rbase).cpu().numpy()
            early_depth = int(np.ceil(len(model.blocks) / 3))
            middle_depth = int(np.ceil(len(model.blocks) / 2))
            late_depth = int(np.ceil(2 * len(model.blocks) / 3))
            for i, row in enumerate(chunk):
                recv_code = int(row["codes"][entity])
                donor_code = int(row["target_code"] if donor_name == "target" else row["null_code"])
                raw = {
                    "source": (patched_states[0][i, source_pos] - rs[0][i, source_pos]).cpu().numpy(),
                    "shift": (patched_states[early_depth][i, SHIFT_POSITION] - rs[early_depth][i, SHIFT_POSITION]).cpu().numpy(),
                    "map_receiver": (patched_states[middle_depth][i, row["positions"][recv_code]] - rs[middle_depth][i, row["positions"][recv_code]]).cpu().numpy(),
                    "map_donor": (patched_states[middle_depth][i, row["positions"][donor_code]] - rs[middle_depth][i, row["positions"][donor_code]]).cpu().numpy(),
                    "boundary": (patched_states[late_depth][i, ANSWER_POSITION] - rs[late_depth][i, ANSWER_POSITION]).cpu().numpy(),
                }
                projected = {k: raw[k] @ projectors[k] for k in raw}
                source80 = projected["source"]
                typed_source = np.zeros(80, np.float32)
                typed_source[row["rep"]*40:(row["rep"]+1)*40] = source80[:40]
                multi = np.concatenate([projected[k] for k in ("shift", "map_receiver", "map_donor", "boundary")])
                multi = np.concatenate([source80[:16], multi])
                typed_multi = np.zeros(80, np.float32)
                compact = np.concatenate([projected[k][:8] for k in ("source", "shift", "map_receiver", "map_donor", "boundary")])
                typed_multi[row["rep"]*40:(row["rep"]+1)*40] = compact
                condition = (projected["shift"] + projected["map_receiver"] + projected["map_donor"]) / np.sqrt(3)
                interaction = np.concatenate([projected["source"][:16], projected["shift"], projected["map_receiver"], projected["map_donor"], projected["source"][:16]*condition])
                out.append({"group": row["group"], "rep": row["rep"], "shift": row["shift"], "donor": donor_name,
                            "actual": actual[i], "source": source80, "typed_source": typed_source,
                            "multi": multi, "typed_multi": typed_multi, "interaction": interaction})
    return out


def main():
    model = train(int(os.environ.get("PHASE1250_SEED", "1250001")))
    rows = worlds(100, 1250555)
    records = []
    for alpha in (.25, .5, .75, 1.0):
        records.extend(collect(model, rows, alpha))
    families = ("source", "typed_source", "multi", "typed_multi", "interaction")
    for family in families:
        train_rows = [r for r in records if r["group"] < 50 and r.get("alpha", None) is None]
        # Alpha is inferred from feature scale; collect order is deterministic.
        train_rows = records[:0]
        train_rows = [r for block, a in enumerate((.25,.5,.75,1.0)) for r in records[block*1600:(block+1)*1600] if r["group"] < 50 and a in (.25,.5)]
        selection = [r for block, a in enumerate((.25,.5,.75,1.0)) for r in records[block*1600:(block+1)*1600] if 50 <= r["group"] < 70 and a == .75]
        confirm = [r for block, a in enumerate((.25,.5,.75,1.0)) for r in records[block*1600:(block+1)*1600] if r["group"] >= 70 and a == 1.0]
        camera = fit(np.stack([r[family] for r in train_rows]), np.stack([r["actual"] for r in train_rows]))
        if family == "source":
            target_norm = np.mean([np.linalg.norm(r["actual"]) for r in confirm if r["donor"] == "target"])
            null_norm = np.mean([np.linalg.norm(r["actual"]) for r in confirm if r["donor"] == "null"])
            print("effect_norms", target_norm, null_norm, "difference", target_norm-null_norm, "ratio", target_norm/max(null_norm, 1e-8), flush=True)
        print(family, "selection", metrics(predict(np.stack([r[family] for r in selection]), camera), np.stack([r["actual"] for r in selection])),
              "confirmation", metrics(predict(np.stack([r[family] for r in confirm]), camera), np.stack([r["actual"] for r in confirm])), flush=True)
        for donor in ("target", "null"):
            for rep in (0, 1):
                subset = [r for r in confirm if r["donor"] == donor and r["rep"] == rep]
                print(" ", donor, "direct" if rep == 0 else "code",
                      metrics(predict(np.stack([r[family] for r in subset]), camera), np.stack([r["actual"] for r in subset])), flush=True)


if __name__ == "__main__":
    main()
