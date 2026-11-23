"""Rebuild or check offline presentation excerpts using indexed local files; no inference."""

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
TRACE = REPO / "tests/glm5/result/live_traces/fcf90469ce80cc33091526080da6c8b11b03da3d9c36a9b9b3031d31f4b1876e"
CHECK = REPO / "tests/glm5/result/workspace_full_native/source-check.json"
ADAPTER = REPO / "research/glm5/docs/previews/free-model-data.js"
SNAPSHOTS = {
    "Qwen3-14B": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "Qwen3-4B": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "GLM-4-9B": "64baf4f7e4fb31aeb689b7cb8a63b2484be0aa1ca5ed7108f3c21f83c5879b3f",
}
# Freeze exactly the material the preview publishes, including headers, config, index, coordinates and BF16 bits.
# This guards against resealing a different local checkpoint under an existing archival model id.
EXCERPT_DIGESTS = {
    "Qwen3-14B": "b82a2dc9ef20715c020c1b9039473548b0371d1289a2f63aee4b1ab9ecdc5dba",
    "Qwen3-4B": "d4083ff515864a32bf353d08f1beb41b4e7d00eaaf34ecd602740c1252a4942b",
    "GLM-4-9B": "ff96d8cea397be0f12435bcf6ce33c2666257058aaaa2344656c0c884565911b",
}
LENS_TOKENS = [(32, "A"), (33, "B"), (34, "C"), (35, "D"),
               (104165, "红色"), (105681, "蓝色"), (6660, "wood"), (81914, "silver")]
GLM_LENS_TOKENS = [(32, "A"), (33, "B"), (34, "C"), (35, "D"),
                   (101520, "红色"), (105047, "蓝色"), (6660, "wood"), (81244, "silver")]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(data):
    return base64.b64encode(data).decode("ascii")


def sample_areas(key, shape, spec):
    nr, *columns = shape
    nc = columns[0] if columns else 1
    norm_height = 128 if "self_attn." in key and "_norm." in key else 8
    areas = [(0, 0, min(nr, norm_height), min(nc, 8)), (nr - 1, nc - 1, 1, 1)]
    if key.startswith("model.layers.26.self_attn.") and key.endswith(".weight"):
        h = 36 if spec["id"] == "Qwen3-14B" else 14
        group = h // (spec["heads"] // spec["kvHeads"])
        for proj, row, col in [("q_proj", 128*h, 0), ("k_proj", 128*group, 0),
                               ("v_proj", 128*group, 0), ("o_proj", 0, 128*h)]:
            if proj in key:
                areas.append((row, col, 8, 8))
    if key.startswith(("model.layers.26.mlp.", "model.layers.36.mlp.")):
        areas.append((0, 2429, 8, 8) if "down_proj" in key else (2429, 0, 8, 8))
        if "gate_up_proj" in key:
            areas.append((spec["ffn"] + 2429, 0, 8, 8))
    if key == "model.norm.weight":
        areas.append((0, 0, nr, 1))
    output_name = "model.embed_tokens.weight" if spec["tied"] else "lm_head.weight"
    if key == output_name:
        areas.extend((token, 0, 1, nc) for token in spec["lensTokenIds"])
    return areas


def collect_model(root, name, snapshot):
    index_raw = (root / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)
    config_raw = (root / "config.json").read_bytes()
    cfg = json.loads(config_raw)
    # The tokenizer uses UTF-8 bytes mapped into the GPT-2 token alphabet.
    # Check the actual decoded strings for all retained single tokens on each model.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokens = GLM_LENS_TOKENS if name == "GLM-4-9B" else LENS_TOKENS
    for token, label in tokens:
        assert tokenizer.decode([token]) == label
        assert tokenizer.encode(label, add_special_tokens=False).ids == [token]
    spec = {
        "id": name, "snapshot": snapshot, "layers": cfg["num_hidden_layers"],
        "heads": cfg["num_attention_heads"], "kvHeads": cfg["num_key_value_heads"],
        "headDim": cfg["head_dim"], "hidden": cfg["hidden_size"], "ffn": cfg["intermediate_size"],
        "vocab": cfg["vocab_size"], "tied": cfg["tie_word_embeddings"],
        "rmsNormEps": cfg["rms_norm_eps"], "lensTokenIds": [token for token, _ in tokens],
        "lensTokens": [{"id": token, "label": label} for token, label in tokens],
        "architecture": cfg["architectures"][0],
        "fusedGateUp": "model.layers.0.mlp.gate_up_proj.weight" in index["weight_map"],
        "weightBytes": index["metadata"]["total_size"], "configSha256": sha(config_raw),
        "indexSha256": sha(index_raw), "shards": [], "entries": [],
    }
    total = 0
    for shard_index, filename in enumerate(sorted(set(index["weight_map"].values()))):
        shard = root / filename
        length = shard.stat().st_size
        with shard.open("rb") as stream:
            n = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < n < min(length, 32 * 1024 * 1024)
            raw = stream.read(n)
            header = json.loads(raw)
            spec["shards"].append({"name": filename, "bytes": length, "headerBytes": n, "headerSha256": sha(raw)})
            for key, item in sorted(header.items()):
                if key == "__metadata__":
                    continue
                assert index["weight_map"][key] == filename and item["dtype"] == "BF16"
                shape = item["shape"]
                assert len(shape) in (1, 2) and all(i > 0 for i in shape)
                count = math.prod(shape)
                nr, nc = shape[0], shape[1] if len(shape) == 2 else 1
                start, end = item["data_offsets"]
                assert end - start == 2 * count and 8 + n + end <= length
                total += count
                samples = []
                for row, col, height, width in sample_areas(key, shape, spec):
                    assert row + height <= nr and col + width <= nc
                    data = bytearray()
                    if width == nc and col == 0:
                        stream.seek(8 + n + start + row * nc * 2)
                        data.extend(stream.read(2 * height * width))
                    else:
                        for i in range(height):
                            stream.seek(8 + n + start + ((row + i) * nc + col) * 2)
                            data.extend(stream.read(width * 2))
                    assert len(data) == 2 * height * width
                    samples.append([row, col, height, width, encoded(data)])
                spec["entries"].append([key, shape, shard_index, samples])
    assert len(spec["entries"]) == len(index["weight_map"]) and total * 2 == spec["weightBytes"]
    spec["entries"].sort(key=lambda item: item[0])
    evidence = {key: spec[key] for key in ("configSha256", "indexSha256", "shards", "entries")}
    identity = sha(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii"))
    if identity != EXCERPT_DIGESTS.get(name):
        raise ValueError(f"{name}: local weights, tensor directory or frozen excerpt coordinates differ from the recorded source ({identity})")
    spec["excerptSha256"] = identity
    return spec


def collect_trace(root):
    raw = (root / "manifest.json").read_bytes()
    m = json.loads(raw)
    check = json.loads(CHECK.read_bytes())
    assert sha(raw) == check["manifest_file_sha256"]
    assert m["model_snapshot"] == SNAPSHOTS["Qwen3-4B"] and m["kind"] == "real"
    pos = check["absolute_position"]
    site = m["readout_positions"].index(pos)
    assert m["token_ids"][pos] == check["token_id"] and m["n_layers"] == 36
    trace = {
        "model": "Qwen3-4B", "root": TRACE.relative_to(REPO).as_posix() + "/", "manifestSha256": sha(raw),
        "run": m["run_id"], "snapshot": m["model_snapshot"], "traceId": m["trace_id"], "caseId": m["sample_id"],
        "text": m["text"], "position": pos, "tokenId": m["token_ids"][pos], "token": m["tokens"][pos],
        "answer": m["output_top"][0]["text"], "tokens": m["tokens"][:pos+1],
        "tokenIds": m["token_ids"][:pos+1], "blocks": [], "hidden": [],
    }
    hashes = {entry["path"]: entry["sha256"] for entry in m["files"]}
    for b in range(m["n_layers"]):
        filename = "block_%02d.npz" % b
        file = root / filename
        assert sha(file.read_bytes()) == hashes[filename]
        block = {"file": filename, "sha256": hashes[filename]}
        with np.load(file, allow_pickle=False) as arrays:
            for key in ("head_z", "q", "k", "v", "activation"):
                arr = arrays[key]
                assert arr.dtype == np.dtype("<u2")
                sample = arr[:, pos, :] if key in ("q", "k", "v") else arr[site]
                width = m["n_kv_heads"] if key in ("k", "v") else m["n_heads"]
                assert sample.size == (m["d_ff"] if key == "activation" else width * m["head_dim"])
                block[key] = encoded(sample.tobytes())
            scores = arrays["attention_rows"]
            assert scores.dtype == np.dtype("<u2") and scores.shape == (3, m["n_heads"], m["n_tokens"])
            block["attention"] = encoded(scores[site, :, :pos+1].tobytes())
        trace["blocks"].append(block)
    hidden = root / "hidden_readouts.npy"
    assert sha(hidden.read_bytes()) == hashes[hidden.name]
    trace["hiddenSha256"] = hashes[hidden.name]
    h = np.load(hidden, mmap_mode="r", allow_pickle=False)
    assert h.shape == (37, 3, 2560) and h.dtype == np.dtype("<u2")
    trace["hidden"] = [{"bits": encoded(field[site].tobytes())} for field in h]
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="replace only the generated payload within the existing adapter")
    mode.add_argument("--check", action="store_true", help="bit-for-bit compare generated payload (default)")
    parser.add_argument("--model-root", type=Path, default=Path("D:/develop/model/Qwen"))
    parser.add_argument("--glm-root", type=Path, default=Path("D:/develop/model/ZhipuAI/glm-4-9b-chat-hf"))
    parser.add_argument("--trace-root", type=Path, default=TRACE)
    args = parser.parse_args()
    models = [collect_model(args.glm_root if name == "GLM-4-9B" else args.model_root / name, name, snapshot)
              for name, snapshot in SNAPSHOTS.items()]
    trace = collect_trace(args.trace_root)
    def compact(obj):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
    # Keep all adapter code intact, with a separately addressable line per tensor / archived block.
    shapes = []
    for spec in models:
        entries = spec.pop("entries")
        shapes.append(compact(spec)[:-1] + ',"entries":[\n' + ',\n'.join(map(compact, entries)) + '\n]}')
    blocks, hidden = trace.pop("blocks"), trace.pop("hidden")
    payload = ('  initialize({"models":[' + ',\n'.join(shapes) + '],"trace":' + compact(trace)[:-1] +
               ',"blocks":[\n' + ',\n'.join(map(compact, blocks)) + '],"hidden":[\n' + ',\n'.join(map(compact, hidden)) + ']}});\n')
    old = ADAPTER.read_text(encoding="utf8")
    start = old.index('  initialize({"models":[')
    end = old.index('})();', start)
    if args.write:
        # Atomic swap also preserves a working preview if the process is interrupted before the write completes.
        staging = ADAPTER.with_suffix(".js.new")
        staging.write_text(old[:start] + payload + old[end:], encoding="utf8")
        staging.replace(ADAPTER)
    elif old[start:end] != payload:
        raise SystemExit("Packaged native data differs from indexed sources. Run --write to regenerate.")
    print("Checked three indexed models, full output vectors and all 36 archived 4B blocks.")


if __name__ == "__main__":
    main()
