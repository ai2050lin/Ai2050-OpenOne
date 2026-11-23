"""Package frozen native-unit observations for the offline client. No inference."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write

PACK = ROOT / "tests/glm5/result/qwen14_client_v2/native"
OUTPUT = ROOT / "research/glm5/docs/previews/client-review/neuron-profiles.json"
SNAPSHOT = "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def build():
    manifest = json.loads((PACK / "manifest.json").read_bytes())
    assert manifest["kind"] == "real" and manifest["model_snapshot"] == SNAPSHOT
    files = {f["path"]: f for f in manifest["files"]}

    def checked(name):
        path = PACK / name
        expected = files[name]
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Native archive mismatch: {path}")
        return path

    documents = json.loads(checked("documents.json").read_bytes())
    assert documents["model_snapshot"] == SNAPSHOT
    positions, docs = documents["positions"], documents["docs"]
    tokenizer = Tokenizer.from_file(str(checked("tokenizer.json")))
    examples = {}
    for di, doc in enumerate(docs):
        if doc["split"] != "holdout":
            continue
        for slot, pos in enumerate(positions):
            # Restrict each excerpt to its visible prefix, keeping the input token.
            ids = doc["tokens"]
            examples[di*len(positions)+slot] = [di, pos, ids[pos],
                tokenizer.decode(ids[max(0, pos-31):pos+1], skip_special_tokens=False)]
    document_ids = {di:{key:doc[key] for key in ("doc_id","source","repo","revision","license")}
                    for di, doc in enumerate(docs) if doc["split"] == "holdout"}
    words = {"": 0}
    layers = {}
    for b in range(28, 40):
        tag = f"B{b}"
        profile = json.loads(checked(tag+"-profile.json").read_bytes())
        cues = json.loads(checked(tag+"-cues.json").read_bytes())
        raw = np.load(checked(tag+"-A.npy"), mmap_mode="r")
        n = profile["width"]
        if not (n == manifest["width"] == len(cues["units"]) and profile["block"] == b
                and profile["model_snapshot"] == cues["model_snapshot"] == SNAPSHOT
                and profile["run_id"] == cues["run_id"] and profile["kind"] == cues["kind"] == "real"
                and raw.shape == (len(docs), len(positions), n) and raw.dtype == np.uint16):
            raise ValueError(f"Invalid layer identity/shape in {tag}")
        stats = profile["fields"]["A"]
        poles = []
        for side in ("positive_indices", "negative_indices"):
            ids = np.array(stats["holdout"][side], dtype=np.int32)[:,0]
            assert all(int(key) in examples for key in np.unique(ids))
            peaks = raw[ids//len(positions), ids%len(positions), np.arange(n)]
            poles.append([array(ids, "<u2"), array(peaks, "<u2")])
        hits = []
        term_codes = []
        for i, row in enumerate(cues["units"]):
            assert row["unit"] == i and not row["semantic_confirmed"]
            terms = row["positive_keywords"]
            term_codes.extend(words.setdefault(t, len(words)) for t in terms[:3])
            term_codes.extend([0] * (3-min(3,len(terms))))
            a, b_hold = (row["repeat"][split] for split in ("confirm", "holdout"))
            assert a["n"] == b_hold["n"] == 4
            hits.append((a["hits"] << 4) | b_hold["hits"])
        layers[b] = {"run": profile["run_id"], "mean": array(stats["mean"], "<f8"),
            "std": array(stats["std"], "<f8"), "poles":poles, "terms":array(term_codes,"<u4"),
            "repeat":array(hits,"u1"), "hashes":{
                suffix:files[tag+suffix]["sha256"] for suffix in ("-profile.json","-cues.json","-A.npy")}}
        print(f"{tag}: {n} native units packaged", flush=True)
    return {"snapshot":SNAPSHOT, "run":manifest["run_id"], "manifest":digest(PACK/"manifest.json"),
        "width":manifest["width"], "docs":document_ids, "examples":examples, "words":list(words),
        "counts":{"tokens":len(docs)*len(positions),"documents":len(docs),"readouts":len(positions)},
        "layers":layers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify reproducibility without writing")
    args = parser.parse_args()
    if not args.check:
        check_write(OUTPUT, 64*2**20, operation="构建静态神经元档案")
    payload = json.dumps(build(), ensure_ascii=False, separators=(",",":")).encode("utf-8")
    if args.check:
        assert OUTPUT.read_bytes() == payload, "Regenerate the preview artefact."
    else:
        staging = OUTPUT.with_suffix(".json.tmp")
        try:
            staging.write_bytes(payload)
            staging.replace(OUTPUT)
        finally:
            staging.unlink(missing_ok=True)
    print(f"{len(payload):,} bytes; SHA-256 {hashlib.sha256(payload).hexdigest()}")
