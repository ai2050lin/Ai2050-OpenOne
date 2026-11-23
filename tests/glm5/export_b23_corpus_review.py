"""Export already published B23 corpus profiles and two identified, full-width BF16 rows."""
import argparse
import json
import subprocess

import numpy as np
from tokenizers import Tokenizer

from export_goodfire_profiles import ROOT, SNAPSHOT, array, digest
from deploy.storage_policy import check_write, staging_file

DIRECTORY = "tests/glm5/result/goodfire_middle_20261113_v2/qwen14"
PACK = ROOT / DIRECTORY
OUTPUT = ROOT / "research/glm5/docs/previews/client-review/b23-corpus-data.json"
REPORT = ROOT / "tests/glm5/result/model_analysis_b23_review/source-check.json"
RUN = "QWEN14-GOODFIRE-B18-B28-20261113-v2"
BLOCK, WIDTH, UNIT = 23, 17408, 7708


def export():
    manifest = json.loads((PACK / "manifest.json").read_bytes())
    assert manifest["run_id"] == RUN and manifest["model_snapshot"] == SNAPSHOT
    assert manifest["kind"] == "real" and manifest["state"] == "complete"
    assert BLOCK in manifest["blocks"] and manifest["width"] == WIDTH
    entries = {entry["path"]: entry for entry in manifest["files"]}
    hashes = {}

    def checked(filename):
        path = PACK / filename
        entry = entries[filename]
        actual = digest(path)
        if path.stat().st_size != entry["bytes"] or actual != entry["sha256"]:
            raise ValueError(f"Archive verification failed for {filename}")
        hashes[filename] = actual
        return path

    profile = json.loads(checked("B23-profile.json").read_bytes())
    cues = json.loads(checked("B23-cues.json").read_bytes())
    documents = json.loads(checked("documents.json").read_bytes())
    for item in (profile, cues, documents):
        assert item["run_id"] == RUN and item["model_snapshot"] == SNAPSHOT and item["kind"] == "real"
    assert profile["width"] == WIDTH and profile["block"] == BLOCK and len(cues["units"]) == WIDTH
    raw = np.load(checked("B23-A.npy"), mmap_mode="r", allow_pickle=False)
    docs, positions = documents["docs"], documents["positions"]
    assert raw.dtype == np.dtype("<u2") and raw.shape == (len(docs), len(positions), WIDTH)
    assert raw.shape[:2] == (1024, 8)
    tokenizer = Tokenizer.from_file(str(checked("tokenizer.json")))
    examples, sources = {}, {}
    rows, poles = [], []
    stats = profile["fields"]["A"]
    for side, key in (("positive_indices", "positive"), ("negative_indices", "negative")):
        # The original study has already released these indices. Use the archived ordering.
        ids = np.asarray(stats["holdout"][side], dtype=np.int64)[:, 0]
        assert ids.shape == (WIDTH,) and ids.min() >= 0 and ids.max() < len(docs) * len(positions)
        for index in np.unique(ids):
            di, slot = divmod(int(index), len(positions))
            doc, pos = docs[di], positions[slot]
            assert doc["split"] == "holdout" and doc["doc_index"] == di
            sources[di] = {k: doc[k] for k in ("doc_id", "repo", "source", "revision", "license")}
            if int(index) not in examples:
                excerpt = tokenizer.decode(doc["tokens"][max(0, pos - 31):pos + 1], skip_special_tokens=False)
                examples[int(index)] = [di, pos, doc["tokens"][pos], excerpt]
        values = raw[ids // len(positions), ids % len(positions), np.arange(WIDTH)]
        poles.append([array(ids, "<u2"), array(values, "<u2")])
        index = int(ids[UNIT])
        di, pos, token_id, text = examples[index]
        slot = index % len(positions)
        vector = raw[di, slot]
        value = float(np.array(int(vector[UNIT]) << 16, dtype="<u4").view("<f4"))
        rows.append({"key": key, "doc": docs[di]["doc_id"], "document": docs[di]["doc_id"],
                     "documentIndex": di, "slot": slot, "readoutIndex": index, "position": pos,
                     "tokenId": token_id, "text": text, "split": docs[di]["split"], "source": sources[di],
                     "activation": value, "bits": array(vector, "<u2")})

    words, terms, repeats = {"": 0}, [], []
    for index, cue in enumerate(cues["units"]):
        assert cue["unit"] == index and cue["semantic_confirmed"] is False
        keywords = cue["positive_keywords"][:3]
        terms.extend(words.setdefault(word, len(words)) for word in keywords)
        terms.extend([0] * (3 - len(keywords)))
        confirm, holdout = (cue["repeat"][s] for s in ("confirm", "holdout"))
        assert confirm["n"] == holdout["n"] == 4 and max(confirm["hits"], holdout["hits"]) <= 4
        repeats.append((confirm["hits"] << 4) | holdout["hits"])
    source = {"directory": DIRECTORY, "manifest": DIRECTORY + "/manifest.json", "sha256": digest(PACK / "manifest.json")}
    counts = {"documents": len(docs), "readouts": len(positions), "tokens": len(docs) * len(positions)}
    return {"model": "Qwen3-14B", "snapshot": SNAPSHOT, "block": BLOCK, "width": WIDTH, "run": RUN,
            "source": source, "hashes": hashes, "counts": counts, "positions": positions, "rows": rows,
            "stats": {"mean": array(stats["mean"], "<f8"), "std": array(stats["std"], "<f8"),
                      "repeat": array(repeats, "u1"), "terms": array(terms, "<u4"), "poles": poles,
                      "hashes": {suffix: hashes["B23" + suffix] for suffix in ("-A.npy", "-profile.json", "-cues.json")}},
            "examples": examples, "docs": sources, "words": list(words)}


def write(path, data):
    check_write(path, len(data) * 2, operation="导出 B23 已归档证据的静态索引")
    path.parent.mkdir(parents=True, exist_ok=True)
    with staging_file(path) as staged:
        staged.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing preview against the archive.")
    parser.add_argument("--verify-report", action="store_true", help="Audit the browser module against every original unit and save its report.")
    args = parser.parse_args()
    pack = export()
    encoded = json.dumps(pack, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if args.check:
        if encoded != OUTPUT.read_bytes():
            raise ValueError("Published preview differs from the archive export")
    else:
        write(OUTPUT, encoded)
    if args.verify_report:
        verified = subprocess.run(["node", str(ROOT / "tests/glm5/verify_b23_corpus_review.cjs")],
                                  cwd=ROOT, capture_output=True, check=True)
        report = json.loads(verified.stdout)
        report["data_sha256"] = digest(OUTPUT)
        write(REPORT, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Verified {pack['width']} profiles and {len(pack['rows'])} whole-layer slices; {len(encoded)} bytes.")
