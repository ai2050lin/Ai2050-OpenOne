"""Describe paired continuations already exposed in Phase 2978. No model calls."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import checked_dir

SOURCE = ROOT / "tests/glm5/result/paradigm_window_20261118_v2"
OUT = ROOT / "tests/glm5/result/paradigm_archive_content_audit_20261118"
# Literal output patterns noticed in opened data. Broader translations are unscored.
PATTERNS = {0: (9, 'print("九艘船")'), 1: (7, 'print("七张地图")')}


def audit():
    manifest = json.loads((SOURCE / "manifest.json").read_bytes())["files"]
    provenance = {}

    def verified(name):
        path = SOURCE / name
        digest = sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[name]["sha256"], name
        provenance[name] = digest
        return path

    design = json.loads(verified("design.json").read_bytes())
    tokenizer_path = Path(design["provenance"]["model_path"]) / "tokenizer.json"
    tokenizer_sha = sha256(tokenizer_path.read_bytes()).hexdigest()
    assert tokenizer_sha == design["model"]["files"]["tokenizer.json"]["sha256"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pairs = defaultdict(dict)
    index = json.loads(verified("generation-files.json").read_bytes())
    for item in index["files"]:
        path = verified(item["file"])
        assert item["sha256"] == provenance[path.name]
        for view in json.loads(path.read_bytes())["views"]:
            if view["arm"] in ("code", "prose"):
                key = view["family"], view["marker"], view["layout"]
                assert view["arm"] not in pairs[key]
                pairs[key][view["arm"]] = view

    banks = {}
    for name in sorted(k for k in manifest if k.startswith("confirm-") and k.endswith(".npz")):
        with np.load(verified(name), allow_pickle=False) as data:
            for family, bank in zip(data["family"], data["bank"], strict=True):
                assert banks.setdefault(str(family), int(bank)) == int(bank)

    totals, by_family, divergence, prefixes, details = Counter(), defaultdict(Counter), Counter(), Counter(), []
    for (family, marker, layout), arms in sorted(pairs.items()):
        code, prose = arms["code"], arms["prose"]
        assert (code["gold"], code["foil"]) == (prose["gold"], prose["foil"])
        assert code["stopped"] and prose["stopped"]
        for view in (code, prose):
            prefixes[tokenizer.decode(view["tokens"][:2], skip_special_tokens=False)] += 1
        count, pattern = PATTERNS[banks[family]]
        events = dict(pairs=True, same_first=code["tokens"][0] == prose["tokens"][0],
                      changed=code["tokens"] != prose["tokens"],
                      code_exact=code["text"] == pattern, prose_exact=prose["text"] == pattern,
                      disjoint=count not in (code["gold"], code["foil"]))
        for arm in ("code", "prose"):
            events[arm + "_exact_disjoint"] = events["disjoint"] and events[arm + "_exact"]
        totals.update(events)
        by_family[family].update(events)
        first = next((i for i, (a, b) in enumerate(zip(code["tokens"], prose["tokens"]), 1) if a != b), None)
        if first is None and events["changed"]:
            first = min(len(code["tokens"]), len(prose["tokens"])) + 1
        divergence[str(first) if first is not None else "identical"] += 1
        details.append(dict(family=family, marker=marker, layout=layout, archive_count=count,
                            gold=code["gold"], foil=code["foil"], first_difference=first,
                            code=code["text"], prose=prose["text"], **events))
    assert len(by_family) == 16 and all(v["pairs"] == 4 for v in by_family.values())
    return dict(status="exploratory reanalysis of exposed families", model_runs=0,
                strict_patterns={k: v[1] for k, v in PATTERNS.items()},
                source_manifest=sha256((SOURCE / "manifest.json").read_bytes()).hexdigest(),
                tokenizer_sha256=tokenizer_sha, first_two_tokens_decoded=dict(prefixes),
                sources=provenance, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                totals=dict(totals), families=len(by_family), first_difference_one_based=dict(divergence),
                by_family=dict(by_family), details=details)


if __name__ == "__main__":
    result = audit()
    output = checked_dir(OUT, required=256 * 1024) / "analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["totals"], result["first_difference_one_based"])
