"""Embed opened historical results in the standalone design-review HTML."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "frontend/src/components/v8"
RESULT = ROOT / "tests/glm5/result"
PREVIEW = ROOT / "research/glm5/docs/previews/research-evidence-workbench.html"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_cues():
    runs = read(UI / "threeCueEvidence.json")["runs"]
    assert [r["id"] for r in runs] == ["clt_b26_three_cues_v3", "clt_b26_competition_v1"]
    assert len({r["snapshot"] for r in runs}) == 1
    name_sets = []
    fields = ("id", "snapshot", "families", "provenance", "model")
    for run in runs:
        folder = RESULT / run["id"]
        for file, identity in run["provenance"].items():
            assert hashlib.sha256((folder / file).read_bytes()).hexdigest() == identity, file
        families = run["families"]
        assert len(families) == 128
        name_sets.append({name for family in families for name in family["names"]})
        shards = [read(folder / f"confirm-{index:02}.json")["scores"] for index in range(4)]
        scores = [{r["id"]: r for r in rows} for rows in shards]
        for family in families:
            assert len(family["cases"]) == 12
            for case in family["cases"]:
                raw = scores[family["id"] // 32][case["id"]]
                assert raw["top1"] == case["tokenId"] and raw["top1_correct"] == case["correct"]
        for i, metric in enumerate(run["natural"]):
            cases = [c for f in families for c in f["cases"] if c["rule"] == i]
            assert len(cases) == 512
            assert all(c["mass"].count(max(c["mass"])) == 1 for c in cases)
            hits = sum(c["mass"].index(max(c["mass"])) == c["gold"] for c in cases)
            assert hits / len(cases) == metric["all_five_gold"]
    assert name_sets[0].isdisjoint(name_sets[1])
    return [{key: run[key] for key in fields} for run in runs]


def main():
    color = RESULT / "color_reuse_v1/greedy/study-package"
    color_rows = [json.loads(line) for line in (color / "case_results.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len({(row["id"], row["arm"]) for row in color_rows}) == 324 * 4 == len(color_rows)
    color_case = next(row for row in color_rows if row["id"] == 44 and row["arm"] == "base")
    path = read(UI / "firstRecordPathStatus.json")
    receipt = read(RESULT / "routine/run-t7dfn8_7/server-receipt.json")
    assert receipt["run_id"] == path["runId"] and receipt["receipt_sha256"] == path["receiptSha256"]
    protocol = ROOT / "shared/dist_protocol"
    tree = read(protocol / "lang_map/lang_map.json")
    questions = read(protocol / "test_records/direction_questions.json")

    def leaf(item, plans):
        return {"id": item["id"], "name": item["name"], "plan": plans[item["id"]]}

    branches = [{"name": d["name"], "groups": [
        {"name": g["name"], "items": [leaf(item, questions["items"]) for item in g["items"]]}
        for g in d["subs"]]} for d in tree["domains"]]
    branches.append({"name": "横切条件", "groups": [{"name": "跨任务变量", "items": [
        leaf(item, questions["dimensions"]) for item in tree["dims"]]}]})
    h0_folder = ROOT / "research/mainline/H0"
    h0 = read(h0_folder / "report_v5.json")
    assert [sum(cell["gate"] for cell in h0["langs"][lang]["cells"]) for lang in ("zh", "en")] == [32, 31]
    shark = [r for r in read(h0_folder / "run_v5.json")["langs"]["zh"]["records"]
             if r["entity"] == "aquatic.shark" and r["cell"][1:] == ["first", 2, 0, "single"]]
    assert {(r["cell"][0], r["choice"]) for r in shark} == {("doc", "purple"), ("real", "gray")}
    source_files = {322: h0_folder / "report_v5.json", 2997: RESULT / "routine/run-t7dfn8_7/server-receipt.json",
                    3008: color / "conclusion.json"}
    source_hashes = {p: hashlib.sha256(f.read_bytes()).hexdigest() for p, f in source_files.items()}
    payload = {"cues": checked_cues(), "path": path, "color": read(color / "conclusion.json"),
               "colorCase": color_case, "colorRows": color_rows, "branches": branches,
               "sourceHashes": source_hashes}
    start = '<script type="application/json" id="evidence-data">'
    before, marker, tail = PREVIEW.read_text(encoding="utf-8").partition(start)
    assert marker and "</script>" in tail
    after = tail[tail.index("</script>"):]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
    PREVIEW.write_text(before + marker + encoded + after, encoding="utf-8", newline="\n")
    print(f"Embedded two verified cue batches (3072 cases), one path study and one color study. {PREVIEW.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()
