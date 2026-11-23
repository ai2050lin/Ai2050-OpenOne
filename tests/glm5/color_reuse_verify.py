"""Verify a completed color comparison package against its frozen material and receipt."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.dist_protocol.test_archive import digest


def verify_archive(directory):
    def read(filename):
        return json.loads((directory/filename).read_text(encoding="utf-8-sig"))
    package=directory/"study-package"
    registered=read("study-desktop.json")
    submission=read("study-package/submission.json")
    receipt=read("study-receipt.json")["receipt"]
    run=registered["run"]["id"]
    if (run!=submission["run_id"] or run!=receipt["run_id"] or not receipt["readback_verified"]
            or receipt["archive_state"]!="server_verified" or receipt["missing_core_roles"]):
        raise ValueError("unverified server receipt")
    if digest(submission["final"])!=receipt["manifest_sha256"]:
        raise ValueError("server manifest differs")
    for entry in submission["final"]["artifacts"]:
        with (package/entry["name"]).open("rb") as src:
            if hashlib.file_digest(src,"sha256").hexdigest()!=entry["sha256"]:
                raise ValueError("archive artifact differs: "+entry["name"])
    return (package, run, len(submission["final"]["artifacts"]), read("study-questions.json"),
            read("study-package/conclusion.json"))


def verify(directory):
    package,run,files,question_pack,result=verify_archive(directory)
    material=question_pack["rows"]
    with (package/"case_results.jsonl").open(encoding="utf8") as f:
        cases=[json.loads(s) for s in f]
    if result.get("recovery"):
        donor=directory.parent/"interrupted-evidence"/"case_results.jsonl"
        with donor.open(encoding="utf8") as f:
            prior=[json.loads(s) for s in f]
        if len(prior)!=result["recovery"]["inherited"] or cases[:len(prior)]!=prior:
            raise ValueError("the inherited rows changed")
    arms=defaultdict(list)
    for r in cases:
        arms[r["id"]].append(r)
    if set(arms)!=set(range(len(material))) or len(cases)!=result["forwards_scored"]:
        raise ValueError("case coverage mismatch")
    prefixes=defaultdict(set)
    for q in material:
        variants=arms[q["id"]]
        expected={"discovery":1,"calibration":19,"confirm":4,"reference":1,"tail":1,"corpus":1}[q["stage"]]
        if len(variants)!=expected or len({(r["arm"],r["block"],r["budget"]) for r in variants})!=expected:
            raise ValueError("duplicate or absent execution for question "+str(q["id"]))
        if q["stage"]=="confirm":
            prefixes[q["item"]].add(tuple(q["tokens"][:q["pos"]+1]))
    if any(len(p)!=1 for p in prefixes.values()):
        raise ValueError("reference or task leaks through the first-item boundary")
    verify_pairs=Counter()
    primary=defaultdict(list)
    examples=[]
    for q in material:
        if (q["stage"],q["task"])!=("confirm","compare"):
            continue
        rs={r["arm"]:r for r in arms[q["id"]]}
        cell=q["color"]+"/"+q["refcolor"]
        verify_pairs[cell]+=rs["base"]["correct"]
        if q["color"]=="red":
            def score(arm):
                a,b=rs[arm]["logp"]
                return a-b
            primary[q["refcolor"]].append(score("cut")-score("ctrl"))
            if q["item"]=="a ripe strawberry" and q["ref"] in (
                    "a ripe beefsteak tomato","a ripe Cavendish banana","a cloudless summer sky at noon"):
                examples.append([q["ref"], rs["base"]["pred"], rs["cut"]["pred"],score("base"),score("cut"),score("ctrl")])
    if any(verify_pairs[k]!=v["correct"] for k,v in result["comparison"].items()):
        raise ValueError("published accuracies differ from cases")
    for color,values in primary.items():
        if abs(sum(values)/len(values)-result["color_delta"][color]["mean"])>1e-6:
            raise ValueError("primary color contrast does not reproduce")
    return dict(run=run,files=files,questions=len(material),executions=len(cases),
                independent_first_items=len(prefixes),cells=dict(verify_pairs),examples=examples)


def verify_continuations(directory):
    package, run, files, questions, result = verify_archive(directory)
    upstream = directory.parent / "completion/study-package/case_results.jsonl"
    if hashlib.sha256(upstream.read_bytes()).hexdigest() != questions["first_token_source_sha256"]:
        raise ValueError("first-token source differs")
    initial = {(r["id"], r["arm"]): r for line in upstream.read_text(encoding="utf8").splitlines()
               if (r := json.loads(line))["stage"] == "confirm" and r["task"] == "compare"}
    cases = [json.loads(line) for line in (package/"case_results.jsonl").read_text(encoding="utf8").splitlines()]
    lookup = {(r["id"], r["arm"]): r for r in cases}
    if len(cases) != 1296 or lookup.keys() != initial.keys():
        raise ValueError("continuations are missing or duplicated")
    pattern = re.compile(r"(?<![A-Za-z])(same|different)(?![A-Za-z])", re.I)
    forms = Counter()
    cells = defaultdict(Counter)
    changes = []
    restored = 0
    for key, row in lookup.items():
        match = pattern.search(row["text"])
        answer = match.group().lower() if match else None
        if (row["generated"][0] != initial[key]["top"] or row["first_text"] != initial[key]["top_text"]
                or not 1 <= len(row["generated"]) <= 12 or row["answer"] != answer):
            raise ValueError("generated answer inconsistent with saved tokens or text")
        baseline = lookup[(row["id"], "base")]
        correct = answer == initial[key]["gold"]
        if correct != row["correct"]:
            raise ValueError("incorrect label in stored comparison")
        cell = f'{row["color"]}/{row["refcolor"]}'
        counts = cells[(row["arm"], cell)]
        counts.update(n=1, correct=int(correct), new_error=int(baseline["correct"] and not correct),
                      repaired=int(not baseline["correct"] and correct))
        if row["arm"] == "base":
            forms[row["text"]] += 1
        if row["arm"] == "restore":
            restored += row["generated"] == baseline["generated"]
        if row["answer"] != baseline["answer"]:
            changes.append([row["arm"], row["item"], row["ref"], baseline["text"], row["text"], row["gold"]])
    if any(dict(counts) != result["outcomes"][arm]["cells"][cell] for (arm, cell), counts in cells.items()):
        raise ValueError("recounted cells differ")
    if restored != result["exact_restored_text"] or any(
            sum(change[0] == arm for change in changes) != value["changed"]
            for arm, value in result["outcomes"].items()):
        raise ValueError("restored or changed answers differ")
    return dict(run=run, files=files, executions=len(cases), baseline_forms=forms,
                exact_restored=restored, changes=changes)


if __name__=="__main__":
    study = ROOT/"tests/glm5/result/color_reuse_v1"
    print(json.dumps({"registered_study": verify(study/"completion"),
                      "continuations": verify_continuations(study/"greedy")}, indent=2))
