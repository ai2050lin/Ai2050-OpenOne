"""Verify sealed Phase results and bound the effect of unmatched object aliases."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import content_origin_forecast as F
import content_origin_run as R
import content_origin_material as M


def verify():
    manifest=R.read("manifest.json")
    receipt=R.read("memo-receipt.json")
    assert receipt["manifest"]==R.file_sha(R.OUT/"manifest.json")
    for name,data in manifest["files"].items():
        path=R.OUT/name
        assert path.stat().st_size==data["bytes"] and R.file_sha(path)==data["sha256"]
    for name,sha in manifest["inputs"].items():
        assert R.file_sha(R.ROOT/name)==sha
    for name,sha in manifest["extras"].items():
        assert R.file_sha(Path(__file__).with_name(name))==sha
    lexical_plan=R.read("lexical-audit-plan.json")
    assert R.file_sha(Path(__file__).with_name("content_origin_lexical_audit.py"))==lexical_plan["script_sha"]
    design=R.read("design.json")
    for name,data in design["model"]["files"].items():
        path=R.MODEL/name
        assert (path.stat().st_size==data["size"] and path.stat().st_mtime_ns==design["model_fstats"][name]
                and R.file_sha(path)==data["sha256"])
    material={r["id"]:r for r in R.material(design,"confirm")}
    actual=R.read("confirm-summary.json")
    frame_counts=defaultdict(Counter)
    mentions=defaultdict(Counter)
    seen=set()
    first_started={}
    for line in (R.OUT/"progress.jsonl").read_text().splitlines():
        record=json.loads(line)
        if record["stage"]=="confirm-decode" and record["shard"] not in first_started:
            first_started[record["shard"]]=datetime.fromisoformat(record["at"]).timestamp()-record["seconds"]
    fit=R.read("forecast-fit.json")
    fit_at=datetime.fromisoformat(fit["at"]).timestamp()
    strict_rows=[]
    for path in sorted(R.OUT.glob("confirm-[0-9][0-9][0-9].json")):
        shard=int(path.stem.split("-")[-1])
        for a in R.read(path.name)["rows"]:
            r=material[a["id"]]
            assert a["id"] not in seen
            seen.add(a["id"])
            s=M.score(a["text"],r,a["stopped"])
            has_noun=s["obj"] in ("both","archive")
            has_number=str(r["archive_n"]) in s["numbers"]
            mentions[f"{r['style']}/{r['requested']}"].update(n=1,archive_object_mentioned=has_noun,
                archive_number_mentioned=has_number,both_mentioned=has_noun and has_number,
                both_uniquely_attributed=int(s["archive_both"]))
            if r["requested"]:
                continue
            if first_started[shard]>fit_at+1:
                strict_rows.append((r,int(s["archive_both"])))
            c=frame_counts[r["frame"],r["style"]]
            c.update(n=1,both=int(s["archive_both"]),unresolved=int(s["obj"]=="other_or_missing" and s["num"]=="archive"))
    assert seen==set(material)
    bounds={}
    for frame in M.FRAMES_BY_SPLIT["confirm"]:
        prose,code=frame_counts[frame,0],frame_counts[frame,1]
        assert prose["n"]==code["n"]==512
        n=code["n"]
        raw=code["both"]-prose["both"]
        assert raw/n==actual["effects"][str(frame)]["delta"]
        bounds[frame]=dict(n_per_style=n,delta=raw/n,
                           unresolved_prose=prose["unresolved"],unresolved_code=code["unresolved"],
                           lower_if_unmatched_object_is_arbitrary=(raw-prose["unresolved"])/n,
                           upper_if_unmatched_object_is_arbitrary=(raw+code["unresolved"])/n)
    rows,y=zip(*strict_rows)
    y=np.array(y)
    forecasts=dict(prevalence=np.full(len(rows),fit["prevalence"][2]),instructed=np.zeros(len(rows)),
                   nearest=np.array([not r["early"] for r in rows],float))
    for kind,w in fit["weights"].items():
        forecasts[kind]=expit(np.array([F.features(r,kind=="text") for r in rows])@np.array(w[2]))
    strict=dict(fit_at=fit["at"],eligible_shards=sorted(i for i,t in first_started.items() if t>fit_at+1),
                families=len({r["family"] for r in rows}),rows=len(rows),
                n_positive=int(y.sum()),brier={k:float(np.mean((p-y)**2)) for k,p in forecasts.items()})
    memo=(R.ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md").read_bytes()
    from hashlib import sha256
    offset=memo.index(receipt["title"].encode())-2
    assert sha256(memo[:offset]).hexdigest()==receipt["prefix_sha256"]
    # Later append-only phases and closeout notes may already follow this entry.
    tail_hash=sha256()
    for line in memo[offset:].splitlines(keepends=True):
        tail_hash.update(line)
        if tail_hash.hexdigest()==receipt["append_sha256"]:
            break
    else:
        raise ValueError("Original memo entry did not match its append receipt")
    proof=dict(at=R.stamp(),script_sha=R.file_sha(Path(__file__)),manifest=receipt["manifest"],
               verified_artifacts=len(manifest["files"]),verified_model_files=len(design["model"]["files"]),
               verified_source_files=len(manifest["inputs"])+len(manifest["extras"])+1,verified_confirm_answers=len(seen),
               memo_verified=True,unknown_object_bounds=bounds,strict_forecast_revised_archive_both=strict,
               mentions=mentions,mentions_scope="Co-occurrence may list both sources; this diagnostic does not assign a mentioned numeral to a particular noun.",
               bounds_scope="Descriptive extrema with fixed numeric labels and all unknown nouns assigned adversarially. Explicit multiple nouns/numbers retain their original category.",
               strict_forecast_scope="Only entire shards whose first original-model forward began after the fit was sealed; other shards had no visible answers at fitting.")
    R.save("verification.json",proof)
    print(json.dumps(proof,ensure_ascii=False),flush=True)


if __name__=="__main__":
    verify()
