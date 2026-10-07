"""Read-only evidence audit; no model inference or historical result rewriting."""
import argparse
import hashlib
import json
import re
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_framework_audit_20260923"
MEMO = ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md"
PHASES = [2806,2807,3018,3019,3035,3038,3039,3040,3041,3042,3043,3044,
          3055,3056,3057,3067,3072,3074,3075,3076,3078,3079,3080,
          3081,3082,3083,3093,3094,3095,3096,3097,3098,3099,3100,3104]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    source = MEMO.read_text(encoding="utf-8-sig")
    matches = list(re.finditer(r"^## Phase (\d+):.*$", source, re.M))
    index = {}
    for j, match in enumerate(matches):
        end = matches[j+1].start() if j+1 < len(matches) else len(source)
        index[int(match[1])] = dict(
            line=source.count("\n", 0, match.start())+1,
            text=source[match.start():end].strip())
    selected = {str(i): index[i] for i in PHASES if i in index}
    write("memo_index.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source=str(MEMO.relative_to(ROOT)),
        sha256=sha(MEMO), headings=len(matches), distinct_phase_ids=len(index),
        first=min(index), last=max(index), selected=selected,
        scope="Claim-directed audit, not a re-execution of all historical experiments."))
    (OUT/"evidence_excerpts.md").write_text("\n\n".join(
        f"<!-- Original line {v['line']} -->\n{v['text']}" for v in selected.values()), encoding="utf-8")
    write("audit_execution.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
        mode="Document, implementation and saved-result review; no GPU model run",
        external_primary_sources=["https://arxiv.org/abs/1706.03762",
                                  "https://arxiv.org/abs/1910.07467"]))
    return index

def raw_results():
    base=ROOT/"tests/glm5/result/rdc_query_construction_20260913"
    records=[]
    picked=[3018,3019,3035,3040,3041,3057,3067,3072,3074,3076,3082,3093,3094,3098,3100]
    for phase in picked:
        for path in sorted((base/f"phase{phase}").glob("*/result.json")):
            data=json.loads(path.read_text(encoding="utf-8"))
            record=dict(phase=phase,path=str(path.relative_to(ROOT)),sha256=sha(path),
                        data=data)
            records.append(record)
            print(phase, path.parent.name, "keys", list(data))
    write("saved_result_inventory.json",records)
    return records

def checks():
    base=ROOT/"tests/glm5/result/rdc_query_construction_20260913"
    def arr(i):
        p=next((base/f"phase{i}").glob("*/*.npz"))
        return p,np.load(p,allow_pickle=False)
    p40,d40=arr(3040)
    means=np.stack([d40["V3"][d40["occ_w"]==j].mean(axis=0) for j in range(26)])
    _,s,vt=np.linalg.svd(means,full_matrices=False)
    basis=vt[:int((s>1e-8*s[0]).sum())].T
    energy=((d40["V3"]@basis)**2).sum(axis=1)/(d40["V3"]**2).sum(axis=1)
    err40=float(np.max(np.abs(energy-d40["ew3"])))
    assert err40<1e-10
    p35,d35=arr(3035)
    early=d35["pA_early"]-d35["pA0"][:,None]
    late=d35["pA_late"]-d35["pA0"][:,None]
    attenuation=np.abs(early).mean(axis=1)/np.maximum(np.abs(late).mean(axis=1),1e-30)
    err35=float(np.max(np.abs(attenuation-d35["atten"])))
    assert err35<1e-10
    p98,d98=arr(3098)
    frac={}
    for model in ["4B","14B"]:
        kept=[]
        for pair in ["AB","AC","BC"]:
            f=d98[f"F2S_{model}_{pair}"]
            for ci in range(3):
                med=np.median(f[ci*8:(ci+1)*8],axis=0)
                if abs(med[2]-med[0])>=0.10:
                    a,m=med[1]-med[0],med[2]-med[1]
                    kept.append(abs(m)/(abs(a)+abs(m)))
        frac[model]=dict(active_groups=len(kept),median=float(np.median(kept)))
    r98=json.loads(p98.with_name("result.json").read_text(encoding="utf-8"))
    for model in frac:
        assert abs(frac[model]["median"]-r98["stats"]["frac"][f"{model}_med_frac_mlp"])<1e-12
    checks=dict(
        phase3040=dict(npz_sha256=sha(p40),observations=92,types=26,coordinates=128,
                       median_projection_energy=float(np.median(energy)),max_replay_error=err40,
                       scope="In-sample span of word means; not a fraction of semantic information."),
        phase3035=dict(npz_sha256=sha(p35),tags=11,
                       median_probability_effect_ratio=float(np.median(attenuation)),
                       reciprocal=float(1/np.median(attenuation)),max_replay_error=err35,
                       scope="Ratio of early/late target probability effects across matched dosage grids, not generic perturbation norm attenuation."),
        phase3098=dict(npz_sha256=sha(p98),active_group_fractions=frac,
                       scope="Fraction of absolute two-substep median cosine changes, not a percentage of total language computation."),
        no_model_forward=True)
    write("numerical_rechecks.json",checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--show", default="")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--checks", action="store_true")
    a=p.parse_args()
    idx=collect()
    if a.checks:
        checks()
    elif a.raw:
        raw_results()
    elif a.show:
        for key in map(int, a.show.split(",")):
            print(idx[key]["text"])
    else:
        print(json.dumps(dict(phases=len(idx), selected=len(PHASES), output=str(OUT))))
