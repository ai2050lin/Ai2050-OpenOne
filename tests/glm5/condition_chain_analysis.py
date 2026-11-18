"""Family-level preregistered comparisons and retained failures."""
from collections import defaultdict
import sys

import numpy as np
from scipy.stats import t, ttest_1samp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from scipy.sparse import csr_matrix, hstack

import condition_chain_run as R
import condition_chain_materials as M


def fp(bits):
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(.975,n-1) * sd / np.sqrt(n))
    return dict(n=n,mean=mean,ci95=[mean-half,mean+half],dz=mean/sd if sd else None,
                p=float(ttest_1samp(values,0).pvalue) if sd else (1.0 if mean==0 else 0.0))


def margin(out, j):
    scores = out["option_logits"]
    return scores[j] - max(x for i,x in enumerate(scores) if i != j)


def mapped(model, split, panel):
    rows = R.load(R.OUT / f"materials-{split}-{panel}.json")
    record = R.load(R.OUT / f"{model}-{split}-{panel}.json")
    results = record["results"]
    assert all(r["id"]==m["id"] for r,m in zip(rows,results,strict=True))
    assert R.cell_pass(rows,results,R.THRESHOLDS["behavior"]) == record["gate"] or json_cells(R.cell_pass(rows,results,R.THRESHOLDS["behavior"])) == record["gate"]
    return rows,results,record["gate"]


def json_cells(rows):
    return [{**r,"cell":list(r["cell"])} for r in rows]


def paired_chain(rows, results):
    by_family = defaultdict(dict)
    for r, obs in zip(rows,results,strict=True):
        by_family[(r["task"],r["family"])][tuple(r["factors"])] = r,obs
    effects = defaultdict(list)
    for (task,_), arms in by_family.items():
        family = defaultdict(list)
        for arm,(r,out) in arms.items():
            for i,name in enumerate(("select","update","distract","report")):
                if arm[i]:
                    continue
                other = list(arm)
                other[i] = 1
                r2,out2 = arms[tuple(other)]
                family[name+"-accuracy"].append(int(out2["correct"])-int(out["correct"]))
                family[name+"-margin"].append(margin(out2,r2["answer"])-margin(out,r["answer"]))
                if name in ("select","update"):
                    old = r["answer"]; new = r2["answer"]
                    assert old != new
                    m0 = out["option_logits"][old] - out["option_logits"][new]
                    m1 = out2["option_logits"][new] - out2["option_logits"][old]
                    family[name+"-contrast"].append((m0+m1)/2)
                elif name=="distract":
                    family["distract-follow"].append(
                        (out2["option_logits"][r2["option_order"].index(r2["diagnostic_index"])]
                         -out2["option_logits"][r2["option_order"].index(r["diagnostic_index"])]
                         +out["option_logits"][r["option_order"].index(r["diagnostic_index"])]
                         -out["option_logits"][r["option_order"].index(r2["diagnostic_index"])])/2)
        for name, items in family.items():
            effects[(task,name)].append(float(np.mean(items)))
    return [dict(task=task,measure=name,**stats(values)) for (task,name),values in effects.items()]


def native(split, panel):
    output = {}
    for file in sorted(R.OUT.glob(f"q4-{split}-{panel}-native-*.npz")):
        with np.load(file) as obj:
            record_keys = set(obj.files) & {"probe","embed","prefix0","G","V","A","controls"}
            arrays = {k:fp(obj[k]) for k in record_keys}
            for i,row_id in enumerate(obj["ids"]):
                values = {key:arr[i] for key,arr in arrays.items()}
                if "controls" in arrays:
                    values["selected_norm"] = obj["selected_norm"].item()
                    values["control_norms"] = obj["control_norms"].copy()
                output[str(row_id)] = values
    return output


def word_features(prefixes):
    return HashingVectorizer(n_features=4096,ngram_range=(1,2),alternate_sign=False).transform(prefixes)


def candidate_analysis(row_map, native_map, task):
    threshold = R.THRESHOLDS
    chunks = {}
    for split in ("discover","words","frames"):
        rows = [row for row in row_map[split] if row["task"]==task]
        if any(r["id"] not in native_map[split] for r in rows):
            return dict(task=task,passed=False,status=f"missing behavior-admitted native data: {split}")
        chunks[split] = rows

    def features(split):
        rows = chunks[split]
        data = [native_map[split][r["id"]] for r in rows]
        prefixes = [R.prompts(r)[0][:R.prompts(r)[1][0]] for r in rows]
        w = word_features(prefixes)
        emb = np.asarray([o["prefix0"] for o in data],dtype=np.float64)
        emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-9)
        length = np.asarray([len(s) for s in prefixes])[:,None]/256
        baseline = hstack([w, csr_matrix(emb), csr_matrix(length)],format="csr")
        target = np.asarray([r["factors"][0] for r in rows])
        g = np.asarray([o["G"][0] for o in data])
        return data,baseline,target,g

    d,base,y,g = features("discover")
    fitted = Ridge(alpha=1).fit(base,g)
    with_relation = Ridge(alpha=1).fit(hstack([base,csr_matrix(y[:,None])],format="csr"),g)
    anorm = np.sqrt(np.mean(np.square([o["A"][0] for o in d])))
    cnorm = np.sqrt(np.mean(np.square([o["controls"] for o in d]),axis=0))
    dnorm = d[0]["selected_norm"]
    cdnorm = d[0]["control_norms"]
    control_index = int(np.argmin(np.abs(np.log((cnorm+1e-8)/(anorm+1e-8))) + np.abs(np.log(cdnorm/dnorm))))
    block,unit,_ = M.CANDIDATES[task]
    match = (unit + 37*(control_index+1)) % 9728
    tests = []
    for split in ("words","frames"):
        native_data,x,y,g = features(split)
        baseline_err = np.mean((fitted.predict(x)-g)**2)
        relation_err = np.mean((with_relation.predict(hstack([x,csr_matrix(y[:,None])],format="csr"))-g)**2)
        ratio = float(relation_err/baseline_err) if baseline_err else None
        families = defaultdict(dict)
        for row,o in zip(chunks[split],native_data,strict=True):
            families[row["family"]][tuple(row["factors"])] = o
        for cue in (0,1):
            diffs = [float(a[(1,cue)]["G"][0]-a[(0,cue)]["G"][0]) for a in families.values()]
            test = stats(diffs)
            test["passed"] = (test["dz"] is not None and test["dz"]>=threshold["min_dz"] and
                              test["p"]<threshold["alpha"] and ratio is not None and ratio<=threshold["mse_ratio"])
            tests.append(dict(split=split,cue=cue,mse_ratio=ratio,baseline_mse=float(baseline_err),
                              supplemental={k:stats([float(a[(1,cue)][k][0]-a[(0,cue)][k][0]) for a in families.values()])
                                            for k in ("V","A")},**test))
    return dict(task=task,block=block,unit=unit,match=match,pass_response=all(r["passed"] for r in tests),
                match_rms_ratio=float(cnorm[control_index]/max(anorm,1e-8)),
                match_norm_ratio=float(cdnorm[control_index]/dnorm),tests=tests)


def probe_analysis(row_map, native_map):
    out = []
    for task in ("stock","calendar"):
        all_rows = {}
        for split in ("discover","words","frames"):
            all_rows[split] = [r for r in row_map[split] if r["task"]==task and r["id"] in native_map[split]]
        train = all_rows["discover"]
        if not train:
            continue
        for site in range(4):
            def data(rows,split):
                text = []
                for r in rows:
                    prompt,cuts = R.prompts(r)
                    text.append(prompt[:cuts[site]] if site < 3 else prompt)
                w = word_features(text)
                n = native_map[split]
                h = np.asarray([n[r["id"]]["probe"][site] for r in rows])
                h = h / np.maximum(np.linalg.norm(h,axis=1,keepdims=True), 1e-9)
                embedding = np.asarray([n[r["id"]]["embed"][site] for r in rows])
                return w,h,embedding
            x = data(train,"discover")
            targets = {"selected": lambda r:r["factors"][0],
                       "terminal":lambda r:r["truth"][1],
                       "report": lambda r:r["answer"]}
            for target, label in targets.items():
                models = [RidgeClassifier(alpha=1).fit(features,[label(r) for r in train]) for features in x]
                for split in ("words","frames"):
                    if not all_rows[split]:
                        continue
                    y = [label(r) for r in all_rows[split]]
                    predictions = [model.predict(features) for model,features in zip(models,data(all_rows[split],split))]
                    out.append(dict(task=task,site=site,target=target,split=split,n=len(y),
                                    accuracy=dict(zip(["word","block24","embedding_site"],
                                                      [float(np.mean(p==y)) for p in predictions]))))
    return out


def main():
    R.setup()
    R.protocol()
    screens = []
    row_map = defaultdict(dict)
    for model in M.MODELS:
        for split in M.SPLITS:
            for panel in ("chain","candidate"):
                path=R.OUT/f"{model}-{split}-{panel}.json"
                if not path.exists():
                    continue
                rows,results,gate=mapped(model,split,panel)
                if model=="q4":
                    row_map[panel][split]=rows
                totals={}
                for task in sorted({r["task"] for r in rows}):
                    keep=[(r,k) for r,k in zip(rows,results,strict=True) if r["task"]==task]
                    natural=[k["natural_correct"] for _,k in keep if k["natural_correct"] is not None]
                    totals[task]=dict(correct=sum(k["correct"] for _,k in keep),n=len(keep),
                                      gates_pass=sum(c["pass_gate"] for c in gate if c["cell"][0]==task),
                                      gates_total=sum(c["cell"][0]==task for c in gate),
                                      natural_n=len(natural),natural_correct=sum(natural))
                failure=[dict(id=r["id"],task=r["task"],factors=r["factors"],answer=r["answer"],scores=k["option_logits"],
                              natural_response=k["natural_answer"]) for r,k in zip(rows,results,strict=True)
                         if not k["correct"] or k["natural_correct"] is False]
                screens.append(dict(model=model,split=split,panel=panel,totals=totals,gates=gate,failures=failure,
                                    effects=paired_chain(rows,results) if panel=="chain" and split!="pilot" else []))
    c_native={s:native(s,"candidate") for s in ("discover","words","frames")}
    candidates=[candidate_analysis(row_map["candidate"],c_native,task) for task in M.CANDIDATES] if row_map["candidate"] else []
    if row_map["chain"]:
        probes=probe_analysis(row_map["chain"],{s:native(s,"chain") for s in ("discover","words","frames")})
    else:
        probes=[]
    R.write("analysis.json",dict(run_id=R.RUN,plan_sha=R.file_sha(R.OUT/"plan.json"),screens=screens,
                               candidates=candidates,probes=probes,behavior_effect_scope="descriptive family intervals"))
    print("Analysis saved",[(c["task"],c.get("pass_response"),c.get("status")) for c in candidates],flush=True)


if __name__=="__main__":
    main()
