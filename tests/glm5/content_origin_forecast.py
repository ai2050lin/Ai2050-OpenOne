"""Discovery-only input forecasts for which archive fields appear during free generation."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import argparse
import sys
import time
from zlib import crc32

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import content_origin_run as R
import content_origin_material as M

OUTCOMES=("archive_object","archive_number","archive_both")


def features(r,ngrams):
    factors=[int(r[k]) for k in ("style","requested","early","brackets")]
    vector=[1.,*factors,*[a*b for a,b in combinations(factors,2)]]
    if ngrams:
        body=r["prompt"].lower()
        bag=Counter(crc32(body[j:j+3].encode()) % 128 for j in range(len(body)-2))
        norm=sum(v*v for v in bag.values())**.5 or 1.
        vector.extend(bag[j]/norm for j in range(128))
    return vector


def collect(split):
    rows=R.material(R.read("design.json"),split)
    byid={r["id"]:r for r in rows}
    grades={}
    binding=R.file_sha(R.OUT/"design.json")
    for path in sorted(R.OUT.glob(split+"-[0-9][0-9][0-9].json")):
        d=R.read(path.name)
        assert d["binding"]==binding
        for answer in d["rows"]:
            q=byid[answer["id"]]
            assert answer["id"] not in grades
            sc=M.score(answer["text"],q,answer["stopped"])
            grades[answer["id"]]=(sc["obj"]=="archive",sc["num"]=="archive",sc["archive_both"])
    assert len(rows)==len(grades)
    return rows,np.asarray([grades[r["id"]] for r in rows],dtype=float)


def regress(x,y):
    penalty=np.full(x.shape[1],2.)
    penalty[0]=.01
    def objective(w):
        z=x@w
        return (np.logaddexp(0.,z).sum()-y@z+.5*np.dot(penalty*w,w),
                x.T@(expit(z)-y)+penalty*w)
    sol=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-12})
    assert sol.success,sol.message
    return sol.x.tolist()


def freeze():
    assert not (R.OUT/"forecast-plan.json").exists() and not list(R.OUT.glob("confirm-*.json"))
    R.save("forecast-plan.json",dict(at=R.stamp(),code=R.file_sha(Path(__file__)),design=R.file_sha(R.OUT/"design.json"),
            outcomes=OUTCOMES,scope="Secondary; fit all discovery families only. Freeze before first confirmation shard; no changes to main behavior gate.",
            models="Smoothed prevalence; instructed source; nearest record; 4 input factors + pair interactions, logistic L2=2; same + 128-bin hashed char trigrams. Threshold .5, report Brier and balanced accuracy+both recalls. No heldout training or model selection."))


def train():
    plan=R.read("forecast-plan.json")
    assert plan["code"]==R.file_sha(Path(__file__)) and not (R.OUT/"forecast-fit.json").exists()
    assert (R.OUT/"discovery-summary.json").exists() and not list(R.OUT.glob("confirm-[0-9]*.json"))
    rows,y=collect("discovery")
    weights={}
    for kind in ("factors","text"):
        x=np.array([features(r,kind=="text") for r in rows])
        weights[kind]=[regress(x,y[:,j]) for j in range(3)]
    R.save("forecast-fit.json",dict(at=R.stamp(),plan=R.file_sha(R.OUT/"forecast-plan.json"),
           discovery=R.file_sha(R.OUT/"discovery-summary.json"),prevalence=((y.sum(0)+.5)/(len(y)+1)).tolist(),weights=weights))


def evaluate():
    import content_origin_forecast_frozen_v1 as original
    fit=R.read("forecast-fit.json")
    d=R.read("forecast-plan.json")
    implementation=original if d["code"]==R.file_sha(Path(original.__file__)) else sys.modules[__name__]
    assert d["code"]==R.file_sha(Path(implementation.__file__)) and fit["plan"]==R.file_sha(R.OUT/"forecast-plan.json")
    rows,y=implementation.collect("confirm")
    result={}
    requested=np.asarray([r["requested"] for r in rows])
    forecasts=dict(prevalence=np.broadcast_to(fit["prevalence"],y.shape),
                   instructed=np.repeat(requested[:,None],3,axis=1),
                   nearest=np.repeat(np.array([not r["early"] for r in rows])[:,None],3,axis=1))
    for kind,w in fit["weights"].items():
        forecasts[kind]=expit(np.asarray([implementation.features(r,kind=="text") for r in rows])@np.asarray(w).T)
    for name,p in forecasts.items():
        for panel,mask in (("all",np.ones(len(y),bool)),("revised",requested==0),("archive",requested==1)):
            for j,target in enumerate(OUTCOMES):
                actual=y[mask,j].astype(bool)
                pred=p[mask,j]>=.5
                tp=float(pred[actual].mean()) if actual.any() else None
                tn=float((~pred[~actual]).mean()) if (~actual).any() else None
                result[f"{name}/{panel}/{target}"]=dict(n=len(actual),positive=int(sum(actual)),positive_recall=tp,
                    negative_recall=tn,balanced=(tp+tn)/2 if tp is not None and tn is not None else None,
                    brier=float(np.square(p[mask,j].astype(float)-actual.astype(float)).mean()))
    R.save("forecast-evaluation.json",dict(at=R.stamp(),fit=R.file_sha(R.OUT/"forecast-fit.json"),metrics=result,
           training_source=d["code"],evaluation_source=R.file_sha(Path(__file__)),
           repair="Cast the nearest-record Boolean predictions and Boolean outcomes to float for numeric Brier subtraction. Frozen features, predictions, fit, gates and primary generation outputs retained."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","train","wait-train","evaluate"))
    action=p.parse_args().action
    R.setup()
    if action=="wait-train":
        while not (R.OUT/"discovery-summary.json").exists():
            time.sleep(20)
        train()
    else:
        dict(freeze=freeze,train=train,evaluate=evaluate)[action]()
