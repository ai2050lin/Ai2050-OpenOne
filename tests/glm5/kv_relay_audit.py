"""Recompute holdout predictions, branch interventions, and original-dose rescue."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kv_relay_native as N
import kv_relay_run as C
import kv_relay_study as S
from kv_address_audit import family_stats


A=C.A


def log(a):
    return np.log(np.maximum(a,1e-12))


def rmse(a,axes):
    return np.sqrt((a.astype(np.float64)**2).mean(axis=axes))


def current_arms(record):
    return np.stack([record[:,i,:,slot] for i,(slot,_) in enumerate(N.CONDITIONS)],axis=1)


def cap_flags(natural,changed):
    target=np.minimum(natural.max(-1),2*natural.min(-1))
    g=target[:,None]/np.maximum(current_arms(changed),1e-12)
    return g>2.05+2e-6,g


def diagnostic(actual,expected):
    e=actual-expected
    return dict(rmse=rmse(e,(0,1)).tolist(),maxerr=float(np.max(abs(e))))


def file_arrays(f,e):
    files={}
    for name,h in sorted(f["shards"].items()):
        assert S.sha(name)==h
        paired=name.replace("forecast-","edited-")
        assert S.sha(paired)==e["shards"][paired]
        with np.load(C.OUT/name) as a, np.load(C.OUT/paired) as b:
            assert str(b["forecast"])==e["forecast"]
            assert np.array_equal(a["ids"],b["ids"]) and np.array_equal(a["good"],b["good"])
            assert np.array_equal(a["full_qk"][a["good"]],b["actual-qk"][a["good"]])
            for key in set(a.files)|set(b.files):
                if key in ("forecast","dose-labels"):
                    continue
                data=a[key] if key in a else b[key]
                files.setdefault(key,[]).append(data)
    return {key:np.concatenate(v) for key,v in files.items()}


def summary():
    from transformers import GenerationConfig
    p,freeze=S.eligible_freeze()
    forecasts=A.load("forecast.json")
    done=A.load("edited.json")
    assert forecasts["freeze"]==S.sha("freeze.json") and done["forecast"]==S.sha("forecast.json")
    tok=A.Base.tok()
    eos=GenerationConfig.from_pretrained(A.Base.MODEL,local_files_only=True).eos_token_id
    stub=object.__new__(A.Task)
    stub.tok=tok
    stub.eos=set(eos if isinstance(eos,list) else [eos])
    stub.letters={c:tok.encode(" "+c,add_special_tokens=False)[0] for c in "PQ"}
    counts={}
    for phase in ("discover","confirm"):
        rows=A.load(f"material-{phase}.json")
        behavior=A.load(f"behavior-{phase}.json")
        assert behavior["passed"]
        for row,obs in zip(rows,behavior["rows"],strict=True):
            x,m=A.encoded(tok,row)
            assert stub.parse(row,x,m,obs["tokens"])==obs
        counts[phase]=behavior["cells"]
    by={r["id"]:r for r in rows}
    data=file_arrays(forecasts,done)
    include=data["good"]
    assert int(include.sum())==forecasts["n"]==done["n"]
    data={k:x[include] for k,x in data.items()}
    ids=data["ids"].tolist()
    vs=[by[x] for x in ids]
    familyidx=defaultdict(list)
    for i,r in enumerate(vs):
        familyidx[r["family"]].append(i)
    assert all(len(b)==8 for b in familyidx.values())
    alpha=data["source-alpha"]
    anchor=data["natural"]
    before=log(anchor[:,None])
    after=log(alpha)
    actual=after-before
    target,need=cap_flags(anchor,alpha)
    native=data["trio-gain"][:,:,:2]>2.05+2e-6
    assert np.array_equal(target,native)
    fullerr=float(np.abs(data["full"]-alpha).max())
    assert fullerr<=p["prediction"]["local_max_alpha_tolerance"]
    assert np.abs(data["qk-block-alpha"]-anchor[:,None]).max()<3e-6
    assert np.abs(data["qk-restore-alpha"]-alpha).max()<3e-6
    assert np.abs(data["mlp-block-alpha"]-data["direct"]).max()<3e-6
    assert np.abs(data["mlp-restore-alpha"]-data["mlp"]).max()<3e-6
    assert np.abs(data["trio-q19-alpha"][:,:,1]-alpha[:,:,1]).max()<3e-6
    interaction=after+before-log(data["direct"])-log(data["mlp"])
    means=np.array([freeze["baseline"][str(x)] for x in data["strata"]])
    baseline=np.clip(anchor[:,None]+means,1e-8,1.)
    predictors={}
    for z,pred in [(k,data[k]) for k in ("full","direct","mlp","sparse","random","q_only","k_only")]+[("cell",baseline)]:
        cap,g=cap_flags(anchor,pred)
        err=log(pred)-after
        tp=(cap&target).sum((0,1)); fp=(cap&~target).sum((0,1))
        tn=(~cap&~target).sum((0,1)); fn=(~cap&target).sum((0,1))
        sensitivity=tp/np.maximum(1,tp+fn)
        specificity=tn/np.maximum(1,tn+fp)
        predictors[z]=dict(
            logalpha_rmse=rmse(err,(0,1,3)).tolist(),
            alpha_max=float(np.max(abs(pred-alpha))),
            cap_balanced_accuracy=((specificity+sensitivity)/2).tolist(),
            cap_tp=tp.tolist(),cap_fp=fp.tolist(),cap_fn=fn.tolist(),
            capped_views=cap.any(1).sum(0).tolist(),
            cap_agreement=(cap==target).mean((0,1)).tolist(),
            slot_gap_rmse=rmse((log(pred)[...,0]-log(pred)[...,1])-(after[...,0]-after[...,1]),(0,1)).tolist())
    sparse_test=[x<.9*z and x<q for x,z,q in zip(predictors["sparse"]["logalpha_rmse"],
                     predictors["direct"]["logalpha_rmse"],predictors["random"]["logalpha_rmse"],strict=True)]
    gain0=data["trio-gain"]
    primary_overflow=gain0[:,:,0]>2.05+2e-6
    n_pair=anchor.sum(-1)[:,None]
    c_pair=alpha.sum(-1)
    n_share=anchor[:,None]/n_pair[...,None]
    c_share=alpha/c_pair[...,None]
    mass_only=np.broadcast_to(n_share,alpha.shape)*c_pair[...,None]
    share_only=c_share*n_pair[...,None]
    cap_mass,_=cap_flags(anchor,mass_only)
    cap_share,_=cap_flags(anchor,share_only)
    overflow_components={}
    for h in range(2):
        found=target[:,:,h]
        pair_change=log(c_pair[:,:,h])-log(n_pair[:,:,h])
        mass_change=current_arms(c_share)[:,:,h]/np.maximum(current_arms(np.broadcast_to(n_share,alpha.shape))[:,:,h],1e-12)
        split_change=log(mass_change)
        overflow_components[str(h)]=dict(
            events=int(found.sum()),
            median_log_total_mass_change=float(np.median(pair_change[found])) if found.any() else None,
            median_log_within_pair_share_change=float(np.median(split_change[found])) if found.any() else None,
            reproduced_by_mass_only=int((cap_mass[:,:,h]&found).sum()),
            reproduced_by_share_only=int((cap_share[:,:,h]&found).sum()),
            reproduced_by_both_separately=int((cap_mass[:,:,h]&cap_share[:,:,h]&found).sum()),
            log_required_gain_median=float(np.median(np.log(need[:,:,h][found]))) if found.any() else None)
    cap_cells=Counter()
    for j,(s,t) in enumerate(N.CONDITIONS):
        cap_cells[int(s),int(t)]=int(primary_overflow[:,j].sum())
    condition_overflow={}
    for ctx in (0,1):
        for order in (0,1):
            indices=np.array([r["context"]==ctx and r["order"]==order for r in vs])
            condition_overflow[f"{ctx}{order}"]=dict(n=int(indices.sum()),h19_marked=int(primary_overflow[indices].any(1).sum()),
                                                      h25_marked=int(target[indices,:,1].any(1).sum()))
    # Score signed alternative-minus-correct margin separately from attention.
    sgn=np.array([1 if x["alt_code"]=="Q" else -1 for x in vs])
    contrast=lambda s:(s[...,1]-s[...,0])*(sgn[:,None] if s.ndim==3 else sgn)
    original_margin=contrast(data["native_scores"])
    armstats={}
    for name in ("source","mlp-block","mlp-restore","units-block","units-restore","null-units-block","null-units-restore",
                 "q-block","q-restore","k-block","k-restore","qk-block","qk-restore"):
        read=log(data[name+"-alpha"])
        distance0=rmse(read-before,(0,1,3))
        distance1=rmse(read-after,(0,1,3))
        scores=contrast(data[name+"-score"])
        rel=rmse(actual,(0,1,3))
        h19_errors=(read-after)[:,:,0] if name.endswith("-restore") else (read-before)[:,:,0]
        worst=np.argsort(np.max(abs(h19_errors),axis=(1,2)))[::-1]
        qk=data[name+"-drift"]
        armstats[name]=dict(
            logalpha_rms_from_natural=distance0.tolist(),logalpha_rms_from_source=distance1.tolist(),
            removal_ratio=(1-distance0/rel).tolist(),restoration_ratio=(1-distance1/rel).tolist(),
            alternative_code_margin_mean_abs=float(np.abs(scores-original_margin[:,None]).mean()),
            q_and_self_key_change_avg_norm=qk.mean((0,1)).tolist(),
            h19_counterexamples=[dict(id=ids[i],rms=float(rmse(h19_errors[i],None))) for i in worst[:6]])
    k_restore=log(data["k-restore-alpha"])
    natural_odds=before[...,0]-before[...,1]
    odds_error=float(np.max(np.abs(k_restore[...,0]-k_restore[...,1]-natural_odds)))
    assert odds_error<3e-6
    direct_cap,_=cap_flags(anchor,data["mlp-block-alpha"])
    cap_switch=dict(prevented=(target&~direct_cap).sum((0,1)).tolist(),
                    introduced=(~target&direct_cap).sum((0,1)).tolist())
    doses={}
    original=None
    all_status={}
    for mode in ("trio","trio-mlp","trio-q19","trio-q","trio-qk"):
        flags=data[mode+"-dose-flags"]
        pair=data[mode+"-pair-flags"]
        assert np.array_equal(data[mode+"-want"],data["trio-want"])
        assert np.array_equal(flags[...,1],data[mode+"-gain"]>2.05+2e-6)
        slices=(~flags.any(axis=(1,2,3))) & (~pair.any(axis=(1,2)))
        if original is None:
            original=slices.copy()
        switched=slices&~original
        lost=(~slices)&original
        fam=[i for i in familyidx.values() if slices[i].all()]
        om=contrast(data[mode+"-score"])
        sgap=(om[:,1]-om[:,0])-(om[:,3]-om[:,2])
        stats=family_stats([sgap[i].mean() for i in fam])
        admitted=len(fam)>=p["triad"]["minimum_full_families"]
        equ=bool(admitted and stats["ci95"][0]>-.25 and stats["ci95"][1]<.25)
        all_status[mode]=slices
        doses[mode]=dict(
            views=int(slices.sum()),rescued_views=int(switched.sum()),lost_views=int(lost.sum()),
            families=len(fam),family_coverage=admitted,source_difference=stats,equivalence=equ,
            median_min_write=np.median(data[mode+"-actual"].min(1)[slices],axis=0).tolist() if slices.any() else None,
            by_head_reason=flags.any(1).sum(0).tolist(),pair_failure_per_head=pair.any(1).sum(0).tolist(),
            h19_caps=int(flags[:,:,0,1].any(1).sum()),h25_caps=int(flags[:,:,1,1].any(1).sum()),
            rescued_examples=[ids[i] for i in np.flatnonzero(switched)[:8]],
            lost_examples=[ids[i] for i in np.flatnonzero(lost)[:8]],
            remaining_examples=[ids[i] for i in np.flatnonzero(~slices)[:8]])
    inds=np.argsort(need[:,:,0].max(1))[::-1]
    examples=[]
    for i in inds[:16]:
        j=int(np.argmax(need[i,:,0]))
        s=N.CONDITIONS[j][0]
        examples.append(dict(id=ids[i],case=j,text=vs[i]["text"],head19_natural_alpha=float(anchor[i,0,s]),
                             head19_live_alpha=float(alpha[i,j,0,s]),head19_gain=float(need[i,j,0]),
                             direct=float(data["direct"][i,j,0,s]),mlp=float(data["mlp"][i,j,0,s]),
                             q_only=float(data["q_only"][i,j,0,s]),k_only=float(data["k_only"][i,j,0,s]),
                             matched={key:bool(value[i]) for key,value in all_status.items()}))
    at=np.linalg.norm(data["actual-qk"]-data["na_qk"][:,None],axis=-1)
    A.save("audit.json",dict(snapshot=p["snapshot"],provenance=p["provenance"],eligible_views=len(ids),
        eligible_families=len(familyidx),behavior=counts,forecast_sha=done["forecast"],predictors=predictors,
        exact_full_alpha_error=fullerr,discovery_unit_score_fraction=freeze["score_fraction"],sparse_pass=sparse_test,
        key_only_pair_odds_max_error=odds_error,qk_change_median_norm=np.median(at,axis=(0,1)).tolist(),
        residual_mlp_logalpha_interaction_rms=rmse(interaction,(0,1,3)).tolist(),
        median_direct_residual_norm=float(np.median(data["direct_norm"])),median_delta_mlp_norm=float(np.median(data["mlp_norm"])),
        original_caps=target.any(1).sum(0).tolist(),
        original_cap_arms=target.sum((0,1)).tolist(),h19_cap_by_slot_signal={f"{s}{t}":n for (s,t),n in cap_cells.items()},
        cap_by_factors=condition_overflow,overflow_components=overflow_components,mlp_block_cap_switches=cap_switch,
        interventions=armstats,doses=doses,counterexamples=examples,
        analysis_sha=A.digest(Path(__file__)),scope="Same-node outcome and arithmetic verification. Cached prior keys invariant per runtime assert."))


if __name__=="__main__":
    C.configure()
    summary()
