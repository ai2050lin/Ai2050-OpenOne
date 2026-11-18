"""Independent CPU scoring, array checks and family-wise exact sign randomization."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import own_state_explicit as E
R=E.R
V2=R.OUT
V1=V2.with_name("own_state_20261115_v1")
ARMS=("replay","state_token_swap","value_patch","matched_null")


def official_snapshot(key, files):
    from shared.dist_protocol.runner_v3 import SNAPSHOT_PATTERNS
    folder=R.ROOT/"models/hf"/R.MODELS[key]
    paths={p.name:p for pattern in SNAPSHOT_PATTERNS for p in folder.glob(pattern) if p.is_file()}
    data={n:dict(sha256=files.get(n) or R.digest(p),size=p.stat().st_size) for n,p in paths.items()}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode("utf-8")).hexdigest()


def independent_score(row, record, tokenizer, eos, native):
    tokens=record["tokens"]
    assert tokens
    assert all(k not in eos for k in tokens[:-1])
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    assert text==record["text"]
    fields=text.strip().splitlines()
    state,code=None,None
    if len(fields)==2:
        value=fields[0].strip().lower()
        code_header, sep, value_code=fields[1].strip().partition(":")
        if value.isascii() and value.isalpha() and sep and code_header.strip().casefold()=="code" and value_code.strip().upper() in {"P","Q"}:
            state,code=value,value_code.strip().upper()
    early=tokenizer.decode(tokens[:2],skip_special_tokens=True)
    first=tokenizer.decode(tokens[:1],skip_special_tokens=False).strip()
    viable=(len(tokens)>=3 and tokens[0] in native.values() and first.isascii() and first.isalpha()
            and not re.search(r"\b[QP]\b",early))
    assert record["state"]==state and record["code"]==code
    assert record["site"]==viable and record["eos"]==(tokens[-1] in eos)
    assert record["state_correct"]==(state==row["state"])
    assert record["code_correct"]==(code==row["code"])
    assert record["joint"]==(state==row["state"] and code==row["code"] and record["eos"] and viable)
    assert record["flips"]==(code==row["alt_code"] and record["eos"] and state is not None)


def material_audit(row):
    u,d,m=row["factors"]
    assert row["state"]==(row["new"] if u else row["old"])
    assert row["alt"]==(row["old"] if u else row["new"])
    old,new=(("Q","P") if m else ("P","Q"))
    assert row["code"]==(new if u else old) and row["alt_code"]==(old if u else new)
    if row["text"].startswith("Read one "):
        text=row["text"]
        assert ("Action: Replace" if u else "Action: Keep") in text
        assert (text.index("Unrelated training example value:")>text.index("Action:"))==bool(d)
        assert f'{row["old"]} -> {old}\n{row["new"]} -> {new}\n' in text


def behavior_audit(path):
    from transformers import AutoTokenizer
    load=R.Previous.load
    material={r["id"]:r for split in ("pilot","confirm") for r in load(path/f"materials-{split}.json")}
    assert len(material)==192+1536 and len({r["text"] for r in material.values()})==len(material)
    keys={}
    results=[]
    for key,folder in R.MODELS.items():
        identity_file=path/f"{key}-model.json"
        if not identity_file.exists():
            continue
        identity=load(identity_file)
        assert R.Previous.digest(identity["files"])==identity["model_manifest_sha"]
        assert R.digest(path/"plan.json")==identity["plan_sha"]
        tokenizer=AutoTokenizer.from_pretrained(R.ROOT/"models/hf"/folder,local_files_only=True)
        keys[key]=dict(snapshot=official_snapshot(key,identity["files"]),eos=identity["eos"])
        for split in ("pilot","confirm"):
            file=path/f"{key}-{split}.json"
            if not file.exists():
                continue
            actual=load(file)
            records=actual["records"]
            checked=[]
            outcomes=[]
            for r in records:
                row=material[r["id"]]
                assert row["split"]==split
                material_audit(row)
                expected, mapping=R.encode(tokenizer,row)
                assert r["input_ids"]==expected and r["native_values"]==mapping
                independent_score(row,r["natural"],tokenizer,set(identity["eos"]),mapping)
                if not r["natural"]["joint"]:
                    outcomes.append(dict(id=r["id"],task=row["task"],factors=row["factors"],expected=[row["state"],row["code"]],
                                         actual=r["natural"]))
                checked.append(row)
            assert len({r["id"] for r in records})==len(records)
            assert actual["cells"]==R.marks(checked,records)
            assert actual["admitted"]==R.approved(actual["cells"])
            results.append(dict(model=key,split=split,view_count=len(records),cells=actual["cells"],
                                admitted=actual["admitted"],failures=outcomes))
            print(path.name,key,split,"checked",len(records),flush=True)
    return dict(models=keys,runs=results)


def f32(bits):
    return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)


def round_bf16(x):
    bits=x.view(np.uint32)
    return ((((bits+np.uint32(0x7fff)+(bits>>16&1))>>16)<<16).view(np.float32))


def verify_norms(archive,rows):
    values=f32(archive["value_bits"])
    assert tuple(values.shape)==(len(rows),36,2,8,128)
    base=values[:,:,0]
    true=values[:,:,1]
    differences=true-base
    scramble=np.stack([np.roll(differences[:,l],shift=(13+11*l)%127+1,axis=-1) for l in range(36)],axis=1)
    norm=np.linalg.norm(differences.reshape(len(rows),-1).astype(np.float64),axis=1)
    shuffled=(round_bf16(base+scramble)-base).reshape(len(rows),-1)
    norm_control=np.linalg.norm(shuffled.astype(np.float64),axis=1)
    assert np.isfinite(norm).all() and np.isfinite(norm_control).all()
    assert np.allclose(norm,[r["target_norm"] for r in rows],rtol=2e-6,atol=2e-6)
    assert np.allclose(norm_control,[r["control_norm"] for r in rows],rtol=2e-6,atol=2e-6)
    ratio=np.abs(norm-norm_control)/np.maximum(1e-20,norm)
    assert np.max(ratio)<=R.GATES["max_norm_relative_error"]
    return dict(max_norm_error=float(np.max(ratio)),min_target_norm=float(norm.min()),max_target_norm=float(norm.max()))


def sign_flip(integer_differences):
    observed=sum(integer_differences)
    counts=Counter({0:1})
    nonzero=[abs(j) for j in integer_differences if j]
    for step in nonzero:
        nxt=Counter()
        for value,count in counts.items():
            nxt[value+step]+=count
            nxt[value-step]+=count
        counts=nxt
    return sum(count for value,count in counts.items() if value>=observed)/2**len(nonzero)


def pair_effect(groups, key_a, key_b, denom=8):
    scores=[sum(int(hit(r,key_a))-int(hit(r,key_b)) for r in runs) for runs in groups]
    differences=np.asarray(scores,dtype=float)/denom
    avg=differences.mean()
    drawn=np.random.default_rng(20261115).choice(differences,size=(6000,len(differences)),replace=True).mean(1)
    return dict(n=len(scores),difference=float(avg),ci95=np.quantile(drawn,[.025,.975]).tolist(),
                p_sign_flip=sign_flip(scores),nonzero_families=sum(s!=0 for s in scores))


def hit(r,arm):
    return r["applied"] and r[arm]["flips"]


def analyze_task(task,rows,materials):
    families=defaultdict(list)
    cells=defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
        cells[tuple(row["factors"])].append(row)
    assert all(len(v)==8 for v in families.values())
    assert len(cells)==8 and len(families)==96
    arms={k:dict(flips=sum(hit(r,k) for r in rows),wellformed=sum(r[k]["state"] is not None and r[k]["eos"] for r in rows))
          for k in ARMS}
    cell_ledger=[dict(factors=f,n=len(g),**{k:sum(hit(r,k) for r in g) for k in ARMS}) for f,g in sorted(cells.items())]
    token=pair_effect(families.values(),"state_token_swap","replay")
    value=pair_effect(families.values(),"value_patch","matched_null")
    ctl=(arms["matched_null"]["flips"]/len(rows)<=R.GATES["max_noise_flip"]
         and (arms["replay"]["wellformed"]-arms["matched_null"]["wellformed"])/len(rows)<=R.GATES["max_control_format_loss"])
    token_pass=(token["difference"]>=R.GATES["min_token_effect"] and token["p_sign_flip"]<=R.GATES["alpha"]
                and all(r["state_token_swap"]>r["replay"] for r in cell_ledger))
    value_pass=(value["difference"]>=R.GATES["min_v_minus_control"] and value["p_sign_flip"]<=R.GATES["alpha"]
                and all(r["value_patch"]>=r["matched_null"] for r in cell_ledger)
                and all(sum(int(hit(r,"value_patch"))-int(hit(r,"matched_null")) for r in rows if r["factors"][2]==m)>0 for m in (0,1)))
    failures={k:[dict(id=r["id"],factors=r["factors"],out=r[k]) for r in rows
                 if hit(r,k)!=(k in ("state_token_swap","value_patch"))] for k in ARMS}
    vocabulary=sorted({materials[r["id"]][field] for r in rows for field in ("old","new")})
    exclusion=[]
    for word in vocabulary:
        kept=[r for r in rows if word not in (materials[r["id"]]["old"],materials[r["id"]]["new"])]
        exclusion.append(dict(excluded=word,n=len(kept),
                              delta=(sum(hit(r,"value_patch") for r in kept)-sum(hit(r,"matched_null") for r in kept))/len(kept)))
    return dict(task=task,n=len(rows),families=len(families),applied=sum(r["applied"] for r in rows),arms=arms,
                cells=cell_ledger,token=token,value=value,native_gate=ctl and token_pass and value_pass,
                token_gate=token_pass,value_gate=value_pass,control_gate=ctl,failures=failures,
                exclude_one_value=exclusion,
                example=next((r for r in rows if hit(r,"value_patch") and not hit(r,"matched_null")),None))


def causal_audit(behavior):
    from transformers import AutoTokenizer
    record=R.read("causal-run.json")
    if record["status"]!="complete":
        return dict(status=record["status"],tasks=[],views=0)
    assert record["behavior_sha"]==R.digest(V2/"q4-confirm.json")
    admitted={r["id"]:r for r in R.read("q4-confirm.json")["records"]}
    materials={r["id"]:r for r in R.read("materials-confirm.json")}
    tok=AutoTokenizer.from_pretrained(R.ROOT/"models/hf"/R.MODELS["q4"],local_files_only=True)
    eos=set(behavior["models"]["q4"]["eos"])
    seen=[]
    checks=[]
    start=time.monotonic()
    for step,blob in enumerate(record["batches"],1):
        assert R.digest(R.OUT/blob["name"])==blob["sha"]
        batch=R.read(blob["name"])
        assert R.digest(R.OUT/batch["slot_file"])==batch["slot_sha"]
        verify=batch["checks"]
        assert all(verify[k] for k in ("admitted_tokens_match","exact_tokens","bitwise_logits","source_kv_immutable",
                                       "norm_match","natural_state_cache_immutable"))
        assert verify["natural_logits_sha"]==verify["replay_logits_sha"] and len(verify["original_prompt_kv_sha"])==36
        rows=batch["rows"]
        with np.load(R.OUT/batch["slot_file"]) as archive:
            assert archive["ids"].tolist()==[r["id"] for r in rows]
            quality=verify_norms(archive,rows)
            donor=archive["actual_donor_token"].tolist()
        for r,tok_donor in zip(rows,donor,strict=True):
            row=materials[r["id"]]
            orig=admitted[r["id"]]
            assert r["task"]==row["task"] and r["factors"]==row["factors"] and r["family"]==row["family"]
            assert r["natural"]==r["replay"]==orig["natural"]
            assert r["applied"]==r["natural"]["joint"]
            if r["applied"]:
                assert tok_donor==orig["native_values"][row["alt"]] and r["state_token_swap"]["tokens"][0]==tok_donor
            for arm in ARMS:
                independent_score(row,r[arm],tok,eos,orig["native_values"])
            assert all(r[k]["tokens"][:2]==r["natural"]["tokens"][:2] for k in ("matched_null","value_patch"))
        field=verify["field"]
        if field is not None:
            with np.load(R.OUT/field) as source:
                assert source["hidden_bits"].ndim==4 and source["hidden_bits"].shape[1:]==(38,16,2560)
                assert source["ids"].tolist()==[r["id"] for r in rows if materials[r["id"]]["index"]<2]
                assert np.isfinite(f32(source["hidden_bits"])).all()
        seen.extend(rows)
        checks.append(dict(batch=blob["name"],native=quality,field=field))
        if step%8==0 or step==len(record["batches"]):
            dt=time.monotonic()-start
            print(f"causal audit {step}/{len(record['batches'])}, {dt:.1f}s elapsed, ETA {dt/step*(len(record['batches'])-step):.1f}s",flush=True)
    assert len(seen)==len({r["id"] for r in seen})==len([r for r in admitted if materials[r]["task"] in record["scope"]])
    return dict(status="complete",views=len(seen),checks=checks,
                tasks=[analyze_task(t,[r for r in seen if r["task"]==t],materials) for t in record["scope"]])


if __name__=="__main__":
    E.select()
    R.configure()
    R.check_plan()
    base=behavior_audit(V1)
    version2=behavior_audit(V2)
    combined=causal_audit(version2)
    R.write("audit.json",dict(v1=base,v2=version2,causal=combined,qa="Independent CPU scoring and raw array reconstruction, no independent GPU node."))
