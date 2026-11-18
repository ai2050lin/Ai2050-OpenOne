"""Freeze Phase 2945 and stream the original Qwen3-14B blocks through CUDA."""

from collections import defaultdict
from itertools import batched
from pathlib import Path
import sys
import time

import numpy as np

import kv_bf_q14 as Native
import kv_reach_run as Prior
import kv_natural_material as M


A, B, P = Prior.A, Prior.B, Prior.P
ROOT = Prior.ROOT
OUT = ROOT / "tests/glm5/result/kv_natural_20261115_v1"
TMP = ROOT / "tests/glm5_temp/kv_natural_20261115_v1"
RUN = "KV-NATURAL-20261115-v1"
TARGET = {25: (18,), 26: (4, 34)}
SLOT_COUNT = len(M.FIELDS)
BATCH = 8
SLAB = 24
ALPHA = .05/6
SHAM_TOL = 1.
RESTORE_TOL = .125


def setup():
    A.OUT, A.TEMP, A.RUN = OUT, TMP, RUN
    A.configure()
    B.C.OUT = OUT


def old_words():
    words = set()
    for folder in (Prior.OUT, Prior.OLD, Prior.OUT.with_name("kv_address_20261115_v1"),
                   Prior.OUT.with_name("kv_relay_20261115_v1")):
        for p in folder.glob("material-*.json"):
            words.update(Prior.words(P.load(p)))
    return words


def split_material(split, tok, excluded, revision=1, families=None):
    material = M.generate(split, tok, excluded, revision, families)
    for r in material:
        r["token"] = M.token_plan(tok, r)
        assert (min(r["token"]["gap"]) >= 35) == bool(r["distance"]), r["token"]["gap"]
    return material


def words(rows):
    return {w for r in rows for w in [*r["values"].values(), r["rejected"]]}


def plan():
    assert not (OUT/"plan.json").exists()
    pilot = A.load("pilot-behavior.json")
    assert pilot["passed"], pilot["cells"]
    excluded = old_words() | words(A.load("material-pilot.json"))
    tok = A.Base.tok()
    binding = {}
    for split in ("discover", "confirm"):
        rows = split_material(split, tok, excluded)
        assert not words(rows)&excluded
        excluded.update(words(rows))
        name = f"material-{split}.json"
        A.save(name, rows)
        binding[split] = dict(name=name, hash=B.sha(name), families=len(rows)//8, n=len(rows), words=sorted(words(rows)))
    identity = B.model_identity("q14")
    last = P.load(Prior.OUT/"plan.json")
    assert last["identity"]["q14"] == identity
    selected = P.load(Prior.OUT/"q14-freeze.json")["selected"]
    assert selected["source"] == [25,18] and selected["receiver"] == [4,34]
    sources = set(B.SOURCES) | {"tests/glm5/kv_bf_q14.py", "tests/glm5/kv_reach_run.py"}
    sources.update(x.relative_to(ROOT).as_posix() for x in Path(__file__).parent.glob("kv_natural_*.py"))
    A.save("plan.json", dict(run=RUN, time=Prior.stamp(), targets=TARGET, identity=identity, seed=M.SEED,
        materials=binding, sources={k:A.digest(ROOT/k) for k in sorted(sources)}, runtime=A.Base.versions(),
        schema="Eight views are extra competitors × generated receipt length × State/Note order. Roles and wording match across each view except prescribed changes. Five lexemes sampled without family reuse, vocabularies disjoint across splits and prior route panels. Code letter and two rule variants balanced by family. Original authored text. Code metric is full-vocabulary greedy at Code:.",
        behavior="Verify every continuation token against uncoerced full-vocabulary logits starting at the chat boundary. Only line-ending whitespace can adapt from the model's actual next token, using the frozen finite token set, with at most five rechecks, and no changes to values/receipt/Code marker. Next code token is predicted without constraints. Per 8 cells at least 90% verified greedy prefix and correct uncoerced code in pilot, discovery and confirmation. Analysis uses whole eight-view eligible families. Preserve every first-mismatch prefix and whitespace decision. Minimum 80 eligible confirmation families for inferential claim. Causal replay window ends after the actual code token.",
        intervention="During the final Code: query only, set the selected pre-W_O head output to zero or restore the intact head's BF16 vector through the original projection. Full live Q and K recompute per branch; earlier token history identical by causal construction. Source-only, each receiver alone, both receivers, reversible source/receiver self restores, upstream block + native receiver pair patch, upstream block + matched random receiver pair patch, and joint block for occlusion. Matched controls do the same edits on same-layer non-target heads.",
        control="Use discovery only; choose two random controls by fixed seed from closest quartile of geometric mean native projected write norm at layer 25, and from distinct GQA groups at layer 26 matched separately to H4 and H34. Use the first and second draws for matched control mean. Restore random receiver writes in the corresponding upstream-off arm, with dynamic Q/K. All confirmation heads fixed before opening behavior.",
        primary=dict(alpha=ALPHA, confidence=1-ALPHA, endpoints=["source margin loss vs mean same-layer controls",
                      "H4 margin loss vs mean same-layer controls", "H34 margin loss vs mean same-layer controls",
                      "pair margin loss vs mean matched pairs", "recipient-specific margin rescue vs two randomized patch pairs within upstream block",
                      "recipient-specific absolute rescue within upstream block"], min_mean_logit=.10,
                      rule="Report 99.1667% paired family-bootstrap intervals, mean >=0.10 and lower bound >0. A chain interpretation also needs upstream loss above control, pair loss above control, positive rescue above control and absolute rescue, plus verified physical output changes. Accuracy and sign flips are separate and descriptive; leave-one-lexeme-out as sensitivity.",
                      power="128 paired families give normal-approximate power .97 at standardized paired effect .4, two-sided alpha .05/6, before behavior exclusions or lexical dependence. Evaluate effect sizes and coverage at actual n."),
        trace="Store full selected BF16 head writes, per-head softmax on 4 generated slots and prompt same-token aggregate, reconstructed local projected norms, actual post-W_O and post-MLP delta norms, final response, P/Q scores, and first mismatch/counterexamples. Behavior, discovery and freeze are separate on-disk stages. Calculate effects against shape-matched, untouched packed sham.",
        numeric=dict(sham_raw_logit_max=SHAM_TOL, restore_packed_max=RESTORE_TOL, require_same_full_argmax=True,
                     calibration="Packed one-query and full-prefix BF16 use different kernels. Pilot showed same full argmax and up to .50 raw P/Q differences. Preserve each untreated baseline and use only matched packed arms in paired endpoints."),
        pilot={k:B.sha(k) for k in ("material-pilot.json","pilot-behavior.json","stock-check.json","pilot-trace.json",
                                   "pilot-behavior-v1.json","material-pilot-v1.json","pilot-behavior-v2.json","material-pilot-v2.json")}))


def verify():
    p = A.load("plan.json")
    assert p["runtime"] == A.Base.versions()
    assert all(B.sha(x["name"]) == x["hash"] for x in p["materials"].values())
    assert all(A.digest(ROOT/k) == v for k,v in p["sources"].items())
    assert all(B.sha(k) == v for k,v in p["pilot"].items())
    return p


def progress(stage, i, n, clock, **details):
    P.stage_log("q14", stage, i, n, clock, **details)


class Engine(Native.Engine):
    def packet(self, rows):
        t = self.t
        b = [r["token"] for r in rows]
        width = max(len(x["ids"]) for x in b)
        pads = [width-len(x["ids"]) for x in b]
        ids = t.tensor([[self.tok.pad_token_id]*p+x["ids"] for p,x in zip(pads,b,strict=True)], device="cuda")
        axis = t.arange(width, device="cuda")
        valid = axis[None] >= t.tensor(pads,device="cuda")[:,None]
        mask = valid[:,None,None] & (axis[None,:]<=axis[:,None])[None,None]
        positions = (valid.cumsum(-1)-1).masked_fill(~valid,1)
        x = self.decoder.embed_tokens(ids)
        cs = tuple(y.cpu() for y in self.rope(x,positions))
        slots = t.as_tensor([z["slots"] for z in b])+t.as_tensor(pads)[:,None]
        same = t.zeros((len(b), SLOT_COUNT, width), dtype=t.bool)
        for i,(z,pad) in enumerate(zip(b,pads,strict=True)):
            for j,indices in enumerate(z["prompt_slots"]):
                same[i,j,[k+pad for k in indices]] = True
        return dict(rows=rows, x=x.cpu(), cs=cs, mask=mask.cpu(), ids=ids.cpu(), pos=positions.cpu(),
                    starts=[z["start"]+p for z,p in zip(b,pads,strict=True)], padding=pads,
                    slots=slots, same=same, norms={}, records={})

    def projections(self, layer, heads, wrote):
        # FP32 decomposition of the original O weight. Actual full BF16 projection is recorded separately.
        w = layer.self_attn.o_proj.weight.reshape(self.width,self.heads,128)[:,heads].float()
        return self.t.einsum("...hd,mhd->...hm",wrote.float(),w)

    def base_step(self, layer, i, p, mode, commands):
        t = self.t
        x = p["x"].cuda()
        mask = p["mask"].cuda()
        cs = tuple(z.cuda() for z in p["cs"])
        q,k,v = self.qkv(layer,x,cs)
        o = self.attention(layer,q,k,v,mask)
        if mode == "discover" and i in TARGET:
            norms = t.linalg.vector_norm(self.projections(layer,tuple(range(self.heads)),o[:,-1]),dim=-1)
            p["norms"][i] = Native.f32(norms)
        if mode in ("intervene","pilot-trace") and i >= min(TARGET):
            from kv_natural_trace import branch
            branch(self,layer,i,p,commands,x,mask,cs,q,k,v,o)
        p["x"] = self.residual(layer,x,o).cpu()

    def greedy_check(self, p):
        t = self.t
        start = min(p["starts"])-1
        logits = self.logits(p["x"].cuda()[:,start:])
        predicted = logits.argmax(-1).cpu().numpy()
        candidate = p["ids"].numpy()
        letters = p["rows"][0]["token"]["letters"]
        result = []
        for j,row in enumerate(p["rows"]):
            ix = p["starts"][j]
            z = predicted[j,ix-1-start:-1]
            expected = candidate[j,ix:]
            wrong = np.flatnonzero(z!=expected)
            m = int(wrong[0]) if len(wrong) else len(z)
            top = int(predicted[j,-1]) if m == len(z) else int(z[m])
            result.append(dict(id=row["id"], prefix_valid=m==len(z), code_correct=m==len(z) and top==letters[row["code"]=="Q"],
                prefix_tokens=m, expected_prefix_tokens=len(z), generated_tokens=[*expected[:m].tolist(),top], top_token=top,
                rendered=self.tok.decode([*expected[:m].tolist(),top]), first_wrong_expected=int(expected[m]) if m<len(z) else None,
                pq=Native.f32(logits[j,-1,letters]).tolist() if m==len(z) else None))
        return result

    def stock_check(self, rows):
        t = self.t
        p = self.packet(rows)
        for i in range(len(self.layers)):
            with self.block(i) as layer:
                self.base_step(layer,i,p,"pilot",None)
        start = min(p["starts"])-1
        got = self.logits(p["x"].cuda()[:,start:])
        with self.stock_forward():
            out = self.model(input_ids=p["ids"].cuda(), attention_mask={"full_attention":p["mask"].cuda()},
                             position_ids=p["pos"].cuda(), use_cache=False, logits_to_keep=got.shape[1]).logits
        good = t.arange(got.shape[1],device="cuda")[None]>=t.tensor(p["starts"],device="cuda")[:,None]-1-start
        agreement = bool(t.equal(got.argmax(-1)[good],out.argmax(-1)[good]))
        err = float((got.float()-out.float()).abs()[good].max())
        assert agreement and err <= .25, (agreement,err)
        return dict(max_logit_error=err,greedy_agreement=agreement,rows=[r["id"] for r in rows],positions=int(good.sum()))

    def batches(self, rows, mode, commands=None):
        batches = list(batched(rows,BATCH))
        slabs = list(batched(batches,SLAB))
        for j,work in enumerate(slabs,1):
            packets = [self.packet(r) for r in work]
            start = time.monotonic()
            progress(mode,0,len(self.layers),start,part=j,parts=len(slabs),batches=len(work))
            for i in range(len(self.layers)):
                with self.block(i) as layer:
                    for p in packets:
                        self.base_step(layer,i,p,mode,commands)
                progress(mode,i+1,len(self.layers),start,part=j,parts=len(slabs),batches=len(work))
            yield j,packets

    def observe(self, rows, mode):
        pending = rows
        finished, norms, history = {}, {}, defaultdict(list)
        for attempt in range(6):
            retry=[]
            for _,packets in self.batches(pending,mode):
                for p in packets:
                    for j,(row,o) in enumerate(zip(p["rows"],self.greedy_check(p),strict=True)):
                        proposed=row["token"]
                        index=proposed["start"]+o["prefix_tokens"]
                        output=self.tok.decode(o["top_token"])
                        if not o["prefix_valid"] and output.endswith("\n") and attempt<5:
                            adapted=self.whitespace_proposal(row,index,o["top_token"],output)
                            if adapted is not None:
                                change,new=adapted
                                history[row["id"]].append(dict(position=index,line=change,emitted=o["top_token"],text=output))
                                retry.append(dict(row,token=new))
                                continue
                        o["whitespace"]=history[row["id"]]
                        if o["whitespace"]:
                            o["resolved_token"]=proposed
                        finished[row["id"]]=o
                        if mode=="discover":
                            norms[row["id"]]={l:x[j] for l,x in p["norms"].items()}
            if not retry:
                break
            pending=retry
        return [finished[r["id"]] for r in rows],norms

    def whitespace_proposal(self,row,index,token,shown):
        old=row["token"]
        for i,end in enumerate(old["break_positions"]):
            if index not in (end-1,end):
                continue
            for style in sorted(M.VALID_BREAKS,key=len,reverse=True):
                if not shown.endswith(style):
                    continue
                # Forecast remaining separators with the latest style. Every forecast must still pass greedy replay.
                try:
                    proposal=M.token_plan(self.tok,row,old["breaks"][:i]+[style]*(len(old["breaks"])-i))
                except AssertionError:
                    continue
                if proposal["ids"][:index]==old["ids"][:index] and proposal["ids"][index]==token:
                    return i,proposal
        return None


def behavior(rows, seen):
    cells,families = defaultdict(list),defaultdict(list)
    for r,o in zip(rows,seen,strict=True):
        assert r["id"]==o["id"]
        cells[r["id"].split("-")[-1]].append(o)
        families[r["family"]].append(o)
    c = {k:dict(n=len(v),greedy=sum(x["prefix_valid"] for x in v),correct=sum(x["code_correct"] for x in v),
               passed=sum(x["code_correct"] for x in v)>=.90*len(v)) for k,v in sorted(cells.items())}
    return dict(cells=c,passed=all(x["passed"] for x in c.values()),rows=seen,
                eligible=[k for k,v in families.items() if len(v)==8 and all(x["code_correct"] for x in v)])


def pilot(m):
    rows = split_material("pilot",m.tok,old_words())
    A.save("material-pilot.json",rows)
    seen,_ = m.observe(rows,"pilot")
    b = behavior(rows,seen)
    A.save("pilot-behavior.json",b)
    if b["passed"]:
        clock=time.monotonic()
        progress("stock",0,1,clock)
        resolved={x["id"]:x["resolved_token"] for x in seen if "resolved_token" in x}
        actual=[dict(r,token=resolved.get(r["id"],r["token"])) for r in rows]
        A.save("stock-check.json",m.stock_check([actual[0],actual[3]]))
        progress("stock",1,1,clock)
        pilot_trace(m)
    print("Pilot",b["cells"],flush=True)


def pilot_trace(m):
    from kv_natural_trace import arms,pack_result
    b=A.load("pilot-behavior.json")
    assert b["passed"] and A.load("stock-check.json")["greedy_agreement"]
    known={x["id"]:x for x in b["rows"]}
    eligible=set(b["eligible"])
    rows=A.load("material-pilot.json")
    panel=[dict(r,token=known[r["id"]].get("resolved_token",r["token"])) for r in rows if r["family"] in eligible][:8]
    commands=arms(dict(source=[0,1],receivers=[[1,10],[2,11]]))
    z=next(pack_result(m,x,commands,known) for _,packets in m.batches(panel,"pilot-trace",commands) for x in packets)
    A.save("pilot-trace.json",dict(n=len(z["ids"]),dimensions={k:list(a.shape) for k,a in z.items()},
                                 max_sham_pq_difference=float(abs(z["pq"][:,0]-z["untreated"]).max())))


def run(stage):
    import torch

    p = None if stage in ("pilot","pilot-trace") else verify()
    identity = P.load(Prior.OUT/"plan.json")["identity"]["q14"] if p is None else p["identity"]
    assert B.model_identity("q14") == identity
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    with P.GpuLease(),torch.inference_mode():
        clock=time.monotonic()
        progress("load",0,1,clock)
        m=Engine()
        progress("load",1,1,clock)
        try:
            if stage=="pilot":
                pilot(m)
            elif stage=="pilot-trace":
                pilot_trace(m)
            else:
                from kv_natural_trace import discover, confirm
                {"discover":discover,"confirm":confirm}[stage](m,p)
        finally:
            m.close()


if __name__=="__main__":
    # Trace/analysis helpers import the same module identity when invoked via CLI.
    sys.modules["kv_natural_run"] = sys.modules[__name__]
    setup()
    if sys.argv[1]=="plan":
        plan()
    else:
        run(sys.argv[1])
