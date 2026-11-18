"""Freeze and collect original BF16 language-chain experiments on one GPU at a time."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import file_sha, digest, save_json, GpuLease
from source_binding_resources import resources
import condition_chain_materials as M

OUT = ROOT / "tests/glm5/result/condition_chain_20261115_v1"
TMP = ROOT / "tests/glm5_temp/condition_chain_20261115_v1"
BUDGET = 370 * 2**20
RUN = "CONDITION-CHAIN-20261115-v1"
THRESHOLDS = dict(behavior=.85, alpha=.05 / 12, min_dz=.35, mse_ratio=.90,
                  max_good_rmse=.10, min_patch=.02, max_control=.01)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, record):
    save_json(check_write(OUT / name), record)


def stage_log(model, stage, done, total, start, **details):
    elapsed = time.monotonic() - start
    record = dict(model=model, stage=stage, done=done, total=total, seconds=elapsed,
                  eta_seconds=elapsed / done * (total - done) if done else None, **resources(), **details)
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        import torch
        record["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    with check_write(OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    check_write(OUT)
    assert record["shared_gpu_bytes"] is not None and record["shared_gpu_bytes"] < 2**30


def setup():
    check_write(OUT, BUDGET, operation=RUN)
    for path in (OUT, TMP):
        check_write(path).mkdir(exist_ok=True)
    for variable in ("TMP", "TEMP", "TMPDIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME",
                     "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
        os.environ[variable] = str(TMP)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def identity(key):
    model_path = ROOT / "models/hf" / M.MODELS[key]
    start = time.monotonic()
    files = {}
    material = sorted(p for p in model_path.iterdir() if p.suffix in {".json", ".safetensors", ".txt", ".model"})
    for i, path in enumerate(material, 1):
        files[path.name] = file_sha(path)
        stage_log(key, "snapshot", i, len(material), start)
    return files


def preregister():
    if (OUT / "plan.json").exists():
        raise FileExistsError("Frozen protocol already exists")
    panels = {}
    for split in M.SPLITS:
        for panel in ("chain", "candidate"):
            rows = M.rows(split, panel)
            name = f"materials-{split}-{panel}.json"
            write(name, rows)
            panels[name] = dict(n=len(rows), families=len({r["family"] for r in rows}), sha256=file_sha(OUT / name))
    source = [Path(__file__), Path(M.__file__), Path(__file__).with_name("condition_chain_analysis.py")]
    write("plan.json", dict(run_id=RUN, precision="BF16", dispatch={"q4":"cuda","q14":"11GiB GPU and CPU","glm":"11GiB GPU and CPU"},
                            models=M.MODELS, created=datetime.now(ZoneInfo("America/Chicago")).isoformat(),
                            license="Original agent-authored, fictional material; no external corpus",
                            candidates=M.CANDIDATES, thresholds=THRESHOLDS, materials=panels,
                            primary="12 comparisons: 3 G candidates x 2 heldouts x 2 independently crossed lexical-cue strata",
                            candidate_readout="last prefix-stable token in Record ends marker, after all clues and before question",
                            field_sample="q4 chain only, i<2 in every discovery/heldout task; all four sites and all raw block boundaries plus final norm",
                            intervention="Conditional on passing all response gates; frozen early Record ends token; ablate native A, match within block by discovery activation and down-column norm",
                            chain_claim="Factor means, discovery-trained heldout probes; truth availability and controls recorded; narrow option distribution",
                            sources={p.name:file_sha(p) for p in source},
                            runtime={p:importlib.metadata.version(p) for p in ("torch","accelerate","numpy","transformers","safetensors","scikit-learn","scipy")}))


def protocol():
    p = load(OUT / "plan.json")
    for name, sha in p["sources"].items():
        assert file_sha(Path(__file__).with_name(name)) == sha, name
    for name, entry in p["materials"].items():
        assert file_sha(OUT / name) == entry["sha256"], name
    for package, version in p["runtime"].items():
        assert importlib.metadata.version(package) == version
    return p


def prompts(row):
    head = "Choose one option. Output only its letter.\n"
    cuts = []
    for piece in row["pieces"]:
        head += piece + "\nRecord ends:"
        cuts.append(len(head))
        head += "\n"
    return head + row["ending"] + "\nAnswer:", cuts


def encode(tokenizer, row):
    prompt, cuts = prompts(row)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    sites = []
    for cut in cuts:
        marker_start = cut - len("Record ends:")
        for i in range(len(ids)-1,-1,-1):
            a,b = offsets[i]
            if marker_start < b <= cut and tokenizer.encode(prompt[:b], add_special_tokens=False) == ids[:i+1]:
                sites.append(i)
                break
        else:
            raise ValueError(f"No prefix-stable token in readout marker for {row['id']}")
    assert offsets[-1][1] == len(prompt)
    return ids, sites + [len(ids)-1]


class Engine:
    def __init__(self, key):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.key = key
        where = ROOT / "models/hf" / M.MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(where, local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        if key == "q4":
            kwargs["device_map"] = {"": 0}
        else:
            kwargs.update(device_map="auto", max_memory={0:"11GiB", "cpu":"28GiB"}, offload_folder=str(TMP),
                          offload_buffers=False)
        self.model = AutoModelForCausalLM.from_pretrained(where, **kwargs).eval()
        placement = getattr(self.model, "hf_device_map", None)
        if placement is None:
            devices = {str(parameter.device) for parameter in self.model.parameters()}
            assert devices == {"cuda:0"}, devices
            placement = {"": "cuda:0"}
        assert all(device != "disk" for device in placement.values()), placement
        self.layers = self.model.model.layers
        self.letters = [self.tok.encode(" " + letter, add_special_tokens=False) for letter in "ABCD"]
        assert all(len(tokens) == 1 for tokens in self.letters)
        self.letters = [tokens[0] for tokens in self.letters]
        self.dimension = self.model.config.hidden_size
        self.placement = placement

    def close(self):
        import torch
        del self.layers, self.model
        gc.collect()
        torch.cuda.empty_cache()

    def batch(self, rows, panel, capture, field_rows, lexical_sites=True, edits=None, generate=False):
        import numpy as np
        import torch
        tokens_and_sites = [encode(self.tok, row) for row in rows]
        ids, sites = zip(*tokens_and_sites)
        batch = self.tok.pad(dict(input_ids=list(ids)), padding=True, return_tensors="pt").to("cuda")
        width = batch.input_ids.shape[1]
        pos = torch.as_tensor([[i + width - len(r) for i in s] for r,s in zip(ids,sites)],device="cuda")
        batch_indices = torch.arange(len(rows), device="cuda")[:,None]
        outputs = {}
        hooks = []

        def bits(value):
            return value.detach().contiguous().view(torch.uint16).cpu().numpy().copy()

        def hidden(name, val, module, inputs, result):
            if isinstance(result, tuple):
                result = result[0]
            site_values = result[batch_indices, pos]
            if field_rows:
                outputs[name] = bits(site_values[field_rows])
            if val == "probe":
                outputs["probe"] = bits(site_values)
            if val == "embed":
                if panel == "chain":
                    outputs["embed"] = bits(site_values)
                    masked = result.float() * (torch.arange(width,device="cuda")[None,None,:] <= pos[:,:,None]).to(torch.float32)[:,0,:,None]
                    mask = batch.attention_mask[:,:,None]
                    outputs["prefix0"] = bits(((masked*mask).sum(dim=1) / ((pos[:,0] - (width-batch.attention_mask.sum(1))+1)[:,None])).to(torch.bfloat16))
                else:
                    valid = (torch.arange(width,device="cuda")[None,:] <= pos[:,0,None]) & batch.attention_mask.bool()
                    pooled = (result.float()*valid[:,:,None]).sum(1) / valid.sum(1)[:,None]
                    outputs["prefix0"] = bits(pooled.to(torch.bfloat16))

        def axis(name, unit, module, args, result=None):
            signal = args[0] if result is None else result
            samples = signal[batch_indices,pos]
            if edits is not None and name == "A":
                modified = signal.clone()
                modified[torch.arange(len(rows),device="cuda"),pos[:,0],edits] = 0
                return (modified,)
            outputs[name] = bits(samples[...,unit])
            if name == "A":
                controls = [(unit + 37 * j) % signal.shape[-1] for j in range(1,65)]
                outputs["controls"] = bits(samples[:,0,controls])
                outputs["selected_norm"] = float(module.weight[:,unit].float().norm())
                outputs["control_norms"] = module.weight[:,controls].float().norm(dim=0).cpu().tolist()

        if capture and panel == "chain":
            from functools import partial
            hooks.append(self.model.model.embed_tokens.register_forward_hook(partial(hidden,"embed_full","embed")))
            for block, layer in enumerate(self.layers):
                if field_rows or block == 24:
                    hooks.append(layer.register_forward_hook(partial(hidden,f"block{block}","probe" if block==24 else "")))
            if field_rows:
                hooks.append(self.model.model.norm.register_forward_hook(partial(hidden,"final_norm","")))
        elif capture and panel == "candidate":
            from functools import partial
            block, unit, _ = M.CANDIDATES[rows[0]["task"]]
            mlp = self.layers[block].mlp
            if edits is None:
                hooks.append(self.model.model.embed_tokens.register_forward_hook(partial(hidden,"embed_full","embed")))
                hooks.append(mlp.gate_proj.register_forward_hook(partial(axis,"G",unit)))
                hooks.append(mlp.up_proj.register_forward_hook(partial(axis,"V",unit)))
            hooks.append(mlp.down_proj.register_forward_pre_hook(partial(axis,"A",unit)))

        try:
            logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        finally:
            for hook in hooks:
                hook.remove()
        best = logits.argmax(-1).cpu().tolist()
        letter_scores = logits[:, self.letters].cpu().tolist()
        norm = logits.logsumexp(-1).cpu().tolist()
        generated = self.model.generate(**batch, max_new_tokens=8, do_sample=False, use_cache=True,
                                        pad_token_id=self.tok.pad_token_id)[:,width:].cpu().tolist() if generate else [None]*len(rows)
        records = []
        for i,(r,s,raw,scores,total,top,g) in enumerate(zip(rows,sites,ids,letter_scores,norm,best,generated)):
            options = [float(score) for score in scores[:4 if panel == "chain" else 2]]
            true = r["answer"]
            correct = options[true] > max(x for j,x in enumerate(options) if j!=true)
            answer = None if g is None else self.tok.decode(g,skip_special_tokens=True).strip()
            records.append(dict(id=r["id"], tokens=raw, sites=s, option_logits=options, total_logsumexp=total,
                                top_token=top, correct=correct, generated=g, natural_answer=answer,
                                natural_correct=None if g is None else answer == "ABCD"[true]))
        return records, outputs


def cell_pass(rows, results, behavior_floor):
    from collections import defaultdict
    counts = defaultdict(list)
    for row,result in zip(rows,results,strict=True):
        counts[(row["task"],*row["factors"])].append(result["correct"])
    return [dict(cell=cell,n=len(values),correct=sum(values),pass_gate=sum(values)/len(values) >= behavior_floor)
            for cell,values in counts.items()]


def blocks(rows, n):
    from itertools import groupby
    for _, seq in groupby(rows, key=lambda row:row["task"]):
        group = list(seq)
        for i in range(0,len(group),n):
            yield group[i:i+n]


def execute(key):
    import numpy as np
    import torch
    p = protocol()
    path = OUT / f"{key}-model.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    with GpuLease(), torch.inference_mode():
        total_start = time.monotonic()
        stages = identity(key)
        e = Engine(key)
        try:
            write(path.name, dict(files=stages,snapshot=digest(stages),letters=e.letters,
                                  placement=e.placement,plan_sha=file_sha(OUT/"plan.json")))
            for split in M.SPLITS:
                for panel in ("chain","candidate"):
                    if key != "q4" and (panel == "candidate" or split == "discover"):
                        continue
                    stage = f"{key}-{split}-{panel}"
                    rows = load(OUT / f"materials-{split}-{panel}.json")
                    batches = list(blocks(rows,64))
                    results = []
                    begin = time.monotonic()
                    stage_log(key,stage,0,len(batches),begin)
                    for i, batch in enumerate(batches,1):
                        raw,_ = e.batch(batch,panel,False,[],generate=split=="pilot")
                        results.extend(raw)
                        stage_log(key,stage,i,len(batches),begin)
                    gate = cell_pass(rows,results,p["thresholds"]["behavior"])
                    write(stage+".json",dict(gate=gate, results=results, seconds=time.monotonic()-begin))
            collect(key,e,p)
        finally:
            e.close()
        stage_log(key,"complete",1,1,total_start)


def collect(key, e, p):
    if key != "q4":
        return
    import numpy as np
    for panel in ("chain","candidate"):
        for split in ("discover","words","frames"):
            rows = load(OUT / f"materials-{split}-{panel}.json")
            eligible = set(M.CANDIDATES if panel=="candidate" else ["stock","calendar"])
            for check in ("pilot",split):
                available = load(OUT / f"{key}-{check}-{panel}.json")
                eligible -= {r["cell"][0] for r in available["gate"] if not r["pass_gate"]}
            rows = [r for r in rows if r["task"] in eligible]
            batches = list(blocks(rows,64))
            base = {r["id"]:r for r in load(OUT/f"{key}-{split}-{panel}.json")["results"]}
            stage = f"{key}-{split}-{panel}-native"
            start = time.monotonic()
            stage_log(key,stage,0,len(batches),start)
            for i,batch in enumerate(batches,1):
                sample = [j for j,r in enumerate(batch) if r["index"]<2] if panel=="chain" else []
                raw,native=e.batch(batch,panel,True,sample)
                assert all(base[r["id"]]["option_logits"] == r["option_logits"] for r in raw)
                if panel=="candidate" and split!="discover":
                    native.pop("controls")
                if panel=="chain":
                    native.pop("prefix0")
                np.savez(check_write(OUT/f"{stage}-{i:03d}.npz"), **native,
                         ids=[r["id"] for r in raw], field_indices=sample)
                stage_log(key,stage,i,len(batches),start)
            write(stage+".json",dict(eligible=sorted(eligible),family_count=len({r["family"] for r in rows}),batches=len(batches)))


def intervene():
    import torch
    p = protocol()
    analysis = load(OUT/"analysis.json")
    assert analysis["plan_sha"] == file_sha(OUT/"plan.json")
    eligible = [r for r in analysis["candidates"] if r.get("pass_response")]
    if not eligible:
        write("interventions.json",dict(status="not admitted",results=[]))
        return
    assert not (OUT/"interventions.json").exists()
    info = load(OUT/"q4-model.json")
    with GpuLease(),torch.inference_mode():
        assert identity("q4")==info["files"]
        engine = Engine("q4")
        experiments = []
        try:
            from condition_chain_analysis import margin, stats
            for candidate in eligible:
                for split in ("words","frames"):
                    start = time.monotonic()
                    rows = [r for r in load(OUT/f"materials-{split}-candidate.json") if r["task"]==candidate["task"]]
                    batches=list(blocks(rows,64))
                    base = {r["id"]:r for r in load(OUT/f"q4-{split}-candidate.json")["results"]}
                    records=[]
                    stage_log("q4",f"edit-{candidate['task']}-{split}",0,len(batches),start)
                    for i,batch in enumerate(batches,1):
                        common,_ = engine.batch(batch,"candidate",False,[])
                        assert all(base[o["id"]]["option_logits"]==o["option_logits"] for o in common)
                        ablate,_ = engine.batch(batch,"candidate",True,[],edits=candidate["unit"])
                        control,_ = engine.batch(batch,"candidate",True,[],edits=candidate["match"])
                        for r,a,b in zip(batch,ablate,control,strict=True):
                            records.append(dict(id=r["id"],family=r["family"],target=r["factors"][0],
                                                correct_option=r["answer"],baseline=base[r["id"]]["option_logits"],
                                                ablation=a["option_logits"],control=b["option_logits"]))
                        stage_log("q4",f"edit-{candidate['task']}-{split}",i,len(batches),start)
                    from collections import defaultdict
                    families = defaultdict(list)
                    for r in records:
                        if r["target"]:
                            j = r["correct_option"]
                            base_score = margin({"option_logits":r["baseline"]},j)
                            families[r["family"]].append([base_score-margin({"option_logits":r[arm]},j) for arm in ("ablation","control")])
                    import numpy as np
                    pairs = np.array([np.mean(v,axis=0) for v in families.values()])
                    target,control,specific = stats(pairs[:,0]),stats(pairs[:,1]),stats(pairs[:,0]-pairs[:,1])
                    limits=p["thresholds"]
                    passed=(target["mean"]>=limits["min_patch"] and specific["mean"]>=limits["min_patch"]
                            and specific["p"]<limits["alpha"] and abs(control["mean"])<=limits["max_control"]
                            and all(.5<=candidate[k]<=2 for k in ("match_rms_ratio","match_norm_ratio")))
                    experiments.append(dict(task=candidate["task"],split=split,passed=passed,target=target,
                                            control=control,specific=specific,records=records))
        finally:
            engine.close()
    write("interventions.json",dict(status="complete",results=experiments))


if __name__ == "__main__":
    setup()
    if sys.argv[1] == "freeze":
        preregister()
    elif sys.argv[1] == "run":
        execute(sys.argv[2])
    elif sys.argv[1] == "intervene":
        intervene()
