"""Phase 960 minimal: single head decomposition for GLM4."""
import sys, gc, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np, torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir

RESULT_DIR = Path("results/phase960_single_head")
PROMPTS = [
    "The capital of France is",
    "What is water?",
    "The sky is blue.",
    "How many continents?",
    "The largest ocean is",
]
MAX_TOKENS = 30

HEADS = [
    {"layer": 39, "head": 21, "d_head": 128, "name": "L39_H21"},
    {"layer": 38, "head": 0, "d_head": 128, "name": "L38_H0"},
    {"layer": 38, "head": 7, "d_head": 128, "name": "L38_H7"},
]

EXPECTED = {
    "The capital of France is": "Paris",
    "How many continents?": "seven",
    "The largest ocean is": "Pacific",
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def make_hook(sc, ec, scale):
    def hook(mod, args):
        inp = args[0] if isinstance(args, tuple) else args
        p = inp.clone()
        if p.ndim >= 3:
            if scale == 0: p[:, :, sc:ec] = 0
            else: p[:, :, sc:ec] = p[:, :, sc:ec] * scale
        return (p,)
    return hook

def gen(model, tokenizer, device, prompt, layers, head_info=None, scale=0.0, all_heads=None):
    handles = []
    if head_info == "all":
        for h in (all_heads or HEADS):
            sc = h["head"] * h["d_head"]
            ec = sc + h["d_head"]
            handles.append(layers[h["layer"]].self_attn.o_proj.register_forward_pre_hook(
                make_hook(sc, ec, scale)))
    elif head_info is not None:
        sc = head_info["head"] * head_info["d_head"]
        ec = sc + head_info["d_head"]
        handles.append(layers[head_info["layer"]].self_attn.o_proj.register_forward_pre_hook(
            make_hook(sc, ec, scale)))

    try:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        gen_tokens = out[0][ids.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
        n = len(gen_tokens)
    except Exception as e:
        text = f"ERROR: {e}"; eos = False; n = 0
    for h in handles: h.remove()
    return text, eos, n

def eval_clean(prompt, text, eos, n):
    exp = EXPECTED.get(prompt, "")
    ascii_ok = all(ord(c) < 256 for c in text)
    short = 0 < n < 15
    has_exp = exp == "" or exp.lower() in text.lower()
    return eos and short and has_exp and ascii_ok, eos, ascii_ok, short, has_exp

def run_model(name):
    log(f"\n=== {name} ===")
    model_dir = RESULT_DIR / name
    ensure_dir(model_dir)
    model, tokenizer, device = load_model(name)
    info = get_model_info(model, name)
    layers = get_layers(model)
    log(f"  {info.model_class}, {info.n_layers}L")

    heads = HEADS if name == "glm4" else (
        [{"layer": 35, "head": 0, "d_head": 80, "name": "L35_H0"},
         {"layer": 33, "head": 8, "d_head": 80, "name": "L33_H8"}] if name == "qwen3" else
        [{"layer": 26, "head": 19, "d_head": 112, "name": "L26_H19"},
         {"layer": 26, "head": 25, "d_head": 112, "name": "L26_H25"}]
    )

    conditions = [("normal", None, 0.0)]
    for h in heads:
        conditions.append((f"ablate_{h['name']}", h, 0.0))
        conditions.append((f"boost2.0_{h['name']}", h, 2.0))
    conditions.append(("ablate_all", "all", 0.0))
    conditions.append(("boost2.0_all", "all", 2.0))

    results = []
    for pi, prompt in enumerate(PROMPTS):
        for cname, hinfo, scale in conditions:
            text, eos, n = gen(model, tokenizer, device, prompt, layers, hinfo, scale, all_heads=heads)
            clean, _, ascii_ok, short, has_exp = eval_clean(prompt, text, eos, n)
            results.append({
                "prompt": prompt, "condition": cname,
                "text": text[:200], "eos": eos, "n_tokens": n,
                "strict_clean": clean, "lang_switch": not ascii_ok,
                "periods": text.count("."),
            })
        log(f"  {pi+1}/{len(PROMPTS)} prompts")

    # Aggregate
    agg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "tokens": []})
    for r in results:
        a = agg[r["condition"]]
        a["eos"] += int(r["eos"])
        a["clean"] += int(r["strict_clean"])
        a["switch"] += int(r["lang_switch"])
        a["n"] += 1
        a["tokens"].append(r["n_tokens"])

    summary = {c: {
        "eos_rate": d["eos"]/max(d["n"],1),
        "clean_rate": d["clean"]/max(d["n"],1),
        "switch_rate": d["switch"]/max(d["n"],1),
        "mean_tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
    } for c, d in agg.items()}

    output = {"model": name, "n_prompts": len(PROMPTS), "max_tokens": MAX_TOKENS,
              "summary": summary, "raw_results": results}
    (model_dir / "single_head_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"  Summary:")
    for c in [c[0] for c in conditions]:
        if c in summary:
            s = summary[c]
            log(f"    {c:25s}: eos={s['eos_rate']:.2f}  clean={s['clean_rate']:.2f}  "
                f"switch={s['switch_rate']:.2f}  tokens={s['mean_tokens']:.1f}")

    log(f"  Sample (prompt 0):")
    for r in results:
        if r["prompt"] == PROMPTS[0] and r["condition"] in ["normal", "ablate_all", "boost2.0_all"]:
            log(f"    {r['condition']:25s}: eos={r['eos']}  text={r['text'][:80]}")

    release_model(model)
    log(f"  {name} done")

for m in ["glm4", "qwen3", "deepseek7b"]:
    run_model(m)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

log("Phase 960 complete!")
