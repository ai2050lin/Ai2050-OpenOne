import json, os, torch
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"D:/develop/model/Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained(r"D:/develop/model/Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0").eval()
cap=[json.loads(x) for x in Path(r"D:/AI2050/Ai2050-OpenOne/data/lang_bench/q9b_capgate.v1.jsonl").read_text(encoding="utf-8").splitlines()][:3]
print("letter ids:", {c: tok(c, add_special_tokens=False).input_ids for c in "ABCD"})
for q in cap:
    p = f"{q['question']}\nA. {q['choices'][0]}\nB. {q['choices'][1]}\nC. {q['choices'][2]}\nD. {q['choices'][3]}\n答案："
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): lg = m(input_ids=ids).logits[0,-1].float()
    top = torch.topk(lg, 8)
    print("Q:", q["question"][:50], "| gold:", q["answer"])
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print("   ", round(v,2), repr(tok.decode([i])))
    print("   logits A/B/C/D:", [round(float(lg[tok(c,add_special_tokens=False).input_ids[0]]),2) for c in "ABCD"])
