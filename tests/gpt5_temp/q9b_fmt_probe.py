# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
import numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
tok = AutoTokenizer.from_pretrained(r"D:/develop/model/Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained(r"D:/develop/model/Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0").eval()
cap = [json.loads(x) for x in (ROOT/"data/lang_bench/q9b_capgate.v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
cap = cap[:60]
L = "ABCD"
lm = {c: tok(c, add_special_tokens=False).input_ids[0] for c in L}
def fmt(q, mode):
    body = f"{q['question']}\nA. {q['choices'][0]}\nB. {q['choices'][1]}\nC. {q['choices'][2]}\nD. {q['choices'][3]}"
    if mode == "plain": return body + "\n答案："
    if mode == "instr": return "以下是一道单选题，请只回答正确选项的字母。\n" + body + "\n答案："
    if mode == "fewshot":
        return ("以下是一道单选题，请只回答正确选项的字母。\n示例：中国的首都是？\nA. 上海\nB. 北京\nC. 广州\nD. 深圳\n答案：B\n\n" + body + "\n答案：")
    raise ValueError(mode)
for mode in ["plain", "instr", "fewshot"]:
    ok = 0
    for q in cap:
        ids = tok(fmt(q, mode), return_tensors="pt").input_ids[0]
        rows = torch.stack([torch.cat([ids, torch.tensor([lm[c]])]) for c in L]).cuda()
        with torch.no_grad():
            lg = m(input_ids=rows).logits[:, -1, :].float()
        lp = [float(lg[j, lm[c]]) for j, c in enumerate(L)]
        ok += int(np.argmax(lp) == L.index(q["answer"]))
    print(f"{mode}: acc={ok/len(cap):.3f} ({ok}/{len(cap)})", flush=True)
