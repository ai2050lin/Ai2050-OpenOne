"""Debug tokenization - standalone"""
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    trust_remote_code=True, local_files_only=True, use_fast=False)

for sent, target in [('the person is happy', 'happy'), ('the person is not happy', 'happy'), ('that thing is not big', 'big')]:
    toks = tok.encode(sent, add_special_tokens=False)
    found = False
    for prefix in ['', ' ']:
        target_ids = tok.encode(prefix + target, add_special_tokens=False)
        decoded_sent = [tok.decode([x]) for x in toks]
        print(f'  sent="{sent}" prefix="{prefix}" target_ids={target_ids} decoded_sent={decoded_sent}')
        for i in range(len(toks) - len(target_ids) + 1):
            if toks[i:i+len(target_ids)] == target_ids:
                print(f'    MATCH at pos {i} (with BOS: {i+1})')
                found = True
    if not found:
        for i in range(len(toks) - 1, -1, -1):
            decoded = tok.decode([toks[i]]).strip().lower()
            if target.lower() in decoded:
                print(f'    FUZZY at pos {i} (with BOS: {i+1}), decoded="{decoded}"')
                found = True
                break
    if not found:
        print(f'    NO MATCH!')
