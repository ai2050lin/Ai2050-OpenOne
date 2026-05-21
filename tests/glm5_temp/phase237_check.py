import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

for m in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        r = json.load(open(f'tests/glm5_temp/phase237_{m}_results.json', encoding='utf-8'))
        print(f'=== {m} ===')
        if 'expB' in r:
            b = r['expB']
            print(f'  ExpB: simple_acc={b.get("simple_accuracy","?")}, entail_acc={b.get("entail_accuracy","?")}, overall={b.get("overall_accuracy","?")}')
        if 'expC' in r:
            c = r['expC']
            print(f'  ExpC k90: {c.get("k90_summary","?")}')
        if 'expA' in r:
            a = r['expA']
            hs = a.get('hidden_svd',{})
            ls = a.get('logit_svd',{})
            print(f'  ExpA: HS k90={hs.get("k90","?")}, LS k90={ls.get("k90","?")}')
            # d_not top tokens
            print(f'  Top boosted (logit): {a.get("top_boosted_logit",[("?","?")])[:5]}')
            print(f'  Top suppressed (logit): {a.get("top_suppressed_logit",[("?","?")])[:5]}')
    except Exception as e:
        print(f'{m}: {e}')
