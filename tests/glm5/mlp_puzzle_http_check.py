"""Read only local center visibility checks; no network writes or GPU calls."""
import json
from urllib.request import urlopen
from pathlib import Path
OUT=Path(__file__).resolve().parent/'result/mlp_puzzle_v1'
results=[]
for port in [5001,5032,5173]:
    try:
        endpoint='/api/traces/research-space' if port!=5173 else '/workbench'
        with urlopen(f'http://127.0.0.1:{port}'+endpoint,timeout=10) as r:
            body=r.read()
            data=json.loads(body) if port!=5173 else {}
            results.append({'port':port,'status':r.status,'bytes':len(body),
              'puzzle_studies':[s for s in data.get('studies',[]) if s['run_id'].startswith('MLP-NATIVE-PUZZLES')]})
    except Exception as e:results.append({'port':port,'error':str(e)})
(OUT/'http-check.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False))
