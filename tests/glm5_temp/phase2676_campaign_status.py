"""Compact durable campaign status; no model import, no writes or busy model polling."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'tests/glm5/result';O=R/'phase2676_native_mlp_delivery'
rows={}
for key,path in [('pipeline',O/'analysis/pipeline.json'),('tail',O/'analysis/tail_pipeline.json'),
    ('scalar',R/'phase2674_native_mlp_scalar/analysis/progress.json'),
    *[(k,R/f'phase2675_native_mlp_crossmodel/{k}/analysis/progress.json') for k in ('qwen14','glm4','ds7')],
    ('chronology',O/'expansion/analysis/progress.json'),('resolution',O/'numeric_resolution/analysis/progress.json')]:
    if path.exists():rows[key]=json.loads(path.read_text(encoding='utf-8'))
rows['completed_phases']=[p for p in range(2670,2677) if list(R.glob(f'phase{p}_*/analysis/final.json'))]
print(json.dumps(rows,ensure_ascii=True))
