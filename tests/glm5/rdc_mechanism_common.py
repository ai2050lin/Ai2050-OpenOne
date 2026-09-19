"""Shared, bounded mechanism campaign utilities; prior frozen experiments remain immutable."""
from rdc_feature_common import *
PREVIOUS=CAMPAIGN
CAMPAIGN=RESULT/'rdc_mechanism_campaign_20260909'

def checkpoint(key):
    from safetensors import safe_open
    folder=ROOT/'models/hf/qwen3-4b'
    index=read(folder/'model.safetensors.index.json')['weight_map']
    with safe_open(str(folder/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)

def announce(run,**kw):
    save(CAMPAIGN/run/'status.json',dict(run_id=run,updated_at=stamp(),**kw))

def events(run,kind,**kw):
    path=CAMPAIGN/run/'events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    n=sum(1 for _ in path.open(encoding='utf-8')) if path.exists() else 0
    row=dict(cursor=n+1,run_id=run,kind=kind,timestamp=stamp(),**kw)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return row
