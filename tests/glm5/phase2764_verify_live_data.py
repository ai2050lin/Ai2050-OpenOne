"""CPU-only integration check and archive backfill of EXISTING research, no model inference.

Run from the repository root. Writes only the center research index and this
phase's verification manifest. Does not register feature claims automatically.
"""
from pathlib import Path
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deploy'))
from deploy import research_library as R
from deploy.parameter_reader import catalog,header,read_slice


def main():
    before=R.library().summary()
    sync=R.sync()
    after=R.library().summary()
    progress=R.progress()
    checks=[]
    names={'qwen3-4b','glm-4-9b-chat-hf','qwen3-14b'}
    for model in catalog([Path('D:/develop/model'),ROOT/'models'/'hf']):
        if model['name'].lower() not in names:continue
        cfg=model['config'];values=[]
        for file in sorted(model['_path'].glob('*.safetensors')):
            tensors,_,_=header(file)
            for name,meta in tensors.items():
                if f"layers.{cfg['num_hidden_layers']-1}." not in name or not ('.self_attn.' in name or '.mlp.' in name):continue
                if len(meta['shape'])!=2:continue
                value=read_slice(model['_path'],name,meta['shape'][0]-1,meta['shape'][1]-1,1,1)
                values.append(value)
        checks.append({'model':model['name'],'local_id':model['id'],
                       'config':{k:cfg[k] for k in ('hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','vocab_size')},
                       'last_layer_parameter_checks':values})
    report={'kind':'software_integration_check','created_at':time.time(),'inference':False,
            'archive_before':before['artifacts'],'archive_after':after['artifacts'],'sync':sync,
            'graph_revisions':after['graph_revisions'],'progress':progress,'models':checks}
    target=ROOT/'tests/glm5/result/phase2764_live_integration.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'artifacts':after['artifacts'],'new_report_revisions':sync['new_revisions'],'errors':len(sync['errors']),
                      'graph_revisions':after['graph_revisions'],'model_dirs_checked':len(checks),'progress_buckets':len(progress['buckets'])},ensure_ascii=False))
    return 1 if sync['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
