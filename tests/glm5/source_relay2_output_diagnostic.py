"""Descriptive top-token accounting; no refit, model forward, or changed gate."""
import argparse,collections,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D

def freeze():
    assert not (D.OUT/D.MODEL/'lockbox_frame-behavior-v2-started.json').exists()
    D.save('output-diagnostic-protocol.json',{'script_sha256':D.sha(Path(__file__)),
        'recipe':'Classify already saved recurrence top1 IDs as discovery-answer vocabulary, heldout-answer vocabulary, or other; count exact tokens and decode them. No new model inference, selection, or normalization.',
        'exposure':'Confirm recurrence already seen; this is post-hoc for confirm. Recipe fixed before lockbox-frame behavior/internal measurement. All primary protocols unchanged.'})

def analyze(split):
    from transformers import AutoTokenizer
    protocol=D.read(D.OUT/'output-diagnostic-protocol.json');assert protocol['script_sha256']==D.sha(Path(__file__))
    materials=D.read(D.OUT/'qwen14-materials.json')['splits'];path=D.OUT/D.MODEL/f'{split}-update-output.json';r=D.read(path)
    train={x for row in materials['discovery'] for x in row['choice_ids']};held={x for s in ['confirm','lockbox_frame'] for row in materials[s] for x in row['choice_ids']}
    assert not (train & held)
    tok=AutoTokenizer.from_pretrained(D.MODELS[D.MODEL],local_files_only=True);reports={}
    for method,rows in r['rows'].items():
        count=collections.Counter(x['top1'] for x in rows)
        a=sum(v for x,v in count.items() if x in train);b=sum(v for x,v in count.items() if x in held)
        reports[method]={'n':len(rows),'discovery_answer_vocabulary':a,'heldout_answer_vocabulary':b,'other':len(rows)-a-b,
            'token_counts':[{'id':x,'text':tok.decode([x]),'n':n} for x,n in count.most_common()],
            'actual_model_top1_agreement':sum(x['top1_matches_model'] for x in rows)}
    D.save(f'{D.MODEL}/{split}-output-diagnostic.json',{'methods':reports,'input_sha256':D.sha(path),'protocol_sha256':D.sha(D.OUT/'output-diagnostic-protocol.json'),
        'scope':'Token-level descriptive counts, not natural generation or causal explanation of the failed recurrence. Nothing is filtered or rescored.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','analyze']);ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:analyze(a.split)
