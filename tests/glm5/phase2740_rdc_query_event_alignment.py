"""Check generated-character event boundaries against exact decoded prefixes."""
from rdc_query_common import *

def main():
    from transformers import AutoTokenizer
    out=BASE/'events';file=out/'alignment_audit.json'
    if file.exists():return
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);reports=[]
    for item in gzread(out/'material.json.gz')['trajectories']:
        record=read(ROOT/item['native_record']);ids=record['generated_ids'];texts=[tok.decode(ids[:j],skip_special_tokens=True) for j in range(1,len(ids)+1)];text=texts[-1]
        for e in item['events']:
            end=e['character_span'][1];exact=next(i for i,s in enumerate(texts) if s.startswith(text[:end]))
            reports.append({'sample_id':item['row']['sample_id'],'type':e['type'],'character_span':e['character_span'],
              'length_based_step':e['emitted_token_step'],'exact_decoded_prefix_step':exact,'equal':exact==e['emitted_token_step']})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':all(r['equal'] for r in reports),'event_boundaries':len(reports),
      'reports':reports,'seconds':time.monotonic()-start,'scope':'Exact cumulative decoded prefix for each terminal/assignment regex endpoint, not inferred subtoken semantics.'}
    save(file,result);ledger('generated_event_exact_character_alignment',result['seconds']);assert result['all_passed']
    print('EXACT_EVENT_ALIGNMENT',len(reports),flush=True)

if __name__=='__main__':main()
