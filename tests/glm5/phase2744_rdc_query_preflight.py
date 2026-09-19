"""Material/tokenization checks only; does not admit or run another model job."""
from collections import Counter
from phase2744_rdc_query_identifiability import *


def main():
    from transformers import AutoTokenizer
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);reports=[]
    for family in FAMILIES:
      for case in range(16):
       for lang in ['en','zh']:
        ids=[];answers=[];questions=[]
        for world in [0,1]:
            body,question,truth,edges=recipe(family,case,lang,world)
            text=body+'\n'+question+'\n'+('Answer only Yes or No.' if lang=='en' else '只回答是或否。')
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids.append(tok(prompt,add_special_tokens=False)['input_ids']);answers.append(truth);questions.append(question)
        reports.append({'family':family,'case':case,'language':lang,'full_token_histogram_equal':Counter(ids[0])==Counter(ids[1]),
          'question_equal':questions[0]==questions[1],'answer_reversed':answers[0]!=answers[1],'lengths':list(map(len,ids)),
          'different_token_counts':dict(Counter(ids[0])-Counter(ids[1]))})
    result={'timestamp':stamp(),'source':snapshot(__file__),'language_scoring':language_checks(),'material_source':snapshot(Path(__file__).with_name('phase2744_rdc_query_identifiability.py')),
      'all_passed':all(r['full_token_histogram_equal'] and r['question_equal'] and r['answer_reversed'] for r in reports),'pairs':reports,
      'scope':'Tokenizer-only verification, before materialfreeze and any new native outcomes. Does not assert that the prospective whole stage has been admitted.',
      'seconds':time.monotonic()-start}
    previous=OUT/'preflight.json'
    if previous.exists():save(OUT/'preflight_history'/(sha(previous)+'.json'),read(previous))
    save(previous,result);ledger('identity_tokenizer_preflight',result['seconds']);print('IDENTITY_PREFLIGHT',result['all_passed'],len(reports),[r for r in reports if not r['full_token_histogram_equal']],flush=True)
    assert result['all_passed']


if __name__=='__main__':main()
