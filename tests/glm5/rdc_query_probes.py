"""A finite, explicit diagnostic query set; strings are not asserted single tokens."""
from rdc_query_common import *

EN={
 'syntax':[' the',' a',' this',' these',' was',' were',' has',' have',' with',' without'],
 'causal':[' because',' therefore',' although',' however',' unless',' if',' then',' instead',' despite',' otherwise'],
 'role':[' he',' she',' they',' it',' who',' whom',' whose',' to whom',' by whom',' for whom'],
 'relation':[' is a',' is part of',' belongs to',' contains',' causes',' comes from',' is used for',' is larger than',' is not',' refers to'],
 'task':['\nAnswer:','\nIn summary,','\nThe result is','\nFor example,','\nFirst,','\nNext,','\nFinally,','\nIn Python:','\nTranslate:','\nExplain:']}
ZH={
 'syntax':['的','一个','这个','这些','是','不是','有','没有','和','在'],
 'causal':['因为','所以','虽然','但是','除非','如果','那么','反而','尽管','否则'],
 'role':['他','她','他们','它','谁','给谁','谁的','由谁','为了谁','对谁'],
 'relation':['属于','是一种','是其中一部分','包含','导致','来自','用于','大于','并非','指的是'],
 'task':['\n答案：','\n总之，','\n结果是','\n例如，','\n首先，','\n接着，','\n最后，','\nPython代码：','\n翻译：','\n解释：']}

def freeze(tok):
    p=BASE/'probes/protocol.json'
    if p.exists():return read(p)['probes']
    rows=[]
    for lang,bank in [('en',EN),('zh',ZH)]:
      for family,texts in bank.items():
        order=sorted(range(10),key=lambda i:rank(lang+'/'+family+'/'+str(i)))
        fit=set(order[:6]);val=set(order[6:8])
        for i,text in enumerate(texts):
            ids=tok(text,add_special_tokens=False)['input_ids'];assert ids
            rows.append({'probe_id':f'{lang}/{family}/{i:02d}','language':lang,'family':family,'text':text,'token_ids':ids,
              'split':'train_query' if i in fit else 'validation_query' if i in val else 'unseen_query'})
    assert len(rows)==100 and len({tuple(r['token_ids']) for r in rows})==100
    immutable(p,{'timestamp':stamp(),'source':snapshot(__file__),'probes':rows,'empty_baseline_separate':True,
      'query_split_counts':{'train_query':60,'validation_query':20,'unseen_query':20},
      'scope':'Artificial fixed suffix interventions spanning100strings, not100single tokens or semantically valid continuations of every natural prefix. Token IDs are appended without joint re-tokenization. No future gold or model outcome selected these strings.',
      'transfer':'Names indicate researcher query categories, not internal module labels. Cross-language strings are not asserted exact semantic equivalents.'})
    return rows

if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);print('QUERY_PROBES',len(freeze(tok)))
