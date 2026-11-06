"""Bounded implementation/behavior controls; not a language mechanism study.

Check verbatim instruction following and single-vs-batch generation on the SAME
frozen long inputs before interpreting five failed discovery calibrations.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_natural_loop_v5 as P
W=P.W


def main():
    e=P.make_engine();rows=W.read(P.OUT/'pilot-corpus.json');results=[]
    controls=['请只回答字符串 CONTROL2777，不要添加任何其他文字。',
        '请像平时和好朋友聊天一样，讲三句话解释为什么出门要确认时间和地点，语气要亲近自然。',
        '请用自然口语讲三句话，内容是有10份纸，送来3份，交出去2份，现在还有11份纸。请直接跟同事说，不要写成报告。']
    for i,text in enumerate(controls):
        batch=e.tok.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True,
            enable_thinking=False,tokenize=True,return_tensors='pt',return_dict=True)
        if isinstance(batch,e.torch.Tensor):batch={'input_ids':batch,'attention_mask':e.torch.ones_like(batch)}
        batch={k:v.cuda() for k,v in batch.items()}
        with e.torch.inference_mode():o=e.model.generate(**batch,max_new_tokens=160,do_sample=False,use_cache=True)
        results.append({'control':i,'input':text,'output':e.tok.decode(o[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)})
    for row in rows[:4]:
        assert e.tok.decode(row['token_ids'],skip_special_tokens=False)==row['prompt']
        batch=e.batch([row['token_ids']])
        with e.torch.inference_mode():o=e.model.generate(**batch,max_new_tokens=160,do_sample=False,use_cache=True)
        results.append({'sample_id':row['id'],'mode':'single','output':e.tok.decode(o[0,len(row['token_ids']):],skip_special_tokens=True)})
    W.save(P.OUT/'implementation-diagnostic.json',{'kind':'implementation_diagnostic','not_mechanism_evidence':True,'results':results},True)
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
