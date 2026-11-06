"""Recorded container-only compatibility adapter; frozen W6 science unchanged.

Transformers 5.14.1 apply_chat_template(tokenize=True) returns BatchEncoding.
The frozen benchmark method expects the identical input_ids list. Preserve the
frozen producer byte-for-byte and adapt this return container at the boundary.
"""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W


def ids_only(result):
    if isinstance(result,dict) or hasattr(result,'keys'):
        return result['input_ids']
    return result


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(W.MODEL,local_files_only=True)
    messages=[{'role':'user','content':'软件兼容检查：请选择 A。'}]
    result=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    ids=ids_only(result)
    assert isinstance(ids,list) and all(isinstance(i,int) for i in ids)
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert ids==tok.encode(text,add_special_tokens=False),'Container-only repair must preserve exact token IDs'
    amendment={'schema':'execution_compatibility_amendment.v1','run_id':W.RUN,
        'frozen_script_sha256':W.sha(W.__file__),'adapter_sha256':W.sha(__file__),
        'scope':'BatchEncoding to identical input_ids list only',
        'criteria_changed':False,'selection_changed':False,'input_token_ids_changed':False,
        'reason':'Capability entry raised ValueError before evaluating its first batch',
        'registered_at':time.time()}
    p=W.OUT/'execution-amendment.json'
    if not p.exists():W.save(p,amendment,True)
    assert W.read(p)['adapter_sha256']==W.sha(__file__)
    original_init=W.Engine.__init__
    def initialize(self):
        original_init(self)
        original=self.tok.apply_chat_template
        def compatible(*args,**kwargs):
            result=original(*args,**kwargs)
            return ids_only(result) if kwargs.get('tokenize',True) else result
        self.tok.apply_chat_template=compatible
    W.Engine.__init__=initialize
    W.run()


if __name__=='__main__':main()
