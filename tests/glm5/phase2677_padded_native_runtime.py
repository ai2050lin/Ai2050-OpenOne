"""Fixed execution shape with explicitly indexed, unmasked native observations.

Right padding is a numerical control, not a semantic intervention. No padding
token enters an observed-token moment, published trajectory, or query boundary.
Native model arithmetic is untouched; FP64 is only the analysis accumulator.
"""
import numpy as np
import torch
from phase2671_native_mlp_field import Capture,bits

PAD_LENGTH=160
MAX_NEW_TOKENS=16


class PaddedCapture(Capture):
    def reset(self,body,published,task=None):
        super().reset(body,published)
        self.task=task

    def take(self,key,l,t,moments=True):
        if not self.enabled:return
        assert self.task is not None and 0<=self.body<=self.task<t.shape[1]
        # Parent expects its last index to be the real last token. Trim the
        # analysis view only, never the tensor used by the model forward.
        return super().take(key,l,t[:,:self.task+1],moments=moments)


def padded_inputs(model,ids,pad_id,total=PAD_LENGTH):
    assert 0<len(ids)<=total
    device=model.get_input_embeddings().weight.device
    return {'input_ids':torch.tensor([list(ids)+[pad_id]*(total-len(ids))],device=device),
            'attention_mask':torch.tensor([[1]*len(ids)+[0]*(total-len(ids))],device=device),
            'use_cache':False}


def native_pack(pack,published,parameter_published=False):
    # Every case keeps both boundaries of ALL H coordinates and ALL MLP units.
    # Full-token fields are saved only for the prospectively published cases.
    return {k:v for k,v in pack.items() if k in ('h','a') or (published and k=='full__h') or (parameter_published and k.startswith('full__'))}


def group_key(row):
    return f'{row["family"]}_{row["language"]}_{row["field_set"]}_{row["output_function"]}'


def summarize_behavior(records):
    groups={}
    for row in records:
        key=f'{row["family"]}/{row["language"]}/{row["output_function"]}/q{row["polarity"]}'
        d=groups.setdefault(key,{'n':0,'content_correct':0,'strict_correct':0,'eos':0})
        d['n']+=1
        for k in ('content_correct','strict_correct','eos'):d[k]+=int(row[k])
    return groups
