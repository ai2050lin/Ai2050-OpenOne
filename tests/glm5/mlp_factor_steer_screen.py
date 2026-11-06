"""Enroll independent clean-correct low-margin cases after the writer recipe is locked."""
import numpy as np
import torch

import mlp_factor_steer_common as C
from mlp_factor_steer_fit import inputs
from mlp_kv_behavior import score, word_indices
from mlp_read_route_causal import branch
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease
from tools.power_analysis import check_phase


def main():
    output=C.OUT/'screen.json.gz'
    if output.exists():return
    corpus,plan=inputs();frozen=C.read(C.OUT/'frozen-recipe.json')
    execution={**C.identity(),'schema':'factor_steer_screen_execution.v1','source_sha256':C.file_sha(__file__),
               'recipe_sha256':C.file_sha(C.OUT/'frozen-recipe.json'),'inputs_sha256':plan['inputs_sha256'],'layout_batch_size':4}
    C.save(C.OUT/'screen-execution.json',execution)
    torch.set_num_threads(4);summary={};screen={};enrolled={}
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok)
        for task in ['lookup','dispatch']:
            items=[r for r in corpus['heldout'] if r['task']==task];assert len(items)==C.PER_TASK
            rows=[];selected=[]
            for lo in range(0,len(items),4):
                batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
                cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
                logits,tokens=branch(model,x,cache,eos);del cache
                for j,r in enumerate(batch):
                    v=score(r,logits[j],tokens[j],tok,words);eligible=v['correct'] and 0<v['gap']<=8
                    rows.append({'index':lo+j,'id':r['id'],'source_id':r['source_id'],'expected':r['expected'],'clean':v,'eligible':eligible})
                    if eligible and len(selected)<104:selected.append(lo+j)
                if (lo+4)%192==0:print(task,'screened',lo+4,'eligible',len(selected),flush=True)
                if len(selected)==104:break
            screen[task]=rows;enrolled[task]=selected
            power=check_phase(.10,len(selected),alpha=.025) if selected else {'n':0,'n_required':plan['power_plan']['n_required'],'pass':False}
            summary[task]={'screened':len(rows),'available':len(items),'clean_accuracy':float(np.mean([r['clean']['correct'] for r in rows])),
                           'enrolled':len(selected),'power':power,'screen_median_gap':float(np.median([r['clean']['gap'] for r in rows]))}
            print(task,summary[task],flush=True)
    C.write_gz(output,{**execution,'status':'complete','selected_recipe':frozen['recipe'],'rows':screen,'enrolled':enrolled,'summary':summary})


if __name__=='__main__':main()
