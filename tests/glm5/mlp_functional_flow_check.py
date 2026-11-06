"""Recheck the temporal boundary of a last-block Head on exposed lockbox cases."""
import numpy as np
import torch

import mlp_functional_common as C
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_functional_causal import edited
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def main():
    out=C.OUT/'revision';dest=out/'flow-check.json'
    if dest.exists():return
    source=out/'generated-causal.json';result=C.read(source)
    assert result['status']=='complete'
    rows=result['rows'];keys=['KL','answer_gap','generated_ids']
    paired={k:sum(r['conditions']['attention_1'][k]==r['conditions']['head34_2'][k] for r in rows) for k in keys}
    corpus={r['id']:r for r in C.read(out/'inputs.json')['rows']}
    cases=[corpus[r['id']] for r in rows[:32]]
    plan={**C.identity(),'schema':'functional_last_block_flow_check.v1','source_sha256':C.file_sha(__file__),
          'causal_result_sha256':C.file_sha(source),'ids':[r['id'] for r in cases],
          'scope':'post-result engineering canary on exposed cases, isolated B35/H25 removal at prequestion token'}
    C.save(out/'flow-check-prereg.json',plan)
    torch.set_num_threads(4);checks=[]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        for lo in range(0,len(cases),4):
            batch=cases[lo:lo+4];i=torch.arange(len(batch),device='cuda')
            x=torch.tensor([r['tokens'] for r in batch],device='cuda')
            p=torch.tensor([r['prequestion'] for r in batch],device='cuda')
            with edited(model,i,p,[],[],0.,True) as r0:
                clean=model(input_ids=x,use_cache=True,logits_to_keep=1)
            with edited(model,i,p,[(35,25)],[],1.,True) as r1:
                edit=model(input_ids=x,use_cache=True,logits_to_keep=1)
            same_logits=torch.equal(clean.logits,edit.logits)
            layer_kv_equal=[torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
                            for a,b in zip(clean.past_key_values.layers,edit.past_key_values.layers)]
            same_generation=greedy(model,clean.past_key_values,clean.logits[:,-1],eos,limit=4)==greedy(model,edit.past_key_values,edit.logits[:,-1],eos,limit=4)
            norm=torch.linalg.vector_norm(r0[0]-r1[0],dim=-1).cpu().tolist()
            checks.append({'ids':[r['id'] for r in batch],'local_residual_delta_L2':norm,'later_query_logits_bitwise_equal':same_logits,
                           'prefix_KV_bitwise_equal_by_layer':layer_kv_equal,'generated_tokens_equal':same_generation})
    summary={'cases':len(cases),'local_residual_changed':sum(x>0 for r in checks for x in r['local_residual_delta_L2']),
             'local_residual_delta_mean_L2':float(np.mean([x for r in checks for x in r['local_residual_delta_L2']])),
             'all_later_query_logits_bitwise_equal':all(r['later_query_logits_bitwise_equal'] for r in checks),
             'all_prefix_KV_bitwise_equal':all(all(r['prefix_KV_bitwise_equal_by_layer']) for r in checks),
             'all_generated_tokens_equal':all(r['generated_tokens_equal'] for r in checks),
             'joint_attention_vs_B34_only_matching_saved_metrics':paired,'joint_comparison_cases':len(rows)}
    C.save(dest,{**plan,'summary':summary,'checks':checks})
    print(summary,flush=True)


if __name__=='__main__':main()
