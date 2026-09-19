"""Installed native Cache API and complete-entry audit unit check (CPU synthetic)."""
from rdc_law_common import *
from rdc_law_live import compare_caches


def main():
    import torch
    from transformers import DynamicCache, Qwen3Config
    start=time.monotonic();config=Qwen3Config.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    left=DynamicCache(config=config);right=DynamicCache(config=config);rng=torch.Generator().manual_seed(2730)
    for b in range(config.num_hidden_layers):
        k=torch.randn((1,2,3,4),generator=rng).to(torch.bfloat16);v=torch.randn((1,2,3,4),generator=rng).to(torch.bfloat16)
        left.update(k.clone(),v.clone(),b);right.update(k.clone(),v.clone(),b)
    first=compare_caches(left,right);assert first['all_bitwise_equal'] and len(first['layers'])==72
    left.layers[17].keys[0,0,0,0]+=1
    second=compare_caches(left,right);changed=[r for r in second['layers'] if not r['bitwise_equal']]
    assert len(changed)==1 and changed[0]['block']==17 and changed[0]['array']=='keys'
    assert first['total_elements']==36*2*1*2*3*4
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'helper_source':snapshot(ROOT/'tests/glm5/rdc_law_live.py'),
        'passed':True,'scope':'Synthetic CPU tensors using installed DynamicCache API and local Qwen3 config; no model weights or language inference loaded. This is an audit-code check, not native cache-dependency evidence.',
        'compared_arrays':len(first['layers']),'compared_elements':first['total_elements'],'single_changed_array':changed[0],
        'seconds':time.monotonic()-start}
    save(BASE/'verification/cache_math.json',result);ledger('cache_API_complete_entry_unit_check',result['seconds'])
    print('LAW_CACHE_API_UNIT_PASS',first['total_elements'],flush=True)


if __name__=='__main__':main()
