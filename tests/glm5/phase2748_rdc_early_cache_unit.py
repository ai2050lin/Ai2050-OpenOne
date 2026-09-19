"""Partial native-cache branches preserve every early state through continuation."""
from rdc_question_common import *
from rdc_question_native import QuestionNative
from rdc_question_predictor import clone_early_cache,early_cache_id


def main():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM,GlmConfig,GlmForCausalLM
    torch.set_num_threads(2)
    checks=[]
    for key,configclass,modelclass in [('qwen3',Qwen3Config,Qwen3ForCausalLM),('glm',GlmConfig,GlmForCausalLM)]:
        torch.manual_seed(2748006)
        config=configclass(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=18,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
            attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2)
        config._attn_implementation='eager'
        model=modelclass(config).to(torch.bfloat16).eval()
        engine=QuestionNative(model,[],device='cpu')
        try:
            ids=torch.tensor([[1,21,22,23,24]])
            complete={'input_ids':ids,'mode':'context','context_positions_local':[1,2,3,4]}
            early={'input_ids':ids,'mode':'context','context_positions_local':[1,2,3,4]}
            fullout=engine.forward([complete])[0]
            earlyout=engine.forward([early],stop_after=13)[0]
            assert np.array_equal(fullout['fields']['context_H12_mean'],earlyout['fields']['context_H12_mean'])
            prior=early_cache_id(early['cache'])
            for suffix in [[31,32,33],[41,42]]:
                a={'input_ids':torch.tensor([suffix]),'mode':'question','cache':clone_cache(complete['cache'],config)}
                b={'input_ids':torch.tensor([suffix]),'mode':'question','cache':clone_early_cache(early['cache'],config)}
                for step in range(3):
                    fa=engine.forward([a])[0]
                    fb=engine.forward([b],stop_after=13)[0]
                    assert fb['postnorm']is None
                    for name in ['H12_last_BF16','native_source_read_BF16']:
                        assert np.array_equal(fa['fields'][name],fb['fields'][name])
                    early_cache_id(b['cache'])
                    a={'input_ids':torch.tensor([[60+step]]),'mode':'history','cache':a['cache']}
                    b={'input_ids':torch.tensor([[60+step]]),'mode':'history','cache':b['cache']}
                    checks.append({'architecture':key,'suffix':suffix,'step':step,'all_early_coordinates_exact':True,'late_cache_slots_uninitialized':True})
            assert early_cache_id(early['cache'])==prior
        finally:engine.close()
    result={'timestamp':stamp(),'all_passed':True,'checks':checks,'source':snapshot(__file__),
        'predictor':snapshot(Path(__file__).with_name('rdc_question_predictor.py')),
        'native':snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'scope':'CPU synthetic whole-vs-13layer branch+2causalcontinuations; actual pretrained CUDAqualification still required.'}
    immutable(OUT/'unit'/('early_cache_'+str(time.time_ns())+'.json'),result)
    save(OUT/'unit/early_cache_current.json',result)
    print('NATURAL_EARLY_CACHE_UNIT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
