"""CPU random-weight arithmetic fixtures, never passed off as language evidence."""
from rdc_construction_common import *
from rdc_question_native import QuestionNative, native_whole_B1


def main():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, GlmConfig, GlmForCausalLM
    torch.set_num_threads(2)
    start = time.monotonic()
    checks = []
    for kind, config_class, model_class in [('qwen3', Qwen3Config, Qwen3ForCausalLM), ('glm', GlmConfig, GlmForCausalLM)]:
        torch.manual_seed(274801)
        config = config_class(vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=25, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256, attention_dropout=0., pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config._attn_implementation = 'eager'
        model = model_class(config).to(dtype=torch.bfloat16).eval()
        engine = QuestionNative(model, [6, 12, 24], device='cpu')
        try:
            prefix = [3, 6, 4, 9, 11, 5, 8, 7, 12, 14, 19]
            suffixes = [[21, 22, 23, 24], [31, 32, 33], [41, 42], [51, 52, 53, 54, 55]]
            for suffix in suffixes:
                ids = torch.tensor([prefix+suffix])
                ref = native_whole_B1(model, ids)
                wave = engine.forward([{'input_ids': ids, 'mode': 'plain', 'collect_hidden': True}])[0]['fields']
                assert all(np.array_equal(ref[k], wave[k]) for k in ref)
                checks.append({'model_type': kind, 'test': 'native_whole_B1_every_H_and_postnorm', 'suffix_length': len(suffix), 'passed': True})
            context = {'input_ids': torch.tensor([prefix]), 'mode': 'context', 'collect_hidden': True,
                       'full_prefix_H': True, 'context_positions_local': list(range(3, 11))}
            source = engine.forward([context])[0]['fields']
            prior = cache_id(context['cache'])
            def requests(order):
                return [{'input_ids': torch.tensor([suffixes[i]]), 'mode': 'question',
                    'cache': clone_cache(context['cache'], config), 'collect_hidden': True, 'collect_units': True,
                    'account_attention': True, 'save_source_details': True,
                    'question_positions_local': list(range(len(suffixes[i])-1)),
                    'source_positions': list(range(3, 11)), 'value_permutation_seed': 274800,
                    'context_prefix_length': len(prefix)} for i in order]
            forward = engine.forward(requests(range(4)))
            reverse = engine.forward(requests(reversed(range(4))))[::-1]
            for index, (a, b) in enumerate(zip(forward, reverse)):
                assert set(a['fields']) == set(b['fields'])
                assert all(np.array_equal(v, b['fields'][name]) for name, v in a['fields'].items())
                checks.append({'model_type': kind, 'test': 'shared_prefix_branch_order_every_field', 'question': index, 'passed': True})
            assert cache_id(context['cache']) == prior
            assert source['full_prefix_hidden_BF16'].shape == (26, len(prefix), 32)
            assert source['source_H12_BF16'].shape == (len(prefix), 32)
            early = {'input_ids': torch.tensor([prefix]), 'mode': 'context', 'context_positions_local': list(range(3, 11))}
            early_source = engine.forward([early], stop_after=13)[0]
            assert early_source['postnorm'] is None and 'postnorm_BF16' not in early_source['fields']
            assert np.array_equal(early_source['fields']['source_H12_BF16'], source['source_H12_BF16'])
            checks.append({'model_type': kind, 'test': 'source_cache_immutable_and_early_stop_no_target',
                           'attention_replays': engine.attention_replay_checks,
                           'native_product_checked_scalars': engine.product_checked_scalars, 'passed': True})
        finally:
            engine.close()
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'engine': snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'all_passed': True, 'checks': checks, 'seconds': time.monotonic()-start,
        'scope': 'Synthetic random-weight BF16 CPU fixtures; numerical implementation checks only, no pretrained language evidence or GPU qualification.'}
    save(BASE/'phase2748/unit'/('native_engine_'+str(time.time_ns())+'.json'), value)
    save(BASE/'phase2748/unit/native_engine_current.json', value)
    print('QUESTION_NATIVE_UNIT_PASSED', len(checks), flush=True)


if __name__ == '__main__':
    main()
