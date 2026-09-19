"""Synthetic CPU checks of causal teacher alignment and full gradient averaging."""
from rdc_question_common import *
from rdc_question_learning import teacher_states, teacher_objective, mean_gradient, prepare_gradients


def main():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(2748004)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=18, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    rows = {'a': {'tokens': {'input_ids': [1, 11, 12, 13]}},
            'b': {'tokens': {'input_ids': [1, 24, 25, 26, 27]}}}
    target, original, handles = prepare_gradients(model)
    params = list(target.parameters())
    classes = torch.arange(97) % 10
    checks = []
    with torch.no_grad():
        first = teacher_states(model, rows['a'], [31, 32, 33, 2])
        changed = teacher_states(model, rows['a'], [31, 72, 73, 2])
        assert torch.equal(first[:2], changed[:2])
        assert not torch.equal(first[2:], changed[2:])
        # The first output consumes the final prompt token, not first teacher.
        native = model.model(input_ids=torch.tensor([[1,11,12,13,31,32,33]]), use_cache=False).last_hidden_state[0,3:]
        assert torch.equal(first, native)
    checks.append({'case': 'causal_future_teacher_substitution_and_exact_position_alignment', 'passed': True})
    for condition in ['true_complete_answer', 'within_context_permuted_complete_answer', 'surface_class_mass_on_true_teacher_history']:
        draws = [{'question_id': 'a', 'condition': condition, 'teacher_ids_including_EOS': [31,32,33,2]},
                 {'question_id': 'b', 'condition': condition, 'teacher_ids_including_EOS': [51,2]}]
        mean = torch.stack([teacher_objective(model, rows[d['question_id']], d, classes) for d in draws]).mean()
        reference = torch.autograd.grad(mean, params)
        actual, losses = mean_gradient(model, rows, draws, classes, params)
        error = max(float((a-b).abs().max()) for a,b in zip(actual,reference))
        assert error < 1e-6 and abs(float(mean.detach())-np.mean(losses)) < 1e-12
        with torch.no_grad():
            loss, packet = teacher_objective(model, rows['a'], draws[0], classes, True)
            assert abs(float(loss)-packet['statistics'][:,3].mean()) < 1e-12
        checks.append({'case': 'whole_answer_mean_then_example_mean_gradient', 'condition': condition,
            'all_parameter_maximum_gradient_error': error})
    for handle in handles:
        handle.remove()
    result = {'timestamp': stamp(), 'all_passed': True, 'checks': checks,
        'source': snapshot(__file__), 'learning': snapshot(Path(__file__).with_name('rdc_question_learning.py')),
        'bridge': snapshot(Path(__file__).with_name('phase2747_rdc_training.py')),
        'scope': 'CPU synthetic causal and gradient algebra, not an original model fit or language evidence.'}
    immutable(OUT/'unit'/('learning_'+str(time.time_ns())+'.json'), result)
    save(OUT/'unit/learning_current.json', result)
    print('NATURAL_LEARNING_UNIT_PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
