"""Causal complete-answer training primitives; no run starts on import."""
from rdc_question_common import *
from phase2747_rdc_training import prepare_gradients


def teacher_states(model, row, teacher_ids):
    """Joint causal B1 teacher prefix, no cache, never supervise prompt tokens.

    This is the training arithmetic shape, NOT the segmented B1 sequential
    shape used for native behavioral evaluation. Same mathematical causal
    dependency does not imply BF16 bit equality between execution shapes.
    """
    import torch
    prompt = row['tokens']['input_ids']
    assert prompt and teacher_ids
    device = next(model.parameters()).device
    ids = torch.tensor([prompt+list(teacher_ids[:-1])], device=device)
    result = model.model(input_ids=ids, use_cache=False).last_hidden_state
    states = result[0, len(prompt)-1:]
    assert states.shape[0] == len(teacher_ids)
    return states


def teacher_objective(model, row, draw, classes, return_packet=False):
    import torch
    target_ids = draw['teacher_ids_including_EOS']
    states = teacher_states(model, row, target_ids)
    losses, records = [], []
    surface = draw['condition'] == 'surface_class_mass_on_true_teacher_history'
    for t, target in enumerate(target_ids):
        # Original head B1 at each teacher position; never change head arithmetic
        # to a multi-position matmul to save time, never build prompt x vocab.
        lp = model.lm_head(states[t:t+1]).float()[0].double().log_softmax(-1)
        loss = -torch.logsumexp(lp[classes == int(classes[target])], dim=0) if surface else -lp[target]
        losses.append(loss)
        if return_packet:
            records.append([float(-lp[target].detach()), int(lp.detach().argmax()),
                float(-(lp.detach().exp()*lp.detach()).sum()), float(loss.detach())])
    objective = torch.stack(losses).mean()
    if return_packet:
        return objective, {'postnorm_BF16': bits(states), 'teacher_ids': np.asarray(target_ids, np.int64),
            'statistics': np.asarray(records, np.float64)}
    return objective


def mean_gradient(model, rows_by_id, draws, classes, params):
    import torch
    gradients = [torch.zeros_like(p) for p in params]
    losses = []
    for draw in draws:
        loss = teacher_objective(model, rows_by_id[draw['question_id']], draw, classes)
        derivative = torch.autograd.grad(loss, params)
        losses.append(float(loss.detach()))
        for accumulator, grad in zip(gradients, derivative):
            accumulator.add_(grad, alpha=1/len(draws))
        del derivative, loss
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    return gradients, losses


def full_norm(values):
    import torch
    return float(torch.stack([value.detach().double().square().sum() for value in values]).sum().sqrt())


def restore(target, original):
    import torch
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(original[name].to(parameter.device))
            assert torch.equal(parameter.detach().cpu(), original[name])


def parameter_fingerprints(model, exclude_target=False):
    """All parameter words, no index sampling or selected-coordinate hash."""
    result = {}
    for name, parameter in model.named_parameters():
        if exclude_target and name.startswith('model.layers.16.mlp.'):
            continue
        array = parameter.detach().cpu().contiguous()
        raw = array.view(__import__('torch').uint8).numpy()
        result[name] = {'shape': list(parameter.shape), 'dtype': str(parameter.dtype),
            'sha256': hashlib.sha256(memoryview(raw).cast('B')).hexdigest()}
    return result
