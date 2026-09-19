"""Passive full-coordinate observation under native independent-B1 scheduling.

The numerical operator is the installed original decoder. No hidden-coordinate
selection, transplanted difference or approximation to the model is performed.
"""
import sys
from rdc_construction_common import *
from rdc_formation_microbatch import LayerWeights


COORDINATES = ['attention_input', 'attention_write', 'pre_MLP', 'MLP_input', 'MLP_write']
UNITS = ['gate', 'up', 'product']


class QuestionNative:
    def __init__(self, model, selected_blocks, device='cuda'):
        self.model, self.core, self.config, self.device = model, model.model, model.config, device
        assert self.config.model_type in {'qwen3', 'glm'}
        assert not getattr(self.core, 'has_sliding_layers', False)
        self.depth = len(self.core.layers)
        self.module = sys.modules[self.core.__class__.__module__]
        self.selected_blocks = list(selected_blocks)
        self.weights = LayerWeights(device)
        self.current = None
        self.replaying = False
        self.handles = []
        self.product_checked_scalars = 0
        self.attention_replay_checks = 0
        self.layer_calls = 0
        self._hooks()

    def _put(self, block, name, value):
        if self.current is None or self.replaying:
            return
        r, state = self.current
        if r.get('collect_units') and block in self.selected_blocks:
            state['fields'][f'block{block}_{name}_BF16'] = bits(value[0, -1])

    def _hooks(self):
        for block, layer in enumerate(self.core.layers):
            if block in self.selected_blocks:
                self.handles.append(layer.input_layernorm.register_forward_hook(
                    lambda m, a, o, i=block: self._put(i, 'attention_input', o)))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda m, a, i=block: self._put(i, 'pre_MLP', a[0])))
                self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                    lambda m, a, o, i=block: self._put(i, 'MLP_input', o)))
                self.handles.append(layer.mlp.register_forward_hook(
                    lambda m, a, o, i=block: self._put(i, 'MLP_write', o)))
                mlp = layer.mlp
                def gate_up(name, block, value):
                    if self.current is None or self.replaying or not self.current[0].get('collect_units'):
                        return
                    self.current[1]['pending'][(block, name)] = value[0, -1].detach().clone()
                    self._put(block, name, value)
                if hasattr(mlp, 'gate_up_proj'):
                    def combined(m, a, o, i=block):
                        gate, up = o.chunk(2, dim=-1)
                        gate_up('gate', i, gate)
                        gate_up('up', i, up)
                    self.handles.append(mlp.gate_up_proj.register_forward_hook(combined))
                    activation = mlp.activation_fn
                else:
                    self.handles.append(mlp.gate_proj.register_forward_hook(
                        lambda m, a, o, i=block: gate_up('gate', i, o)))
                    self.handles.append(mlp.up_proj.register_forward_hook(
                        lambda m, a, o, i=block: gate_up('up', i, o)))
                    activation = mlp.act_fn
                def product(m, a, i=block, activation=activation):
                    if self.current is None or self.replaying or not self.current[0].get('collect_units'):
                        return
                    import torch
                    pending = self.current[1]['pending']
                    expected = activation(pending.pop((i, 'gate')))*pending.pop((i, 'up'))
                    assert torch.equal(expected, a[0][0, -1]), ('Original MLP product identity', i)
                    self.product_checked_scalars += expected.numel()
                    self._put(i, 'product', a[0])
                self.handles.append(mlp.down_proj.register_forward_pre_hook(product))
            if block in self.selected_blocks or block == 12:
                def attention(m, a, o, i=block):
                    if self.current is None or self.replaying:
                        return
                    self._put(i, 'attention_write', o[0])
                    r, state = self.current
                    if i != 12 or r.get('mode') == 'context':
                        return
                    state['fields']['native_source_read_BF16'] = bits(o[0][0, -1])
                    if r.get('account_attention'):
                        self._attention_accounting(m, o)
                self.handles.append(layer.self_attn.register_forward_hook(attention))
            if block == 12:
                def before_o(m, a):
                    if self.current is not None and not self.replaying and self.current[0].get('account_attention'):
                        self.current[1]['pending']['native_pre_o'] = a[0].detach()
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(before_o))
                def query(m, a, o):
                    if self.current is not None and not self.replaying and self.current[0].get('account_attention'):
                        value = o[0, -1].reshape(self.config.num_attention_heads, self.config.head_dim)
                        self.current[1]['fields']['block12_Q_before_RoPE_BF16'] = bits(value)
                att = layer.self_attn
                self.handles.append((att.q_norm if hasattr(att, 'q_norm') else att.q_proj).register_forward_hook(query))

    def _attention_accounting(self, attention, output):
        import torch
        r, state = self.current
        probability = output[1]
        assert probability is not None and probability.shape[0] == 1
        cache = state['cache'].layers[12]
        keys, values = cache.keys, cache.values
        repeat = self.module.repeat_kv(values, attention.num_key_value_groups)
        head_read = torch.matmul(probability, repeat).transpose(1, 2).contiguous()
        head_read = head_read.reshape(1, r['input_ids'].shape[1], -1).contiguous()
        native = state['pending'].pop('native_pre_o')
        assert torch.equal(head_read, native), 'Same-shape complete source contraction changed native pre-O'
        self.replaying = True
        try:
            reconstructed = attention.o_proj(head_read)
            assert torch.equal(reconstructed, output[0]), 'Original same-shape O contraction mismatch'
            order = torch.arange(values.shape[-2], device=values.device)
            positions = torch.tensor(r['source_positions'], device=values.device, dtype=torch.long)
            rng = np.random.default_rng(r['value_permutation_seed'])
            permuted = rng.permutation(r['source_positions'])
            order[positions] = torch.tensor(permuted, device=values.device)
            shuffled_values = values.index_select(-2, order)
            shuffled = torch.matmul(probability, self.module.repeat_kv(shuffled_values, attention.num_key_value_groups))
            shuffled = shuffled.transpose(1, 2).contiguous().reshape_as(head_read)
            shuffled_write = attention.o_proj(shuffled)
        finally:
            self.replaying = False
        fields = state['fields']
        fields['source_value_pair_shuffle_BF16'] = bits(shuffled_write[0, -1])
        if r.get('save_source_details'):
            fields['block12_attention_BF16'] = bits(probability[0, :, -1])
            fields['block12_pre_O_BF16'] = bits(native[0, -1])
            begin = r.get('context_prefix_length', 0)
            fields['block12_appended_keys_BF16'] = bits(keys[0, :, begin:])
            fields['block12_appended_values_BF16'] = bits(values[0, :, begin:])
            fields['source_value_permutation'] = order.cpu().numpy()
        self.attention_replay_checks += 1

    def _hidden(self, r, state, index):
        h, fields = state['h'], state['fields']
        if r.get('collect_hidden'):
            state['hidden'].append(bits(h[0, -1]))
        if r.get('full_prefix_H'):
            state['fullH'].append(bits(h[0]))
        if index == 12:
            fields['H12_last_BF16'] = bits(h[0, -1])
            if r.get('mode') == 'context':
                fields['source_H12_BF16'] = bits(h[0])
        if index in (0, 12):
            for role in ['context', 'question']:
                positions = r.get(role+'_positions_local')
                if positions:
                    # FP64 means preserve all original coordinates; the raw
                    # embedding is addressable from immutable token IDs/weights.
                    fields[f'{role}_H{index}_mean'] = h[0, positions].double().mean(0).cpu().numpy()

    def forward(self, requests, stop_after=None):
        import torch
        from transformers.cache_utils import DynamicCache
        stop = self.depth if stop_after is None else stop_after
        assert 13 <= stop <= self.depth
        states = []
        with torch.inference_mode():
            for r in requests:
                assert r['input_ids'].ndim == 2 and r['input_ids'].shape[0] == 1
                h = self.core.embed_tokens(r['input_ids'])
                cache = r.get('cache')
                if r.get('use_cache', True) and cache is None:
                    cache = DynamicCache(config=self.config)
                positions = r.get('position_ids')
                if positions is None:
                    seen = cache.get_seq_length() if cache is not None else 0
                    positions = (torch.arange(h.shape[1], device=h.device)+seen).unsqueeze(0)
                mask = self.module.create_causal_mask(config=self.config, inputs_embeds=h,
                    attention_mask=r.get('attention_mask'), past_key_values=cache, position_ids=positions)
                state = {'h': h, 'cache': cache, 'positions': positions, 'mask': mask,
                         'rotary': self.core.rotary_emb(h, position_ids=positions),
                         'fields': {}, 'pending': {}, 'hidden': [], 'fullH': []}
                self._hidden(r, state, 0)
                states.append(state)
            for li, layer in enumerate(self.core.layers[:stop]):
                self.weights.begin()
                try:
                    for r, state in zip(requests, states):
                        self.current = (r, state)
                        state['h'] = layer(state['h'], attention_mask=state['mask'],
                            position_embeddings=state['rotary'], position_ids=state['positions'],
                            past_key_values=state['cache'], use_cache=r.get('use_cache', True))
                        self.current = None
                        self._hidden(r, state, li+1)
                        self.layer_calls += 1
                finally:
                    self.current = None
                    self.weights.end()
            results = []
            for r, state in zip(requests, states):
                assert not state['pending']
                fields = state['fields']
                if state['hidden']:
                    fields['hidden_BF16'] = np.stack(state['hidden'])
                if state['fullH']:
                    fields['full_prefix_hidden_BF16'] = np.stack(state['fullH'])
                if r.get('mode') == 'context':
                    fields['block12_prefix_keys_BF16'] = bits(state['cache'].layers[12].keys[0])
                    fields['block12_prefix_values_BF16'] = bits(state['cache'].layers[12].values[0])
                postnorm = self.core.norm(state['h'])[:, -1] if stop == self.depth else None
                if postnorm is not None:
                    fields['postnorm_BF16'] = bits(postnorm[0])
                assert all(np.isfinite(unbits(v) if v.dtype == np.uint16 else v).all() for v in fields.values())
                r['cache'] = state['cache']
                results.append({'fields': fields, 'postnorm': postnorm})
        return results

    def close(self):
        for hook in self.handles:
            hook.remove()
        self.handles.clear()
        self.current = None
        self.weights.close()


def native_whole_B1(model, ids, weights=None):
    """Read-only installed-model reference, including actual native H boundaries."""
    import torch
    arrays = []
    handles = [model.model.embed_tokens.register_forward_hook(lambda m, a, o: arrays.append(bits(o[0, -1])))]
    handles += [layer.register_forward_hook(lambda m, a, o: arrays.append(bits(o[0, -1])))
                for layer in model.model.layers]
    if weights is not None:
        # Original installed whole-model forward; only immutable checkpoint I/O
        # uses the already audited tensor-slice reader for offloaded layers.
        handles += [layer.register_forward_pre_hook(lambda m, a: weights.begin()) for layer in model.model.layers]
        handles += [layer.register_forward_hook(lambda m, a, o: weights.end()) for layer in model.model.layers]
    try:
        with torch.inference_mode():
            value = model.model(input_ids=ids, use_cache=True)
        result = {'hidden_BF16': np.stack(arrays), 'postnorm_BF16': bits(value.last_hidden_state[0, -1])}
    finally:
        for handle in handles:
            handle.remove()
        if weights is not None:
            weights.end()
    return result
