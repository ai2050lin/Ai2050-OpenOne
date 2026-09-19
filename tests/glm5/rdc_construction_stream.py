"""Exact-shape layer scheduling for independent cached native query branches.

This runs the original decoder classes and weights, not a surrogate.  Prefix and
query microbatches have exactly their ordinary B1/B<=16 shapes.  A layer's prefix
KV is needed only by that same layer's queries and can then be released.  Disk
offloaded layers are temporarily instantiated from the original checkpoint once
per source, avoiding a separate disk load for every query microbatch.
"""
import sys
from collections import defaultdict
from contextlib import contextmanager
from rdc_construction_common import *


class LayerCache:
    def __init__(self, layer_index, keys=None, values=None):
        self.layer_index, self.keys, self.values = layer_index, keys, values

    def update(self, keys, values, layer_index):
        import torch
        assert layer_index == self.layer_index
        self.keys = keys if self.keys is None else torch.cat([self.keys, keys], dim=-2)
        self.values = values if self.values is None else torch.cat([self.values, values], dim=-2)
        return self.keys, self.values


class Engine:
    def __init__(self, model, key, probes):
        self.model, self.key, self.probes = model, key, probes
        self.depth, self.width = len(model.model.layers), model.config.hidden_size
        self.early, self.mid = self.depth // 3, 2 * self.depth // 3
        self.selected = list(range(self.early + 2)) + [self.mid, self.depth]
        self.groups = []
        lengths = defaultdict(list)
        for q, p in enumerate(probes):
            lengths[len(p['token_ids'])].append(q)
        for length, indices in sorted(lengths.items()):
            for j in range(0, len(indices), 16):
                self.groups.append((length, indices[j:j+16]))
        self.path = ROOT / 'models/hf' / MODELS[key]
        self.weight_map = read(self.path / 'model.safetensors.index.json')['weight_map']
        self.copied_blocks = set()
        assert all(t == 'full_attention' for t in getattr(model.config, 'layer_types', ['full_attention']))

    @contextmanager
    def layer(self, index):
        import torch
        native = self.model.model.layers[index]
        offloaded = any(p.device.type != 'cuda' for p in native.parameters())
        if not offloaded:
            yield native
            return
        from safetensors import safe_open
        prefix = f'model.layers.{index}.'
        names = list(native.state_dict())
        needed = sum(p.numel() * 2 for p in native.parameters())
        torch.cuda.empty_cache()
        available = torch.cuda.mem_get_info()[0]
        assert available > needed + 512 * 1024**2, ('Layer staging headroom', index, needed, available)
        with torch.device('meta'):
            local = native.__class__(self.model.config, index)
        state = {}
        try:
            for name in names:
                full = prefix + name
                with safe_open(str(self.path / self.weight_map[full]), framework='pt', device='cpu', backend='pread') as f:
                    tensor = f.get_tensor(full)
                    assert tensor.dtype == torch.bfloat16
                    state[name] = tensor.to(device='cuda')
                    del tensor
            local.load_state_dict(state, strict=True, assign=True)
            local.eval()
            assert all(p.device.type == 'cuda' and p.dtype == torch.bfloat16 for p in local.parameters())
            self.copied_blocks.add(index)
            yield local
        finally:
            state.clear()
            del local
            torch.cuda.empty_cache()

    def mask(self, hidden, seen):
        import torch
        from transformers.cache_utils import DynamicCache
        module = sys.modules[self.model.model.__class__.__module__]
        pos = torch.arange(hidden.shape[1], device=hidden.device)[None] + seen
        # Native mask implementation and native positional embedding; no hand-made
        # RoPE inversion or attention mask is substituted.
        cache = None
        if seen:
            cache = DynamicCache(config=self.model.config)
            dummy = torch.empty((1, self.model.config.num_key_value_heads, seen,
                                 self.model.model.layers[0].self_attn.head_dim), device=hidden.device, dtype=hidden.dtype)
            cache.update(dummy, dummy, 0)
        mask = module.create_causal_mask(config=self.model.config, inputs_embeds=hidden,
            attention_mask=None, past_key_values=cache, position_ids=pos)
        rotary = self.model.model.rotary_emb(hidden, pos)
        return pos, mask, rotary

    @staticmethod
    def hooks(layer, data):
        def put(key, value):
            data[key] = value.detach()
        att = layer.self_attn
        handles = [att.q_proj.register_forward_pre_hook(lambda m, a: put('q_input', a[0])),
                   att.q_proj.register_forward_hook(lambda m, a, o: put('q_projected', o)),
                   att.v_proj.register_forward_hook(lambda m, a, o: put('v_before', o)),
                   att.register_forward_hook(lambda m, a, o: (put('attention', o[1]), put('attention_output', o[0])) and None)]
        qmod = att.q_norm if hasattr(att, 'q_norm') else att.q_proj
        kmod = att.k_norm if hasattr(att, 'k_norm') else att.k_proj
        handles.append(qmod.register_forward_hook(lambda m, a, o: put('q_before', o)))
        handles.append(kmod.register_forward_hook(lambda m, a, o: put('k_before', o)))
        return handles

    def run(self, prefix_ids=None, standalone=False, prototypes=None, feature_fn=None):
        import torch
        n, count = 0 if prefix_ids is None else len(prefix_ids), len(self.probes)
        states = np.empty((self.depth + 1, count, self.width), np.uint16)
        prefix_states = np.empty((self.depth + 1, self.width), np.uint16) if n else None
        packets, detail, attchecks = [], {}, []
        current = None
        if n:
            current = self.model.model.embed_tokens(torch.tensor([prefix_ids], device='cuda'))
            ppos, pmask, protary = self.mask(current, 0)
            prefix_states[0] = bits(current[0, -1])
        for length, ix in self.groups:
            h = self.model.model.embed_tokens(torch.tensor([self.probes[q]['token_ids'] for q in ix], device='cuda'))
            pos, mask, rotary = self.mask(h, n)
            packets.append({'h': h, 'indices': ix, 'length': length, 'pos': pos, 'mask': mask, 'rotary': rotary})
            states[0, ix] = bits(h[:, -1])
        candidates = None
        for index in range(self.depth):
            with self.layer(index) as layer:
                observing = index in [0, self.early]
                data, handles = {}, []
                if observing:
                    handles = self.hooks(layer, data)
                try:
                    prefix_cache = None
                    if n:
                        prefix_cache = LayerCache(index)
                        data.clear()
                        if index == self.early:
                            detail['prefix_early_all_positions'] = bits(current[0])
                        current = layer(current, attention_mask=pmask, position_ids=ppos,
                            position_embeddings=protary, past_key_values=prefix_cache, use_cache=True)
                        prefix_states[index+1] = bits(current[0, -1])
                        if observing:
                            detail[f'L{index}_prefix_keys'] = bits(prefix_cache.keys[0])
                            detail[f'L{index}_prefix_values'] = bits(prefix_cache.values[0])
                    for packet in packets:
                        h, ix, length = packet['h'], packet['indices'], packet['length']
                        data.clear()
                        if standalone and index == self.early:
                            for j, q in enumerate(ix):
                                detail[f'p{q}_Hearly'] = bits(h[j])
                        cache = LayerCache(index,
                            prefix_cache.keys.repeat_interleave(len(ix), 0) if n else None,
                            prefix_cache.values.repeat_interleave(len(ix), 0) if n else None)
                        result = layer(h, attention_mask=packet['mask'], position_ids=packet['pos'],
                            position_embeddings=packet['rotary'], past_key_values=cache, use_cache=True)
                        states[index+1, ix] = bits(result[:, -1])
                        packet['h'] = result
                        if observing:
                            att = layer.self_attn
                            shape = (len(ix), length, -1, att.head_dim)
                            qb = data['q_before'].reshape(shape).transpose(1, 2)
                            apply_rope = sys.modules[att.__class__.__module__].apply_rotary_pos_emb
                            qr, _ = apply_rope(qb, torch.zeros_like(qb), *packet['rotary'])
                            allkeys = cache.keys.repeat_interleave(att.num_key_value_groups, 1)
                            # Match the native Q-length GEMM shape: BF16 kernels
                            # can round differently if only the last row is run.
                            scores = (qr @ allkeys.transpose(-2, -1)) * att.scaling
                            if packet['mask'] is not None:
                                scores = scores + packet['mask']
                            manual = scores.float().softmax(-1).to(torch.bfloat16)[:, :, -1]
                            actual = data['attention'][:, :, -1]
                            err = float((manual.float() - actual.float()).abs().max())
                            assert torch.equal(manual, actual), ('Native attention reconstruction', index, ix, err)
                            attchecks.append({'block': index, 'queries': ix, 'max_abs_error': err, 'bit_equal': bool(torch.equal(manual, actual))})
                            for j, q in enumerate(ix):
                                head = f'p{q}_L{index}_'
                                detail[head+'q_before_rope'] = bits(qb[j, :, -1])
                                detail[head+'q_projected'] = bits(data['q_projected'][j, -1])
                                detail[head+'q_input'] = bits(data['q_input'][j, -1])
                                detail[head+'attention'] = bits(actual[j])
                                detail[head+'attention_output'] = bits(data['attention_output'][j, -1])
                                detail[head+'query_keys_after_rope'] = bits(cache.keys[j, :, n:])
                                detail[head+'query_values'] = bits(cache.values[j, :, n:])
                                if standalone and index == self.early:
                                    detail[f'p{q}_q_before_rope_all'] = bits(qb[j])
                                    detail[f'p{q}_k_before_rope_all'] = bits(data['k_before'].reshape(shape).transpose(1, 2)[j])
                                    detail[f'p{q}_values_all'] = bits(data['v_before'].reshape(shape).transpose(1, 2)[j])
                            del allkeys, scores, manual, actual, qb, qr
                        del cache, h, result
                    if index == self.early and feature_fn is not None:
                        assert n and prototypes is not None
                        candidates = feature_fn(self.model, layer, prototypes, self.probes,
                                                prefix_cache.keys[0], prefix_cache.values[0])
                    del prefix_cache
                finally:
                    data.clear()
                    for handle in handles:
                        handle.remove()
            # The caller's `as layer` reference otherwise keeps the just-staged
            # block alive while the next context manager checks its headroom.
            del layer
        post = np.empty((count, self.width), np.uint16)
        for packet in packets:
            post[packet['indices']] = bits(self.model.model.norm(packet['h'])[:, -1])
        prefix_post = bits(self.model.model.norm(current)[0, -1]) if n else None
        return {'query_all_states': states, 'prefix_layers': prefix_states, 'postnorm': post,
                'prefix_postnorm': prefix_post, 'detail': detail, 'attention_checks': attchecks,
                'candidate_early_output': candidates}


def original_reference(model, prefix_ids, probes, key):
    """Unchanged model.model calls, retaining complete last-position boundaries."""
    import torch
    engine = Engine(model, key, probes)
    states = np.empty((engine.depth+1, len(probes), engine.width), np.uint16)
    post = np.empty((len(probes), engine.width), np.uint16)
    observed, handles = {}, []
    for index, layer in enumerate(model.model.layers, 1):
        handles.append(layer.register_forward_hook(lambda m, a, o, i=index: observed.__setitem__(i, bits(o[:, -1]))))
    handles.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: observed.__setitem__(0, bits(o[:, -1]))))
    cache, prefix, prefix_post = None, None, None
    try:
        if prefix_ids is not None:
            o = model.model(input_ids=torch.tensor([prefix_ids], device='cuda'), use_cache=True)
            cache = o.past_key_values
            prefix = np.stack([observed[i][0] for i in range(engine.depth+1)])
            prefix_post = bits(o.last_hidden_state[0, -1])
            del o
        for _, ix in engine.groups:
            own = clone_cache(cache, model.config, len(ix)) if cache is not None else None
            o = model.model(input_ids=torch.tensor([probes[q]['token_ids'] for q in ix], device='cuda'),
                            past_key_values=own, use_cache=own is not None)
            post[ix] = bits(o.last_hidden_state[:, -1])
            for i in range(engine.depth+1):
                states[i, ix] = observed[i]
            del own, o
    finally:
        for h in handles:
            h.remove()
    return {'query_all_states': states, 'postnorm': post, 'prefix_layers': prefix, 'prefix_postnorm': prefix_post}


def compare(reference, candidate):
    checks = {}
    for key in ['query_all_states', 'postnorm', 'prefix_layers', 'prefix_postnorm']:
        if reference[key] is None:
            assert candidate[key] is None
            continue
        a, b = reference[key], candidate[key]
        checks[key] = {'bit_equal': bool(np.array_equal(a, b)), 'different_scalars': int(np.count_nonzero(a != b)),
                       'max_abs_error': float(np.max(abs(unbits(a).astype(float) - unbits(b).astype(float))))}
    checks['all_bit_equal'] = all(v['bit_equal'] for v in checks.values())
    return checks
