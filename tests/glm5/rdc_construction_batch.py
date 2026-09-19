"""Source-interleaved exact-shape scheduling, without stacking source batches.

Each source still has a B1 prefix and its own exact-length B<=16 query calls.
Only scheduling changes: keep one original block resident while visiting a
small number of independent sources.  Native tensors never cross source IDs.
"""
import sys
from rdc_construction_stream import Engine, LayerCache
from rdc_construction_common import *


class BatchedEngine(Engine):
    def prepare(self, prefix_ids):
        import torch
        n = 0 if prefix_ids is None else len(prefix_ids)
        s = {'n': n, 'states': np.empty((self.depth+1, len(self.probes), self.width), np.uint16),
             'prefix_states': np.empty((self.depth+1, self.width), np.uint16) if n else None,
             'packets': [], 'detail': {}, 'checks': [], 'current': None, 'candidates': None}
        if n:
            s['current'] = self.model.model.embed_tokens(torch.tensor([prefix_ids], device='cuda'))
            s['ppos'], s['pmask'], s['protary'] = self.mask(s['current'], 0)
            s['prefix_states'][0] = bits(s['current'][0, -1])
        for length, ix in self.groups:
            h = self.model.model.embed_tokens(torch.tensor([self.probes[q]['token_ids'] for q in ix], device='cuda'))
            pos, mask, rotary = self.mask(h, n)
            s['packets'].append({'h': h, 'indices': ix, 'length': length, 'pos': pos, 'mask': mask, 'rotary': rotary})
            s['states'][0, ix] = bits(h[:, -1])
        return s

    def advance(self, s, index, layer, standalone, prototypes, feature_fn):
        import torch
        n, detail, data = s['n'], s['detail'], {}
        observing = index in [0, self.early]
        handles = self.hooks(layer, data) if observing else []
        prefix_cache = None
        try:
            if n:
                prefix_cache = LayerCache(index)
                if index == self.early:
                    detail['prefix_early_all_positions'] = bits(s['current'][0])
                s['current'] = layer(s['current'], attention_mask=s['pmask'], position_ids=s['ppos'],
                    position_embeddings=s['protary'], past_key_values=prefix_cache, use_cache=True)
                s['prefix_states'][index+1] = bits(s['current'][0, -1])
                if observing:
                    detail[f'L{index}_prefix_keys'] = bits(prefix_cache.keys[0])
                    detail[f'L{index}_prefix_values'] = bits(prefix_cache.values[0])
            for packet in s['packets']:
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
                s['states'][index+1, ix] = bits(result[:, -1])
                packet['h'] = result
                if observing:
                    att = layer.self_attn
                    shape = (len(ix), length, -1, att.head_dim)
                    qb = data['q_before'].reshape(shape).transpose(1, 2)
                    apply_rope = sys.modules[att.__class__.__module__].apply_rotary_pos_emb
                    qr, _ = apply_rope(qb, torch.zeros_like(qb), *packet['rotary'])
                    allkeys = cache.keys.repeat_interleave(att.num_key_value_groups, 1)
                    # Diagnostic only; preserve native full query GEMM shape.
                    scores = (qr @ allkeys.transpose(-2, -1)) * att.scaling
                    if packet['mask'] is not None:
                        scores = scores + packet['mask']
                    manual = scores.float().softmax(-1).to(torch.bfloat16)[:, :, -1]
                    actual = data['attention'][:, :, -1]
                    err = float((manual.float()-actual.float()).abs().max())
                    assert torch.equal(manual, actual), ('Native attention reconstruction', index, ix, err)
                    s['checks'].append({'block': index, 'queries': ix, 'max_abs_error': err,
                                        'bit_equal': bool(torch.equal(manual, actual))})
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
                s['candidates'] = feature_fn(self.model, layer, prototypes, self.probes,
                                             prefix_cache.keys[0], prefix_cache.values[0])
        finally:
            data.clear()
            for handle in handles:
                handle.remove()

    def finish(self, s):
        post = np.empty((len(self.probes), self.width), np.uint16)
        for packet in s['packets']:
            post[packet['indices']] = bits(self.model.model.norm(packet['h'])[:, -1])
        prefix_post = bits(self.model.model.norm(s['current'])[0, -1]) if s['n'] else None
        return {'query_all_states': s['states'], 'prefix_layers': s['prefix_states'], 'postnorm': post,
            'prefix_postnorm': prefix_post, 'detail': s['detail'], 'attention_checks': s['checks'],
            'candidate_early_output': s['candidates']}

    def run_many(self, prefix_ids, standalone=False, prototypes=None, feature_fn=None):
        assert 1 <= len(prefix_ids) <= 16
        sessions = [self.prepare(ids) for ids in prefix_ids]
        for index in range(self.depth):
            with self.layer(index) as layer:
                for s in sessions:
                    self.advance(s, index, layer, standalone, prototypes, feature_fn)
            del layer
        return [self.finish(s) for s in sessions]

    def run(self, prefix_ids=None, standalone=False, prototypes=None, feature_fn=None):
        return self.run_many([prefix_ids], standalone=standalone, prototypes=prototypes, feature_fn=feature_fn)[0]
