"""Check discovery cases and the quarantined first holdout replay failure."""
import json

import torch
from transformers import AutoModelForCausalLM

from mlp_b34_confirm_v2 import configure
import mlp_b34_confirm_design as D
from shared.dist_protocol.mechanism_trace import GpuLease


def main():
    configure()
    from mlp_b34_confirm_run import NativeAssay

    p = D.checked_plan()
    items = D.READ(D.OUT/'discovery-materials.json')['items'][:4]
    items.append(D.READ(D.OUT/'holdout-materials.json')['items'][0])
    with GpuLease(), torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(D.MODEL, local_files_only=True, dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa').eval()
        assay = NativeAssay(model, p['choice_ids'])
        for item in items:
            original = []
            handle = model.model.layers[35].register_forward_hook(lambda module, args, out: original.append(out))
            logits, base = assay.observe(item)
            handle.remove()
            hidden = base['H34']
            pos = torch.arange(hidden.shape[1], device=hidden.device)[None]
            mask = assay.make_mask(config=model.config, inputs_embeds=hidden, attention_mask=None,
                                   past_key_values=None, position_ids=pos)
            rotary = assay.decoder.rotary_emb(hidden, pos)
            state, short = assay.suffix(hidden, mask, pos, rotary)
            full = model.lm_head(assay.decoder.norm(state)[:, -1:])
            error = lambda value: float((value.float()-logits.float()).abs().max())
            print(json.dumps({'id': item['id'], 'native_a35_equal': torch.equal(assay.data['A35'][:, -1], base['A35'][:, -1]),
                              'native_h35_equal': torch.equal(state, original[0]),
                              'native_r35_equal': torch.equal(assay.data['R35'], base['R35']),
                              'short_norm_max_error': error(short), 'full_norm_max_error': error(full),
                              'full_norm_bitwise': torch.equal(full, logits)}), flush=True)


if __name__ == '__main__':
    main()
