"""Calibrate zero-edit replay, open the frozen behavioral gate, then run intervention."""
from pathlib import Path
from time import monotonic

import numpy as np
import mlp_b31_authority_run as R
import mlp_b31_authority_design as D


def gate(model,plan):
    """Only inspect baseline choices before opening the intervention stage."""
    items=D.items(plan,'holdout')
    records=[]
    started=monotonic()
    R.log('holdout_behavior_gate',0,len(items),started)
    for index,item in enumerate(items,1):
        tokens=R.torch.tensor(item['tokens'],device='cuda')
        logits=model(input_ids=tokens,use_cache=False,logits_to_keep=1).logits
        scores=R.choices(logits,plan['choice_ids'][:2])
        records.append({'id':item['id'],'frame':item['frame'],'scores':scores.tolist(),
                        'both_correct':bool(np.array_equal(scores.argmax(-1),item['correct']))})
        if index%32==0:
            R.log('holdout_behavior_gate',index,len(items),started)
        if monotonic()-started>60*plan['resources']['stage_minutes']:
            raise RuntimeError('Behavior calibration GPU time budget exceeded')
    rate=sum(row['both_correct'] for row in records)/len(records)
    per_frame={name:[r['both_correct'] for r in records if r['frame']==name] for name in {r['frame'] for r in records}}
    result={'provenance':plan['provenance'],'plan_sha256':R.identity(),'selection_sha256':D.HASH(D.OUT/'selection.json'),
            'source_sha256':D.HASH(Path(__file__)),'materials_sha256':plan['inputs']['holdout']['sha256'],
            'pairs':len(records),'both_correct':rate,
            'frame_rates':{k:sum(rows)/len(rows) for k,rows in per_frame.items()},
            'passed':rate>=plan['calibration']['pair_base_accuracy_min'],'rows':records}
    D.SAVE(D.OUT/'holdout-behavior-gate.json',result)
    print({k:v for k,v in result.items() if k!='rows'},flush=True)
    return result


def main():
    torch=R.torch
    plan=D.checked()
    selected=R.selection(plan)
    torch.manual_seed(D.SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with R.GpuLease(),torch.inference_mode():
        started=monotonic()
        R.log('load',0,1,started,requested='confirm')
        model=R.AutoModelForCausalLM.from_pretrained(D.MODEL,local_files_only=True,dtype=torch.bfloat16,
              device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
        R.log('load',1,1,started,requested='confirm')
        R.calibrate(model,plan,selected)
        behavior=gate(model,plan)
        if not behavior['passed']:
            raise ValueError('Unedited holdout behavioral gate failed. Preserve the baseline record and keep intervention closed.')
        R.pairs(model,plan,'holdout',selected)
        R.capability(model,plan,selected)


if __name__=='__main__':
    main()
