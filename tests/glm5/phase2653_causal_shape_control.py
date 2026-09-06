"""Same causal prefix, equal tensor shape: rule out numeric shape artifacts in source fields."""
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr,tensor_output

OUT=RESULT/'phase2653_output_function_scalar_validation'

@torch.inference_mode()
def source_hidden(model,row,length=None):
    ids=row['prompt_ids'];p=[row['entity_spans']['a'][-1],row['entity_spans']['b'][-1]];T=len(ids) if length is None else length
    assert T>=len(ids);mask=[1]*len(ids)+[0]*(T-len(ids));ii=ids+[0]*(T-len(ids))
    em=model.get_input_embeddings()(torch.tensor([ii],device='cpu')).to('cuda:0');values=[arr(em[0,p])];hooks=[]
    def hook(m,inp,out):values.append(arr(tensor_output(out)[0,p]))
    for block in model.model.layers:hooks.append(block.register_forward_hook(hook))
    try:
        kwargs={} if length is None else {'attention_mask':torch.tensor([mask],device='cuda:0')}
        model.model(inputs_embeds=em,use_cache=False,**kwargs)
    finally:
        for h in hooks:h.remove()
    return np.stack(values)

def run_controls(model):
    cases={r['case_index']:r for r in read(RESULT/'phase2648_output_function_contract/material/cases.json')}
    pairs=read(RESULT/'phase2651_output_function_maps/analysis/cross_function_source_traces.json');selected=[]
    for group in sorted({(r['family'],r['language']) for r in pairs}):
        for modes in [('name','cloze'),('truth_a','truth_b')]:
            rr=[r for r in pairs if (r['family'],r['language'])==group and (r['mode_a'],r['mode_b'])==modes]
            worst=max(rr,key=lambda r:max(r['source_hidden_max_error']))
            fixed=next(r for r in rr if all(cases[r['case_a']][k]==0 for k in ('unit','form','target_index','mention_order')))
            for selector,r in [('worst_numeric',worst),('fixed_unit0',fixed)]:selected.append({'selector':selector,'case_a':r['case_a'],'case_b':r['case_b']})
    save(OUT/'protocol/causal_shape_controls.json',{'pairs':selected,'design':'32worst-case and32fixed source-prefix pairs; some may overlap. Diagnostic selected after observedsourceerrors, notsemantic validation. Four original forwards/pair, zero parameter changes; right-padded IDs0 are masked, source positions unchanged.'})
    rows=[]
    for i,s in enumerate(selected):
        a,b=cases[s['case_a']],cases[s['case_b']];T=max(len(a['prompt_ids']),len(b['prompt_ids']))
        ah,bh=source_hidden(model,a),source_hidden(model,b);ap,bp=source_hidden(model,a,T),source_hidden(model,b,T)
        rows.append({**s,'lengths':[len(a['prompt_ids']),len(b['prompt_ids'])],'original_source_max_error':float(np.max(np.abs(ah-bh))),
            'equalshape_source_max_error':float(np.max(np.abs(ap-bp))),'equalshape_source_bitwise':np.array_equal(ap,bp),
            'original_source_relative_l2':float(np.linalg.norm(ah-bh)/max(np.linalg.norm(ah),1e-30)),
            'padding_changed_a_max':float(np.max(np.abs(ah-ap))),'padding_changed_b_max':float(np.max(np.abs(bh-bp)))})
        if (i+1)%16==0:print('source equalshape control',i+1,'/64',flush=True)
    summary={'pair_conditions':len(rows),'forward_calls':len(rows)*4,'unique_pairs':len({(r['case_a'],r['case_b']) for r in rows}),
        'maximum_original_source_error':max(r['original_source_max_error'] for r in rows),'maximum_equalshape_source_error':max(r['equalshape_source_max_error'] for r in rows),
        'equalshape_bitwise_pair_count':sum(r['equalshape_source_bitwise'] for r in rows),'maximum_original_relative_l2':max(r['original_source_relative_l2'] for r in rows)}
    result={'summary':summary,'rows':rows,'boundary':'If equalshape source fields become identical, this supports a numerical tensor-shape explanation for these source discrepancies. It is an implementation calibration, not a linguistic mechanism or proof for every other context.'}
    save(OUT/'analysis/causal_shape_controls.json',result);return result
