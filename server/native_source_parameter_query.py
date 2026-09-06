"""Read-only published native source/head/coordinate/weight inspection. No torch."""
import json
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT

OUT=RESULT/'phase2684_source_campaign_delivery'
SOURCE=RESULT/'phase2679_native_source_ledger'
LAYERS=(23,26,27,28);SITES=((23,6197),(26,3594),(27,3221),(28,5952),(28,8513))


def decode(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a


@lru_cache(maxsize=1)
def metadata():
    try:return json.loads((OUT/'material/published_source_cases.json').read_text(encoding='utf-8'))
    except FileNotFoundError as e:raise HTTPException(404,'Source-coordinate campaign not published yet') from e


def safe_path(relative):
    path=(RESULT/relative).resolve()
    if not path.is_relative_to(RESULT.resolve()) or path.suffix!='.npz':raise HTTPException(400,'Invalid published artifact')
    return path


@lru_cache(maxsize=2)
def source_pack(dataset,case):
    r=next((r for r in metadata() if r['dataset']==dataset and r['case_index']==case),None)
    if r is None:raise HTTPException(404,'Unknown published source case')
    with np.load(safe_path(r['source_path']),allow_pickle=False) as z:data={k:decode(z[k]).astype(np.float64) for k in z.files}
    return data


@lru_cache(maxsize=1)
def weights():
    with np.load(SOURCE/'weights/native_source_weights.npz',allow_pickle=False) as z:return {k:decode(z[k]).astype(np.float64) for k in z.files}


@lru_cache(maxsize=2)
def full_hidden(relative):
    with np.load(safe_path(relative),allow_pickle=False) as z:value=z['full__h']
    value.flags.writeable=False;return value


def options():
    numeric=RESULT/'phase2682_resolved_scalar_paths'
    controls=json.loads((numeric/'protocol/frozen.json').read_text(encoding='utf-8'))['controls'] if (numeric/'analysis/final.json').exists() else []
    return {'phase':2684,'sites':[{'layer':l,'unit':j} for l,j in SITES],'coordinates':2560,'checkpoints':37,'heads':32,'head_coordinates':128,
            'scalar_controls':controls,
            'cases':[{'dataset':r['dataset'],'case':r['case_index'],'label':r['case_id'],'tokens':len(r['token_strings'])} for r in metadata()],
            'boundary':'Source positions hold contextualized K/V, not isolated word meanings. Observed RMS denominator allocations are conditional accounting, not source deletions.'}


@lru_cache(maxsize=1)
def numeric_records():
    path=RESULT/'phase2682_resolved_scalar_paths/analysis/records.json'
    return {r['case_index']:r for r in json.loads(path.read_text(encoding='utf-8'))}


def query(dataset,case,layer,unit,coordinate,checkpoint,source_token,head,head_coordinate,query_position,hidden_token=None):
    row=next((r for r in metadata() if r['dataset']==dataset and r['case_index']==case),None)
    if row is None:raise HTTPException(404,'Unknown published source case')
    if (layer,unit) not in SITES or query_position not in (0,1):raise HTTPException(400,'Unknown unit/query boundary')
    n=len(row['token_strings']);qtoken=(row['body_end_token'],n-1)[query_position];ht=qtoken if hidden_token is None else hidden_token
    if not(0<=coordinate<2560 and 0<=checkpoint<37 and 0<=source_token<n and 0<=head<32 and 0<=head_coordinate<128 and 0<=ht<n):raise HTTPException(400,'Physical index out of range')
    pack=source_pack(dataset,case);d={k.split('__',1)[1]:v for k,v in pack.items() if k.startswith(f'L{layer}__')};w=weights()
    q=query_position;k=coordinate;s=source_token;dh=d['actual_value'].shape[-1];nh=d['actual_probability'].shape[1];nkv=d['actual_value'].shape[1];kv=head//(nh//nkv)
    wo=w[f'L{layer}__Wo'];wv=w[f'L{layer}__J{unit}_gate'];wu=w[f'L{layer}__J{unit}_up'];wd=w[f'L{layer}__J{unit}_down']
    p=d['actual_probability'][q];v=d['actual_value'];qv=d['actual_query_post_rope'][q,head];key=d['actual_key_post_rope'][s,kv]
    projected=np.stack([v[:,h//(nh//nkv)]@wo[k,h*dh:(h+1)*dh] for h in range(nh)])
    terms=p*projected;source_by_token=terms.sum(0);headterm=float(terms[head,s]);pre=d['pre_mlp_norm'][q]
    meta=json.loads((SOURCE/'protocol/native_weights.json').read_text(encoding='utf-8'))['layers'][str(layer)]
    denominator=float(np.sqrt(np.mean(pre*pre)+meta['epsilon']));normscale=float(w[f'L{layer}__gamma'][k]/denominator)
    hidden_bits=full_hidden(row['native_path']);embedding=float(decode(hidden_bits[0,ht,k]));hidden=float(decode(hidden_bits[checkpoint,ht,k]))
    values={'embedding_coordinate':embedding,'hidden_coordinate':hidden,'native_attention_coordinate':float(d['attention_output'][q,k]),
        'Q_post_RoPE_d':float(qv[head_coordinate]),'K_post_RoPE_d':float(key[head_coordinate]),'V_source_d':float(v[s,kv,head_coordinate]),
        'actual_Wo_k_hd':float(wo[k,head*dh+head_coordinate]),'actual_probability':float(p[head,s]),
        'QK_single_dimension_scaled':float(qv[head_coordinate]*key[head_coordinate]*d['scaling']),
        'QK_all_dimensions_scaled_before_mask':float(qv@key*d['scaling']),
        'single_head_source_output_coordinate':headterm,'all_heads_source_output_coordinate':float(source_by_token[s]),
        'all_sources_heads_output_coordinate':float(source_by_token.sum()),'observed_RMS_denominator64':denominator,
        'observed_RMS_coordinate_scale':normscale,'actual_Wgate_jk':float(wv[k]),'actual_Wup_jk':float(wu[k]),'actual_Wdown_kj':float(wd[k]),
        'actual_MLP_input_coordinate':float(d['mlp_x'][q,k]),'native_gate_input_term':float(wv[k]*d['mlp_x'][q,k]),
        'native_up_input_term':float(wu[k]*d['mlp_x'][q,k]),'conditional_head_source_gate_term':float(wv[k]*normscale*headterm),
        'conditional_head_source_up_term':float(wu[k]*normscale*headterm),'actual_gate_unit':float(d['gate'][q,unit]),
        'actual_up_unit':float(d['up'][q,unit]),'actual_product_unit':float(d['mlp_a'][q,unit]),
        'single_unit_down_coordinate':float(wd[k]*d['mlp_a'][q,unit]),'actual_whole_MLP_output_coordinate':float(d['mlp_down'][q,k])}
    assert all(np.isfinite(v) for v in values.values())
    numeric=None
    if dataset=='fresh' and (RESULT/'phase2682_resolved_scalar_paths/analysis/final.json').exists():
        rec=numeric_records().get(case)
        if rec is not None:
            numeric={'scope':'Same learned weights, FP32core teacherforcing ALL observed BF16 generated IDs, not BF16 freegeneration. Local arithmetic prediction and actualcompleteoutput effects separately measured; FP64readout exists only16frozenprefixes.',
                'baseline_logprobs':rec['baseline_logprobs'],'baseline_logprobs64':rec['baseline_logprobs64'],'observed_ids':rec['observed_ids'],
                'observed_text':rec['observed_text'],'noop_exact':rec['noop_exact'],'all12_matrices_restored':rec['all12_matrices_restored'],
                'effects':[c for c in rec['conditions'] if (c['layer'],c['unit'],c['coordinate'])==(layer,unit,k)]}
    return {'phase':2684,'dataset':dataset,'case':case,'case_id':row['case_id'],'layer':layer,'unit':unit,'coordinate':k,'checkpoint':checkpoint,
        'query_position':q,'query_token':qtoken,'query_token_string':row['token_strings'][qtoken],'hidden_token':ht,'hidden_token_string':row['token_strings'][ht],
        'source_token':s,'source_token_string':row['token_strings'][s],'source_role':row['token_regions'][s]['role'],'head':head,'kv_head':kv,'head_coordinate':head_coordinate,
        'values':values,'prompt':row['prompt'],'natural':row['natural'],'numeric_scalar_validation':numeric,
        'source_trace':[{'token':i,'token_string':row['token_strings'][i],'role':row['token_regions'][i]['role'],'all_heads_term':float(source_by_token[i]),'selected_head_term':float(terms[head,i])} for i in range(n)],
        'head_coordinate_trace':{'Q':qv.tolist(),'K':key.tolist(),'V':v[s,kv].tolist(),'Wo':wo[k,head*dh:(head+1)*dh].tolist()},
        'boundary':'Native BF16 values converted exactly; FP64 products are readout/accounting, not FP64model. E/H use hidden_token; source/head andMLP values use the explicitly selected body/taskquery. All-head source sums exclude separately measured AV/Wo/bias rounding terms, so they need not equal native attention output. Source V is contextualized. Observed RMS denominator is endogenous: conditional terms do not predict source ablation. No semantic mechanism closure.'}
