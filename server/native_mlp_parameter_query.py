"""Published native MLP unit/weight paths; readonly, no model import/load."""
import json
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT

OUT=RESULT/'phase2676_native_mlp_delivery'
BF=RESULT/'phase2671_native_mlp_field'
FP=RESULT/'phase2674_native_mlp_scalar'
LAYERS=(23,26,27,28)
SITES=((23,6197),(26,3594),(27,3221),(28,5952),(28,8513))

def decode(a):return (a.astype(np.uint32)<<16).view(np.float32)

@lru_cache(maxsize=1)
def metadata():
    try:return json.loads((OUT/'material/published_mlp_cases.json').read_text(encoding='utf-8'))
    except FileNotFoundError as e:raise HTTPException(status_code=404,detail='Native MLP campaign is not published yet') from e

@lru_cache(maxsize=2)
def native(case):
    if case not in {r['case_index'] for r in metadata()}:raise HTTPException(status_code=404,detail='Unpublished native case')
    with np.load(BF/f'field/case_{case:04d}.npz',allow_pickle=False) as z:a={k:decode(z[k]) for k in z.files}
    for v in a.values():v.flags.writeable=False
    return a

@lru_cache(maxsize=2)
def precise(case):
    if case not in {r['case_index'] for r in metadata()}:raise HTTPException(status_code=404,detail='Unpublished native case')
    with np.load(FP/f'field/case_{case:04d}.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    for v in a.values():v.flags.writeable=False
    return a

@lru_cache(maxsize=1)
def weights():
    with np.load(BF/'weights/native_candidate_vectors.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    for v in a.values():v.flags.writeable=False
    return a

@lru_cache(maxsize=1)
def scalar_validation():
    protocol=json.loads((FP/'protocol/frozen.json').read_text(encoding='utf-8'))
    records=json.loads((FP/'analysis/records.json').read_text(encoding='utf-8'))
    return protocol['sites'],{r['case_index']:r for r in records if r['published']}

def options():
    return {'phase':2676,'cases':[{'case':r['case_index'],'label':r['case_id'],'tokens':len(r['token_strings'])} for r in metadata()],
        'sites':[{'layer':l,'unit':j} for l,j in SITES],'coordinates':2560,'checkpoints':37,
        'validated_scalars':scalar_validation()[0],
        'checkpoint_semantics':'H0 is token embedding; H1..H36 are outputs of blocks0..35. H36 is residual before the model final RMSNorm, not the final normalized readout state. Layer/unit/coordinate indices are zero-based.',
        'boundary':'Five frozen unit windows on all-coordinate background, not a sparse semantic basis. BF16 short instruction and FP32 structured-answer instruction are different measured prefixes.'}

def query(case,layer,unit,coordinate,checkpoint,token):
    r=next((r for r in metadata() if r['case_index']==case),None)
    if r is None:raise HTTPException(status_code=404,detail='Unpublished native case')
    if (layer,unit) not in SITES:raise HTTPException(status_code=400,detail='Choose a published native unit, not an arbitrary hidden-coordinate index')
    t=len(r['token_strings'])-1 if token is None else token
    if not(0<=coordinate<2560 and 0<=checkpoint<37 and 0<=t<len(r['token_strings'])):raise HTTPException(status_code=400,detail='Physical coordinate/checkpoint/token out of range')
    z=native(case);fp=precise(case);wv=weights();li=LAYERS.index(layer);base=f'L{layer}_J{unit}_';k=coordinate
    x=z['full__x'][li,t].astype('float64');g=float(z['full__gate'][li,t,unit]);u=float(z['full__up'][li,t,unit]);a=float(z['full__a'][li,t,unit])
    values={'actual_Wgate_jk':float(wv[base+'gate'][k]),'actual_Wup_jk':float(wv[base+'up'][k]),'actual_Wdown_kj':float(wv[base+'down'][k]),
        'embedding_coordinate':float(z['full__h'][0,t,k]),'hidden_coordinate':float(z['full__h'][checkpoint,t,k]),'normalized_mlp_input':float(x[k]),
        'gate_unit':g,'up_unit':u,'actual_product_unit':a,'full_mlp_output_coordinate':float(z['full__down'][li,t,k]),
        'gate_input_term':float(wv[base+'gate'][k]*x[k]),'up_input_term':float(wv[base+'up'][k]*x[k]),'single_unit_down_term':float(wv[base+'down'][k]*a),
        'gate_all_input_sum':float(wv[base+'gate'].astype('float64')@x),'up_all_input_sum':float(wv[base+'up'].astype('float64')@x)}
    derivatives={};traces={}
    for kind in ('gate','up','down'):
        derivatives[kind]={}
        for part in ('all','content','format','eos'):
            terms=[]
            for branch in ('Y','N'):
                if kind=='down':xx=fp[f'{branch}__{base}a'];gg=fp[f'{branch}__L{layer}_down_g_{part}'][:,k]
                else:xx=fp[f'{branch}__L{layer}_x'][:,k];gg=fp[f'{branch}__{base}{kind}_g_{part}']
                terms.append(xx.astype('float64')*gg.astype('float64'))
            derivatives[kind][part]=float(terms[0].sum()-terms[1].sum())
            if part=='all':traces[kind]={'Y':terms[0].tolist(),'N':terms[1].tolist()}
    sites,records=scalar_validation();record=records[case];validation=[]
    for si,s in enumerate(sites):
        if (s['layer'],s['unit'],s['coordinate'])!=(layer,unit,k):continue
        validation.append({'site_index':si,'kind':s['kind'],'control':s['control'],'original_weight':s['original'],'vector_rms':s['rms'],
            'effects':[e for e in record['effects'] if e['indices']==[si]],
            'scope':'Actual single-scalar changes and halfdose where available; finite FP32 effect and derivative prediction are separate. Joint changes excluded here.'})
    return {'phase':2676,'case':case,'case_id':r['case_id'],'layer':layer,'unit':unit,'coordinate':k,'token':t,'token_string':r['token_strings'][t],'checkpoint':checkpoint,
        'prompt':r['prompt'],'values':values,'scalar_sequence_derivatives':derivatives,'all_token_scalar_terms':traces,
        'actual_scalar_validation':validation,'scalar_noop_error':record['noop_error'],
        'natural_short':r['natural_short'],'natural_structured':r['natural_structured'],'fp_structured_base':r['fp_structured_base'],
        'fp_structured_prompt':r['fp_structured_prompt'],'boundary':'Actual learned weights, hidden coordinates and MLP units are separate. H0=embedding, H1..H36=block0..35 outputs; H36 is before final model RMSNorm. BF16 native arithmetic differs slightly from real-number sums. FP32 derivatives use a different structured-answer prompt, both entire teacher-forced answers plusEOS, not free-generation causal necessity. Single-unit down contribution is only one addend of the wholebranch.'}
