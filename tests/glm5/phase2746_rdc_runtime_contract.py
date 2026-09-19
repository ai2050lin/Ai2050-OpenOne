"""Own-history observation on the complete detailed natural and relation panels."""
from collections import Counter
from rdc_construction_common import *
from rdc_construction_storage import verify_storage

OUT=BASE/'phase2746/runtime'


def freeze():
    verify_storage()
    if (OUT/'protocol.json').exists():
        return read(OUT/'protocol.json'),gzread(OUT/'material.json.gz')
    detailed=set(read(OLD/'material/protocol.json')['detailed_prefix_ids'])
    natural=[r for r in gzread(OLD/'material/natural.json.gz') if r['sample_id'] in detailed]
    controlled=gzread(BASE/'material.json.gz')['models']['qwen4']['rows']
    assert len(natural)==576 and len(controlled)==320
    rows=[]
    for r in natural:
        rows.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'cohort':r['cohort'],
            'family':'natural_'+r['cohort'],'kind':'natural','language':r['language'],'split':r['split'],
            'prompt_ids':r['prompt_ids'],'original_text':r['text'],'source_record':r['sample_id'],
            'reference_field':str((OLD/'capture/fields'/(r['sample_id']+'.npz')).relative_to(ROOT)),
            'reference_field_sha256':read(OLD/'capture/commits'/(r['sample_id']+'.json'))['array_sha256'],
            'max_new_tokens':32,'novelty':'Existing exposed discovery prefix; the native32step continuation and all-layer/all-unit runtime observation are new. Not an independent document confirmation.'})
    for r in controlled:
        rows.append({k:r[k] for k in ['sample_id','source_group','family','language','split','prompt_ids','original_text','pair_id','world','case','target']}|
            {'kind':'controlled','cohort':r['family'],'source_record':r['sample_id'],
             'reference_field':str((BASE/'capture/qwen4/fields'/(r['sample_id']+'.npz')).relative_to(ROOT)),
             'reference_field_sha256':read(BASE/'capture/qwen4/commits'/(r['sample_id']+'.json'))['sha256'],
             'max_new_tokens':128,'novelty':'Existing320exact-token-bag matched expressions; oldnativeB8histories must replay exactly, and are not counted as new trajectories.'})
    assert len({r['sample_id'] for r in rows})==896
    compressed(OUT/'material.json.gz',rows)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),'phase':2746,'model':'qwen4',
        'question':'How do complete native read/gate/write and source-attention responses evolve across all layers and the first own-history steps, on natural language as well as matched relations?',
        'discovery_rows':896,'natural_rows':576,'natural_documents':len({r['source_group'] for r in natural}),
        'controlled_rows':320,'families':dict(Counter(r['family'] for r in rows)),
        'splits':{kind:dict(Counter(r['split'] for r in rows if r['kind']==kind)) for kind in ['natural','controlled']},
        'full_field_steps':8,'natural_generation_cap':32,'controlled_generation_cap':128,
        'batch':8,'precision':'Original BF16 weights and standard original eager modules, no quantization or injected state/answer.',
        'numerical_admission':'B1 original all-hidden-boundary/postnorm fixtures and deterministic B8 pilot repeat; all320controlled fulloutput token sequences must match old B8 native histories.',
        'pilot_batch_indices':[0,80,160,240,320,400,480,575],
        'B1_fixture_indices':[0,192,384,575,576,642,705,771,834,895],
        'fields':'For first8actualsteps: every native H boundary, every block attention/read input/preMLP/x/write, all9728gate/up/product units, allheadQbeforeRoPE and all-source attention. Original KV at blocks0/12/35 stored once for prefix then only newly appended entries; full-source history reconstructable at those blocks. Latersteps keep allH/postnorm but no completeMLP/sourcefields.',
        'noncoverage':'Not every prefix token MLP field; not all100diagnostic queries; not all-layer fullKV histories; no source is replaced by a mean or top-k. These omitted axes have input/checkpoint/config reexecution paths.',
        'labels':'Natural cohort/UD/CMRC metadata can describe examples, not define internal semantics. Generated natural continuations have no unique gold; censoring is not semantic failure.',
        'analysis_plan':'All-unit raw/RMS read-write coactivity, layer/step conditional reuse and whole-coordinate energy with cross terms. Full remaining-network differential and finite-change checks use this fixed material, followed by same-input native-parameter-constrained/unconstrained comparisons. New confirmation must exclude discovery sources.',
        'scope':'An expanded native observation, not an extracted algorithm or completed mechanism. Scientific discovery and subsequent prediction/formation tasks remain open.',
        'material_sha256':sha(OUT/'material.json.gz'),'source_natural_sha256':sha(OLD/'material/natural.json.gz'),
        'source_controlled_sha256':sha(BASE/'material.json.gz'),'storage_manifest_sha256':sha(BASE/'phase2746/storage.json')}
    immutable(OUT/'protocol.json',protocol)
    return protocol,rows


if __name__=='__main__':
    p,r=freeze();print('RUNTIME_PROTOCOL_FROZEN',len(r),p['families'],flush=True)
