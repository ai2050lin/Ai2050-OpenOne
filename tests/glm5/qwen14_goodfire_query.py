"""Query every B28--B39 native unit with source-linked evidence.

Example: python -B tests/glm5/qwen14_goodfire_query.py --block 39 --unit 0
Use --output <result-name.json> to retain a provenance-bearing feature card.
"""
import argparse
import json
import numpy as np
import qwen14_goodfire_extension as E
C=E.C


def full_relations(block,unit):
    """Preserve every adjacent native index; never use a top-k surrogate."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    if block not in E.ALL or not 0<=unit<C.WIDTH:raise ValueError('Invalid native unit')
    result={}
    for direction,other in [('incoming',block-1),('outgoing',block+1)]:
        if other not in E.ALL:continue
        if direction=='outgoing':
            vector=C.unbits(np.load(E.source(block)/f'B{block}-W-down.npy',mmap_mode='r')[:,unit])
            cosines={}
            for branch in ['gate','up']:
                matrix=C.unbits(np.load(E.source(other)/f'B{other}-W-{branch}.npy',mmap_mode='r'))
                den=np.linalg.norm(matrix,axis=1)*np.linalg.norm(vector)
                cosines[branch]=np.divide(matrix@vector,den,out=np.zeros(C.WIDTH),where=den>0).tolist()
                del matrix
        else:
            matrix=C.unbits(np.load(E.source(other)/f'B{other}-W-down.npy',mmap_mode='r'))
            norm=np.linalg.norm(matrix,axis=0);cosines={}
            for branch in ['gate','up']:
                vector=C.unbits(np.load(E.source(block)/f'B{block}-W-{branch}.npy',mmap_mode='r')[unit])
                den=norm*np.linalg.norm(vector)
                cosines[branch]=np.divide(vector@matrix,den,out=np.zeros(C.WIDTH),where=den>0).tolist()
            del matrix
        correlations={}
        for split,(dlo,dhi) in C.SPLITS.items():
            own=C.unbits(E.data(block,'A')[dlo*8:dhi*8,unit]);own-=own.mean()
            denom_own=np.linalg.norm(own);all_values=[]
            raw=E.data(other,'A')
            for lo in range(0,C.WIDTH,256):
                a=C.unbits(raw[dlo*8:dhi*8,lo:lo+256]);a-=a.mean(0)
                den=np.linalg.norm(a,axis=0)*denom_own
                all_values.extend(np.divide(own@a,den,out=np.zeros(a.shape[1]),where=den>0).tolist())
            correlations[split]=all_values
        result[direction]={'other_block':other,'other_units':list(range(C.WIDTH)),
            'static_parameter_cosines':cosines,'observed_A_correlations':correlations,
            'scope':'All adjacent indices in native order. Static alignment omits normalization/attention; correlation does not identify causal edges.'}
    return result


def card(block,unit):
    if block not in E.ALL or not 0<=unit<C.WIDTH:
        raise ValueError('Expected zero-based B28..B39 and U0..U17407')
    from transformers import AutoTokenizer
    root=E.source(block);profile=C.read(root/f'B{block}-profile.json')
    cues=C.read(root/f'B{block}-cues.json')['units'][unit]
    prediction=C.read(root/f'B{block}-prediction.json')['fields']
    readout=C.read(root/f'B{block}-readout.json')
    tokenizer=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    docs=C.read(E.OUT/'documents.json')['docs'];tokens=np.load(E.OUT/'tokens.npy')
    extrema=np.load(root/'prefix-extrema.npz');examples={}
    for field in ['G','V','A']:
        examples[field]={}
        raw=np.load(root/f'B{block}-{field}.npy',mmap_mode='r')
        for split in C.SPLITS:
            rows=[]
            for pole,label in [('positive','high'),('negative','low')]:
                indices=profile['fields'][field][split][pole+'_indices'][unit]
                for rank,idx in enumerate(indices):
                    di,pi=divmod(idx,8);pos=C.POSITIONS[pi]
                    rows.append({'pole':pole,'rank':rank+1,'doc_id':docs[di]['doc_id'],'document_index':di,'position':pos,
                        'activation':float(C.unbits(raw[di,pi,unit])),
                        'prefix':tokenizer.decode(tokens[di,max(0,pos-256):pos+1].tolist()),
                        'future_tokens_excluded':True,'source':docs[di]['source']})
                key=f'B{block}_{field}_{split}_{label}'
                idx=int(extrema[key+'_index'][unit]);di,pos=divmod(idx,1024)
                rows.append({'pole':pole,'scope':'all positions 512..1023 extreme','doc_id':docs[di]['doc_id'],
                    'document_index':di,'position':pos,'activation':float(extrema[key+'_value'][unit]),
                    'prefix':tokenizer.decode(tokens[di,max(0,pos-256):pos+1].tolist()),'future_tokens_excluded':True})
            examples[field][split]=rows
    with np.load(E.OUT/f'B{block}-regimes.npz') as regs:
        regimes={k:np.asarray(regs[k][unit]).tolist() for k in regs.files}
    parents={}
    for b in [block-1,block]:
        if b not in range(28,39):continue
        r=E.OLD if b<35 else E.OUT
        d=C.read(r/f'B{b}-B{b+1}-prediction.json')
        if b==block-1:
            parents['incoming']={'source_block':b,'source_unit':d['parents'][unit],
                'discovery_correlation':d['discovery_correlation'][unit],
                'holdout_mse_ratio':d['splits']['holdout']['ratio'][unit],'edge_type':'observation_predictor','causal_confirmed':False}
        else:
            targets=[i for i,p in enumerate(d['parents']) if p==unit]
            parents['outgoing_selected_by_targets']={'target_block':b+1,'units':targets,
                'scope':'Targets whose discovery-selected predictor uses this unit; all other associations remain computable from raw fields.'}
    effects=[]
    oldrep=C.read(E.OLD/'report.json')
    for summary in [oldrep['intervention'],C.read(E.OUT/'intervention-summary.json') if (E.OUT/'intervention-summary.json').exists() else {}]:
        for stage in ['natural','capability']:
            effects += [{**r,'stage':stage} for r in summary.get(stage,{}).get('rows',[]) if r['block']==block and r['unit']==unit]
    with np.load(E.OUT/f'B{block}-scaled-prediction.npz') as scaled:
        secondary={k:np.asarray(scaled[k][unit]).tolist() for k in scaled.files}
    return {'schema':'qwen14_native_feature_card.v2','feature_id':f'{E.SNAP}:B{block}:mlp.down_proj.input:U{unit}',
        'model_snapshot':E.SNAP,'block':block,'unit':unit,'evidence_level':'observation_with_descriptive_prediction_screen',
        'semantic_confirmed':False,'original_run_id':profile['run_id'],
        'statistics':{f:{k:profile['fields'][f][k][unit] for k in ['mean','std','min','max']} for f in ['G','V','A']},
        'candidate_context_cues':cues,'context_examples':examples,'regimes':regimes,
        'context_display_scope':'At most257 tokens ending at readout; complete1024-token input and full causal prefix recoverable from tokens.npy by document_index.',
        'full_input_tokens':str(E.OUT/'tokens.npy'),
        'secondary_standardized_prediction':secondary,
        'prediction':{f:{s:{k:v[unit] for k,v in prediction[f]['splits'][s].items()} for s in ['confirm','holdout']} for f in ['G','V','A']},
        'parameter_sources':{'root':str(root),'gate':f'B{block}-W-gate.npy row {unit}',
            'up':f'B{block}-W-up.npy row {unit}','down':f'B{block}-W-down.npy column {unit}','all_coordinates':5120},
        'direct_readout_proxy':{pole:[{'token_id':i,'text':tokenizer.decode([i]),'value':v} for i,v in
            zip(readout[pole+'_ids'][unit],readout[pole+'_values'][unit])] for pole in ['positive','negative']},
        'proxy_scope':readout['scope'],'adjacent_relations':parents,'finite_write_interventions':effects,
        'limits':['Context keywords are retrieval cues, not semantic definitions.','No per-unit heldout semantic minimal-pair test.',
                  'Static readout omits intervening computation and context normalization.','Shared token histories can cause adjacent correlation.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--block',type=int,required=True);parser.add_argument('--unit',type=int,required=True)
    parser.add_argument('--output');parser.add_argument('--full-relations',action='store_true');args=parser.parse_args();E.configure()
    result=card(args.block,args.unit)
    if args.full_relations:result['all_adjacent_relations']=full_relations(args.block,args.unit)
    if args.output:
        if '/' in args.output or '\\' in args.output:raise ValueError('Output must be a result filename')
        E.save(E.OUT/args.output,result);print(str(E.OUT/args.output))
    else:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
