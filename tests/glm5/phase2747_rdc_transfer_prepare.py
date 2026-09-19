"""Frozen paired-text mappings, retaining every coordinate and explicit controls."""
from collections import defaultdict
from rdc_formation_common import *
from phase2747_rdc_followup_contract import freeze

DIRECT=[('python','en'),('en','python'),('zh','en'),('en_reordered','en')]
NAMES=['identity','query_only','affine','query_conditioned','shuffled_pair']


def main():
    start=time.monotonic()
    folder=OUT/'transfer'
    if (folder/'preparation.json').exists():
        print('FORMATION_TRANSFER_ALREADY_PREPARED',flush=True)
        return
    protocol,follow=freeze()
    rows=gzread(OLD/'transfer/material.json.gz')
    probes=read(OLD/'probes/protocol.json')['probes']
    lookup=defaultdict(dict)
    for r in rows:lookup[r['source_group']][r['representation']]=r
    with np.load(OLD/'transfer/mapping.npz') as z:
        mapping={k:z[k] for k in z.files}
    with np.load(OLD/'prototypes/qwen4.npz') as z:
        query_only=unbits(z['postnorm']).astype(np.float64)
    records=[]
    for source,target in DIRECT:
        direction=source+'_to_'+target
        for group in follow['program_transfer_groups']:
            src,dst=lookup[group][source],lookup[group][target]
            name=rank(direction+'/'+group)[:24]
            receipt_path=OUT/'transfer_prediction'/direction/'commits'/(name+'.json')
            record_path=folder/'records'/direction/(name+'.json')
            if record_path.exists():
                records.append(read(record_path))
                continue
            with np.load(OLD/'transfer/fields'/(src['sample_id']+'.npz')) as z:
                source_postnorm=z['postnorm']
                x=unbits(source_postnorm).astype(np.float64)
            with np.load(OLD/'transfer/fields'/(dst['sample_id']+'.npz')) as z:
                target_postnorm=z['postnorm']
                y=unbits(target_postnorm).astype(np.float64)
                original_readout_statistics=z['full_vocabulary_statistics']
            xx=np.stack([np.ones_like(x),x,query_only],-1)
            predicted=[x,query_only]
            for candidate in NAMES[2:]:
                beta=mapping[direction+'__'+candidate]
                predicted.append(np.einsum('qdi,di->qd',xx[...,:beta.shape[-1]],beta,optimize=True))
            prediction=np.stack(predicted).astype(np.float32)
            receipt=commit_array('transfer_prediction/'+direction,name,
                predictions=prediction,target_postnorm_BF16=target_postnorm,
                source_postnorm_BF16=source_postnorm,original_readout_statistics=original_readout_statistics)
            metrics=[]
            for i,p in enumerate(probes):
                metrics.append({'query':i,'query_split':p['split'],
                    'MSE':{c:float(((predicted[j][i]-y[i])**2).mean()) for j,c in enumerate(NAMES)}})
            record={'timestamp':stamp(),'direction':direction,'source_group':group,'split':dst['split'],
                'source_sample_id':src['sample_id'],'target_sample_id':dst['sample_id'],
                'source_material':src,'target_material':dst,'field':receipt,'metrics':metrics,
                'available_information':'Complete observed source-expression response to known diagnostic query; no observed target response or correct answer enters mapping.',
                'mapping_sha256':sha(OLD/'transfer/mapping.npz')}
            save(record_path,record)
            records.append(record)
        print('FORMATION_TRANSFER_PREPARED',direction,len(records),flush=True)
    compressed(folder/'records.json.gz',records)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'records':len(records),
        'directions':[a+'_to_'+b for a,b in DIRECT],'heldout_groups':64,'queries':100,'paths':NAMES,
        'full_coordinate_predictions':256*100*5,'new_native_model_calls':0,
        'mapping_sha256':sha(OLD/'transfer/mapping.npz'),
        'previous_fit_sha256':sha(OLD/'transfer/fit_result.json'),
        'material_sha256':sha(folder/'records.json.gz'),'seconds':time.monotonic()-start,
        'boundary':'Reused frozen mapping and previously observed material; new complete-vocabulary/readout controls remain pending. EN/ZH/Python are all text, and mapping is not declared invertible or semantic-only.'}
    save(folder/'preparation.json',result)
    ledger('phase2747_transfer_prepare',result['seconds'])
    print('FORMATION_TRANSFER_PREPARATION_COMPLETE',result['records'],result['seconds'],flush=True)


if __name__=='__main__':main()
