"""Compile all predeclared candidates on validation only, then freeze routes."""
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,final_norm,config,cuda_singleton,CUDA_TASKS
from rdc_history_prediction import past_cache,tensor,position_factors,complete_block,normalized_query,weights
from phase2746_rdc_history_prediction_contract import OUT,freeze

INPUTS=['native_history','source_value_permuted']
CONTROLS=['true_correspondence','target_correspondence_shuffled']
ROUTES=['direct_complete_coordinate_H36','predicted_H35_native_block35',
    'predicted_H35_general_Q35_native_remainder','predicted_H35_general_Q35_native_head_RMS_native_remainder']


def compile_routes(pred,layer,norm,key,value,cos,sin):
    c=config();h35=tensor(pred[:2560])[None,None];h36=tensor(pred[2560:5120])[None,None]
    q=tensor(pred[5120:]).reshape(32,128)
    endpoints=[h36,complete_block(layer,h35,key,value,cos,sin),
        complete_block(layer,h35,key,value,cos,sin,query=q),
        complete_block(layer,h35,key,value,cos,sin,query=normalized_query(q,layer.self_attn.q_norm.weight))]
    return np.stack([final_norm(h,norm,c.rms_norm_eps)[0,0].cpu().numpy() for h in endpoints])


def main():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cuda_singleton(CUDA_TASKS|{'phase2746_rdc_history_features.py','phase2746_rdc_history_fit.py',Path(__file__).name})
    verify_storage(512*1024**2);start=time.monotonic();protocol,rows=freeze();directory=OUT/'validation'
    if (OUT/'frozen.json').exists():return
    validation=[i for i,r in enumerate(rows) if r['split']=='validation'];vrows=[rows[i] for i in validation]
    w=weights(vrows);predictions=[];fitmeta=[]
    for name in INPUTS:
        meta=read(OUT/'fit'/(name+'.json'));assert meta['all_passed'];assert sha(BASE/meta['field_path'])==meta['field_sha256']
        with np.load(BASE/meta['field_path']) as z:
            assert z['validation_row_indices'].tolist()==validation
            predictions.append(z['validation_predictions'])
        fitmeta.append(meta)
    feature=read(OUT/'features/result.json')
    with np.load(BASE/feature['field_path']) as z:target=unbits(z['target_postnorm'][validation]).astype(float)
    c=config();rotary=Qwen3RotaryEmbedding(c,device='cuda');layer=loaded_block(35,torch.float32)
    norm=checkpoint_tensor('model.norm.weight',torch.float32)
    post=np.empty((2,2,4,4,len(validation),2560),np.float32)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    with torch.no_grad():
        for begin in range(0,len(validation),3):
            group=vrows[begin:begin+3];assert len({r['sample_id'] for r in group})==1
            source=BASE/group[0]['field_path'];assert sha(source)==group[0]['field_sha256']
            with np.load(source) as z:
                for offset,row in enumerate(group):
                    index=begin+offset;kk,vv=past_cache(z,row['step'])
                    key,value=tensor(kk)[None],tensor(vv)[None]
                    assert key.shape[-2]==len(row['prompt_ids'])-1
                    cos,sin=position_factors(rotary,len(row['prompt_ids'])-1)
                    for input_id in range(2):
                      for control in range(2):
                        for regularizer in range(4):
                            post[input_id,control,regularizer,:,index]=compile_routes(
                                predictions[input_id][control,regularizer,index],layer,norm,key,value,cos,sin)
            if (begin+3)%48==0:print('HISTORY_VALIDATION',begin+3,len(validation),round(time.monotonic()-start,1),flush=True)
    error=((post.astype(float)-target[None,None,None,None])**2).mean(-1)
    score=np.sum(error*w[None,None,None,None,:],-1)
    chosen=[];selected=[]
    for inp,name in enumerate(INPUTS):
      for control,control_name in enumerate(CONTROLS):
        for route,route_name in enumerate(ROUTES):
            regularizer=int(np.argmin(score[inp,control,:,route]));lam=protocol['lambdas'][regularizer]
            record={'name':name+'__'+control_name+'__'+route_name,'input_variant':name,'control':control_name,
                'route':route_name,'input_index':inp,'control_index':control,'route_index':route,
                'lambda_index':regularizer,'lambda':lam,'validation_postnorm_MSE':float(score[inp,control,regularizer,route]),
                'all_lambda_validation_MSE':score[inp,control,:,route].tolist(),
                'ridge_effective_df':fitmeta[inp]['ridge_effective_degrees_of_freedom'][regularizer]}
            chosen.append(record);selected.append(post[inp,control,regularizer,route])
    assert np.isfinite(post).all() and np.isfinite(error).all()
    path=FIELD_STORE/'history_validation/selected.npz';arrays={'selected_postnorm':np.stack(selected),
        'all_grid_point_MSE':error,'all_grid_weighted_MSE':score,'validation_weights':w,'validation_row_indices':np.array(validation)}
    verify_storage(sum(a.nbytes for a in arrays.values()));npz(path,**arrays)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'validation_points':len(validation),
        'candidate_grid_count':64,'selected_routes':chosen,'field_path':path.relative_to(BASE).as_posix(),
        'field_sha256':sha(path),'field_bytes':path.stat().st_size,'row_ids':[r['point_id'] for r in vrows],
        'test_points_used_for_selection':0,'seconds':time.monotonic()-start,
        'scope':'Validation-selected whole-coordinate predictive routes; heldout and autonomous results not yet computed.'}
    save(directory/'selected.json',result)
    native_candidates=[r for r in chosen if r['input_variant']=='native_history' and r['control']=='true_correspondence' and r['route']!=ROUTES[0]]
    best=min(native_candidates,key=lambda r:r['validation_postnorm_MSE'])
    frozen={'timestamp':stamp(),'source':snapshot(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'validation_result_sha256':sha(directory/'selected.json'),'routes':chosen,
        'autonomous_primary_native_constrained_route':best['name'],
        'autonomous_direct_baseline_route':next(r['name'] for r in chosen if r['input_variant']=='native_history' and r['control']=='true_correspondence' and r['route']==ROUTES[0]),
        'test_results_inspected_before_freeze':False,'new_confirmation_results_inspected_before_freeze':False,
        'secondary_shortcut_baseline':'Training-weighted H36 mean by known current input token ID, with global training mean for unseen IDs. Current native emitted token and future H35/Q remain unavailable.',
        'output_compilation':'Predicted FP32 finalnorm vector rounded BF16, original tied BF16 readout. Every test point projects all20forecast/baseline vectors and1original-state reference together at fixedB21; record reference-vs-original-native-trajectory argmax differences. These are state-forecast scores on native histories, not own-history generation.',
        'scope':'Frozen before test prediction. Selection by validation postnorm MSE does not promise best language behavior or full-vocabulary KL.'}
    immutable(OUT/'frozen.json',frozen);ledger('phase2746_history_validation',result['seconds'])
    print('HISTORY_ROUTES_FROZEN',best['route'],best['lambda'],result['seconds'],flush=True)


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:
        failure(OUT/'validation',start,exc);raise
