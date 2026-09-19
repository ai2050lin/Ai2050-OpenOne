"""Bounded serial nonquantized matched-source replication, all native coordinates retained."""
import argparse,gc,os,sys
from rdc_relation_common import *
from rdc_relation_estimators import Bank,select,predict
MODELS={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def selected_material():
    return [r for split,n in [('train',128),('validation',32)] for r in [x for x in rows() if x['split']==split][:n]],rows(True)[:64]


def load(key,out):
    import torch,psutil
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as loading
    torch.set_num_threads(4);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';available=psutil.virtual_memory().available
    save(out/'resource_check.json',{'timestamp':stamp(),'available_host_after_imports':available,'required_host':11*1024**3,'CUDA_maximum_models':1,'quantization':False})
    if key=='qwen4':
        from phase2662_symmetric_mapping_contract import load_native
        return load_native('qwen4')
    assert available>11*1024**3,('Insufficient host reserve after imports',available)
    path=ROOT/'models/hf'/MODELS[key];tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    prior=loading.safe_open
    def pread(*a,**kw):kw['backend']='pread';return prior(*a,**kw)
    loading.safe_open=pread
    try:
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',max_memory={0:'13GiB','cpu':'10GiB'},offload_folder=str(out/'checkpoint_offload_index'),offload_state_dict=True,offload_buffers=True,local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=prior
    assert not getattr(model,'is_quantized',False) and model.dtype==torch.bfloat16
    disk=sum(p.stat().st_size for p in (out/'checkpoint_offload_index').rglob('*') if p.is_file());assert disk<20*1024**2
    # Auto placement left an untied head on disk. Keep its unchanged BF16 tensor resident,
    # within the separately measured spare GPU budget, to avoid rereading it at every score.
    head_resident=False
    if model.hf_device_map.get('lm_head')=='disk':
        from accelerate.hooks import remove_hook_from_module
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        keyname='lm_head.weight';index=read(path/'model.safetensors.index.json')['weight_map']
        free,total=torch.cuda.mem_get_info();headbytes=model.config.vocab_size*model.config.hidden_size*2
        assert free>headbytes+512*1024**2,('Readout residency reserve',free,headbytes)
        with safe_open(str(path/index[keyname]),framework='pt',device='cpu',backend='pread') as f:head=f.get_tensor(keyname)
        remove_hook_from_module(model.lm_head);set_module_tensor_to_device(model.lm_head,'weight','cuda:0',value=head);del head;head_resident=True
    assert psutil.virtual_memory().available>2*1024**3,('Measured postload host floor',psutil.virtual_memory().available)
    save(out/'load_audit.json',{'timestamp':stamp(),'available_host_after_load':psutil.virtual_memory().available,'device_map':model.hf_device_map,'dtype':str(model.dtype),'quantized':False,'offload_index_bytes':disk,
      'placement_revision':'auto13GiB GPU/10GiB CPU; measured host floor2GiB; original BF16 readout resident on GPU if head was disk-offloaded','untied_readout_resident_GPU':head_resident,
      'offload_method':'References to original local safetensors shards, not duplicated checkpoint files.','other_apps':'Only verified task-owned old read-only API25692/40236 temporarily stopped; editor/browser untouched.',
      'model_source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    return model,tok


class Capture:
    def __init__(self,model):
        self.depth=len(model.model.layers);self.early=self.depth//3;self.positions=[];self.data={};self.handles=[]
        for name,layer in [('early',self.early),('late',self.depth)]:
            def hook(m,a,o,name=name):self.data[name]=bits(o[0,self.positions])
            self.handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def capture(model,tok,observer,r,out,fresh,key):
    import torch
    path=out/f'fields/{r["sample_id"]}.npz';rp=out/f'rows/{r["sample_id"]}.json'
    if rp.exists():
        m=read(rp)
        if key!='qwen4':
            with np.load(path) as z:return {k:unbits(z[k]) for k in z.files},m
    enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True);positions=[];alignment=[]
    for p in r['anchors']:
        end=r['token_offsets'][p][1];choices=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b<=end and b>a];q=max(choices)
        assert q+1<len(enc['input_ids']);positions.extend([q,q+1]);alignment.append({'qwen4_char_end':end,'own_char_end':enc['offset_mapping'][q][1],'exact_endpoint':enc['offset_mapping'][q][1]==end})
    m={k:r[k] for k in ('sample_id','source_group','language','genre','split','text')}|{'positions':positions,'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],'alignment':alignment,'fresh':fresh}
    if key=='qwen4':
        z=load_field(r,fresh);a={'early':unbits(z['h12'][positions]),'late':unbits(z['h36'][[0,1,3,4]])};m['origin']='fresh' if fresh else 'main';m['full_field_sha']=sha(BASE/m['origin']/f'fields/{r["sample_id"]}.npz')
    else:
        observer.positions=positions;observer.data={};device=model.get_input_embeddings().weight.device
        with torch.inference_mode():
            ids=torch.tensor([enc['input_ids']],device=device);y=model.model(input_ids=ids,use_cache=False).last_hidden_state
            if len(list((out/'rows').glob('*.json')))<2:
                observer.positions=positions;again=model.model(input_ids=ids,use_cache=False).last_hidden_state;assert torch.equal(y,again);m['same_shape_repeat_bitwise']=True
            packet=observer.data;a={k:unbits(v) for k,v in packet.items()};assert all(np.isfinite(v).all() for v in a.values());npz(path,**packet);m['field_sha']=sha(path)
    save(rp,m);return a,m


def pack(model,collected):
    import torch
    arrays={k:[] for k in ('early','late','new_embedding','next_late')};meta=[]
    for a,m in collected:
        for anchor,j in enumerate((0,2)):
            tid=m['prompt_ids'][m['positions'][j]+1];device=model.get_input_embeddings().weight.device
            with torch.inference_mode():e=model.get_input_embeddings()(torch.tensor([tid],device=device))[0].float().cpu().numpy()
            for k,v in [('early',a['early'][j]),('late',a['late'][j]),('next_late',a['late'][j+1]),('new_embedding',e)]:arrays[k].append(v)
            meta.append({k:m[k] for k in ('sample_id','source_group','language','genre','split')}|{'anchor':anchor,'position':m['positions'][j],'next_position':m['positions'][j+1]})
    return {k:np.stack(v) for k,v in arrays.items()},meta


def dots(raw,training,scales):return {k:raw[k].astype(float)@training[k].astype(float).T/scales[k] for k in scales}


def kernel(dd,name):
    if name=='current_linear':return {'current':dd['early']},'current'
    if name=='current_quadratic':return {'current':dd['early']},'full_quadratic'
    x,e=dd['late'],dd['new_embedding']
    return {'current':x,'history_mean':e if name=='temporal_linear' else e+x*e},'history_mean'


def logprob(model,h):
    import torch
    device=model.get_input_embeddings().weight.device;x=torch.tensor(np.array(h),dtype=torch.bfloat16,device=device)
    return model.lm_head(model.model.norm(x)).float().log_softmax(-1)


def main(key):
    import torch
    out=BASE/'scale'/key;mainrows,freshrows=selected_material();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    guard(22*1024**2)
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'main_sources':160,'training_sources':128,'validation_sources':32,'fresh_sources':64,
      'main_ids':[r['sample_id'] for r in mainrows],'fresh_ids':[r['sample_id'] for r in freshrows],'shape':'Native batch1 unpadded raw text, no chat, no KV cache; BF16 nonquantized.',
      'early_layer':'floor(native_depth/3), index is a within-model checkpoint, not cross-model coordinate isomorphism.',
      'retention':'Four full early/final states around two anchors per larger-model source. Qwen4 reuses existing native fields without copy. Every coordinate enters fitting and scoring.',
      'alignment':'Last own native token whose end is not later than each Qwen4 anchor char endpoint; exact endpoints reported. Native IDs/widths differ and are not aligned by coordinate index.',
      'training':'Current linear/quadratic and previous-state+known-token embedding linear/bilinear. Same128 train32 validation64 fresh sources per model; hyperparameters selected independently.',
      'resources':'22MiB anticipated output per larger model, actual budget guarded; original checkpoint disk references, one loaded model. Native8-step continuation64 sources, complete probabilities and full-coordinate streaming errors, first4 full-field fixtures.',
      'limits':'Small scale replication, not all relation-atlas analyses repeated or proof of model-size law. Selection after material freeze but own fresh model fields captured only after own fit freeze.'})
    if (out/'load_audit.json').exists() and not (out/'placement_revision.json').exists():
        save(out/'initial_load_audit.json',read(out/'load_audit.json'));save(out/'initial_resource_check.json',read(out/'resource_check.json'))
        save(out/'placement_revision.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'reason':'First16 sources215.1seconds; projected full native generation exceeded original5400seconds. Change residency only, retain BF16 and exact checkpoint.',
          'initial_run_seconds_charged':330,'max_cumulative_running_seconds':5400,'guard':'Recompute first2 committed sources with changed placement and require all early/late bits equal before reusing earlier fields. Unchanged GPU arithmetic required; no prediction retuning.',
          'scope':'All original source/selection/generation tasks retained subject to a measured remaining runtime check.'})
    if key=='qwen14' and (out/'placement_revision.json').exists() and not (out/'software_retry.json').exists():
        save(out/'software_retry.json',{'timestamp':stamp(),'stage':'Residency reload completed but scalar bookkeeping called single-argument read with a default argument. No new states were committed.',
          'correction':'Use an explicit exists check; preserve all prior data. Charge40 additional rounded-up seconds for the unsuccessful loader attempt.','charged_seconds':40,'code':snapshot(Path(__file__))})
    if key=='qwen14' and not (out/'generation_batch_revision.json').exists():
        save(out/'generation_batch_revision.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'generation_code':snapshot(ROOT/'tests/glm5/phase2717_rdc_scale_generation.py'),
          'reason':'Improved residency passed bitwise checks but single-prefix generation still projects excessive repeated disk reads. Preserve64 cases and8 steps with batch4 left-padding and explicit native positions, then audit execution-shape drift.',
          'second_completed_capture_attempt_rounded_charge_seconds':500,'prior_saved_sources':len(list((out/'rows').glob('*.json'))),'source_capture_shapes_unchanged':True})
    model,tok=load(key,out);observer=Capture(model);start=time.monotonic();charged=read(out/'placement_revision.json').get('initial_run_seconds_charged',0) if (out/'placement_revision.json').exists() else 0
    if (out/'software_retry.json').exists():charged+=read(out/'software_retry.json')['charged_seconds']
    if (out/'generation_batch_revision.json').exists():charged+=read(out/'generation_batch_revision.json')['second_completed_capture_attempt_rounded_charge_seconds']
    save(out/'runtime.json',{'timestamp':stamp(),'model':key,'depth':observer.depth,'early_layer':observer.early,'width':model.config.hidden_size,'dtype':str(model.dtype),'quantized':False,'tokenizer_sha':sha(ROOT/'models/hf'/MODELS[key]/'tokenizer.json')})
    try:
      if key!='qwen4' and (out/'placement_revision.json').exists():
        checks=[]
        with torch.inference_mode():
          for r in mainrows[:2]:
            rp=out/f'rows/{r["sample_id"]}.json'
            if not rp.exists():continue
            m=read(rp);observer.positions=m['positions'];observer.data={};device=model.get_input_embeddings().weight.device
            model.model(input_ids=torch.tensor([m['prompt_ids']],device=device),use_cache=False)
            with np.load(out/f'fields/{r["sample_id"]}.npz') as z:
                for k,v in observer.data.items():assert np.array_equal(v,z[k]),('Placement changes native state',r['sample_id'],k)
            checks.append(r['sample_id'])
        save(out/'placement_invariance.json',{'timestamp':stamp(),'passed':True,'sources':checks,'all_retained_coordinates_bitwise_equal':True})
      collected=[]
      for i,r in enumerate(mainrows):
        collected.append(capture(model,tok,observer,r,out,False,key))
        if i%16==15:print('SCALE_CAPTURE',key,i+1,160,round(time.monotonic()-start,1),flush=True)
        guard(8*1024**2);assert time.monotonic()-start+charged<5400
      raw,meta=pack(model,collected);tr=np.array([i for i,m in enumerate(meta) if m['split']=='train']);va=np.array([i for i,m in enumerate(meta) if m['split']=='validation']);scales={k:float(np.mean(np.sum(raw[k][tr].astype(float)**2,axis=1))) for k in ('early','late','new_embedding')};dd=dots(raw,raw,scales);models={};choices={};grids={}
      for name in ('current_linear','current_quadratic','temporal_linear','temporal_bilinear'):
        kd,kn=kernel(dd,name);y=raw['next_late'] if name.startswith('temporal') else raw['late'];models[name],choices[name],grids[name]=select(kd,kn,y,tr,va,[(0,y.shape[1])]);print('SCALE_FIT',key,name,choices[name],flush=True)
      selected={s:min([n for n in choices if n.startswith(s)],key=lambda n:choices[n]['validation_normalized_MSE']) for s in ('current','temporal')};freeze_path=out/'frozen.json'
      frozen={'model':key,'choices':choices,'selected_by_validation':selected,'scales':scales,'training_indices':tr.tolist(),'main_rows':meta,'fresh_ids':[r['sample_id'] for r in freshrows],
        'model_array_sha':{n:{k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in m.items()} for n,m in models.items()},'main_artifact_sha':{p.name:sha(p) for p in (out/'rows').glob('*.json') if p.stem not in [r['sample_id'] for r in freshrows]},'fit_source_sha':sha(ROOT/'tests/glm5/rdc_relation_estimators.py'),
        'model_weights_storage':'Fitted dual arrays reproducible from retained all-coordinate training states, original embedding rows, exact ridge/mix/scales and archived source; hashes saved, no truncation.'}
      immutable(freeze_path,frozen);collectedfresh=[]
      for i,r in enumerate(freshrows):
        collectedfresh.append(capture(model,tok,observer,r,out,True,key))
        if i%16==15:print('SCALE_FRESH',key,i+1,64,flush=True)
      fresh,fr=pack(model,collectedfresh);trainraw={k:v[tr] for k,v in raw.items()};cross=dots(fresh,trainraw,scales);reports={};preds={}
      with torch.inference_mode():
       for name in models:
        kd,kn=kernel(cross,name);p=predict(models[name],Bank.gram(kd,kn,choices[name]['mix']));preds[name]=p;target=fresh['next_late'] if name.startswith('temporal') else fresh['late'];mse=np.mean((p-target)**2,axis=1);per=[]
        for begin in range(0,len(p),8):
            lq=logprob(model,target[begin:begin+8]);lp=logprob(model,p[begin:begin+8]);kl=(lq.exp()*(lq-lp)).sum(-1);ag=lq.argmax(-1)==lp.argmax(-1)
            for j in range(len(kl)):per.append({'sample_id':fr[begin+j]['sample_id'],'anchor':fr[begin+j]['anchor'],'language':fr[begin+j]['language'],'MSE':float(mse[begin+j]),'KL':float(kl[j]),'argmax_agreement':bool(ag[j])})
        reports[name]={'MSE':float(mse.mean()),'relative_MSE':float(mse.mean()/np.mean(target**2)),'KL':float(np.mean([r['KL'] for r in per])),'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in per]))};save(out/f'fresh_{name}.json',{'summary':reports[name],'rows':per})
        if name in selected.values():npz(out/f'fresh_{name}.npz',prediction=p,coordinate_MSE=np.mean((p-target)**2,axis=0))
        print('SCALE_RESULT',key,name,reports[name],flush=True)
      from phase2717_rdc_scale_generation import run as run_generation
      generation=run_generation(model,tok,observer,scales,models,choices,selected,trainraw,collectedfresh,out,key)
      assert time.monotonic()-start+charged<5400
      result={'timestamp':stamp(),'model':key,'depth':observer.depth,'width':model.config.hidden_size,'early_layer':observer.early,'fresh_sources':64,'fresh_anchors':len(fr),'selected_before_fresh':selected,'reports':reports,
        'exact_char_endpoint_fraction':float(np.mean([x['exact_endpoint'] for a,m in collected+collectedfresh for x in m['alignment']])),
        'generation':generation['by_scope'],'generation_shape_audit':generation,
        'elapsed_seconds_this_attempt':time.monotonic()-start,'cumulative_seconds_with_rounded_restart_charges':time.monotonic()-start+charged,'scope':'Within-model predictors on matched sources. Not shared coordinate indices or semantic accuracy; only conditional native generation, not self-fed larger-model surrogate.'}
      save(out/'result.json',result);print('RELATION_SCALE_COMPLETE',key,result,flush=True)
    finally:observer.close();del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);a=p.parse_args();main(a.model)
