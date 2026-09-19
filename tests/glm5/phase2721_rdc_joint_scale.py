"""Sequential nonquantized matched-source layer/first-boundary/state replication."""
import argparse,gc,os,sys
from rdc_joint_common import *
from rdc_joint_capture import ledger,array_identity
from rdc_relation_estimators import select,predict,Bank
from phase2717_rdc_relation_scale import dots,kernel

MODELS={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def selected():
    material=[]
    for split,n in [('train',128),('validation',32)]:
        for lang in ('en','zh'):material.extend([r for r in rows() if r['split']==split and r['language']==lang][:n//2])
    fresh=[]
    for lang in ('en','zh'):fresh.extend([r for r in rows(True) if r['language']==lang][:32])
    return material,fresh


def load(key,out):
    import torch,psutil
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as loading
    torch.set_num_threads(4);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    available=psutil.virtual_memory().available
    save(out/'resource_check.json',{'timestamp':stamp(),'host_available_bytes':available,'host_preflight_minimum':10*1024**3,
        'maximum_cuda_models':1,'quantization':False,'actual_other_app_action':'No process termination by this loader; caller must verify idle previous model jobs.'})
    if key=='qwen4':
        from phase2662_symmetric_mapping_contract import load_native
        return load_native('qwen4')
    assert available>10*1024**3,('Measured host reserve below reduced9GiB CPU dispatch requirement',available)
    path=ROOT/'models/hf'/MODELS[key]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    prior=loading.safe_open
    def pread(*a,**kw):kw['backend']='pread';return prior(*a,**kw)
    loading.safe_open=pread
    try:
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',max_memory={0:'13GiB','cpu':'9GiB'},
            offload_folder=str(out/'checkpoint_offload_index'),offload_state_dict=True,offload_buffers=True,
            local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=prior
    disk=sum(p.stat().st_size for p in (out/'checkpoint_offload_index').rglob('*') if p.is_file())
    assert disk<20*1024**2
    head_resident=False
    if model.hf_device_map.get('lm_head')=='disk':
        from accelerate.hooks import remove_hook_from_module
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        index=read(path/'model.safetensors.index.json')['weight_map'];headbytes=model.config.vocab_size*model.config.hidden_size*2
        free,total=torch.cuda.mem_get_info();assert free>headbytes+512*1024**2
        with safe_open(str(path/index['lm_head.weight']),framework='pt',device='cpu',backend='pread') as f:head=f.get_tensor('lm_head.weight')
        remove_hook_from_module(model.lm_head);set_module_tensor_to_device(model.lm_head,'weight','cuda:0',value=head)
        del head;head_resident=True
    assert not getattr(model,'is_quantized',False) and model.dtype==torch.bfloat16
    assert psutil.virtual_memory().available>2*1024**3
    save(out/'load_audit.json',{'timestamp':stamp(),'model':key,'dtype':str(model.dtype),'quantized':False,'device_map':model.hf_device_map,
        'host_available_after_load':psutil.virtual_memory().available,'offload_index_bytes':disk,'head_original_BF16_GPU_resident':head_resident,
        'method':'Original checkpoint references, no duplicated full checkpoint, auto13GiB GPU/9GiB CPU with measured2GiB host floor. CPU dispatch lowered1GiB after11GiB host preflight missed by79MiB; not quantization.',
        'model_source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    return model,tok


class Capture:
    def __init__(self,model):
        self.depth=len(model.model.layers);self.early=self.depth//3;self.mid=2*self.depth//3
        self.positions=[];self.data={};self.handles=[]
        def embedding(m,a,o):self.data[0]=bits(o[0,self.positions])
        self.handles.append(model.get_input_embeddings().register_forward_hook(embedding))
        for index,layer in enumerate(model.model.layers,1):
            def hook(m,a,o,index=index):self.data[index]=bits(o[0,self.positions])
            self.handles.append(layer.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def main(key):
    import torch
    out=BASE/'scale'/key
    if (out/'result.json').exists():return
    mainrows,freshrows=selected();guard(70*1024**2)
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,
        'main_ids':[r['sample_id'] for r in mainrows],'fresh_ids':[r['sample_id'] for r in freshrows],
        'train':128,'validation':32,'fresh':64,'max_models_in_parallel':1,'nonquantized':True,
        'collection':'Embedding and every block at first,second,and two anchors/+1 in RAM; all-coordinate stratum moments retained. Full early/mid/final/postnorm at all6 positions retained for each source plus per-layer energies/array identities.',
        'early_mid':'floor(depth/3),floor(2*depth/3), each own coordinate system; Q4 H12/H24 (not old H23), Q14/GLM own depths.',
        'alignment':'Last own token whose char endpoint is no later than Q4 anchor; exact char endpoint coverage reported. Future-token positions are native model positions, not a cross-tokenizer one-to-one assumption.',
        'rules':'Current linear/quadratic and previous H+known newE linear/bilinear;128train32validation select before64fresh. Old4B full-data model is not transplanted into a different width.',
        'boundary':'Training-only shared first early-to-mid update, coordinatewise affine/identity controls; no universal boundary/sink assumption.',
        'probability':'Same actual native BF16 norm/head for both predicted and reference raw states; original saved postnorm used for separate numerical floor. Different from main FP32 candidate readout; no silent numeric mixing.',
        'scope':'Targeted independent model replication, not repeating every relation matrix, KL training or32step surrogate on larger models.'})
    save(out/'execution_source.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_protocol_sha':sha(out/'protocol.json'),
        'alignment_refinement':'Preserve original native Q4 anchor index; other tokenizers use documented last available character endpoint. Shared multibyte-piece offsets can otherwise select a different Q4 token. Existing scientific partitions/candidates unchanged.'})
    start=time.monotonic();model,tok=load(key,out);observer=Capture(model);device=model.get_input_embeddings().weight.device
    depth,width=observer.depth,model.config.hidden_size
    strata={};captured=[];models={};choices={};frozen=None;scalar={}
    def collect(r):
        enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
        pos=[0,1];align=[]
        for p in r['anchors']:
            end=r['token_offsets'][p][1]
            q=(p if key=='qwen4' else max(i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and b<=end))
            assert q+1<len(enc['input_ids']);pos.extend([q,q+1]);align.append(enc['offset_mapping'][q][1]==end)
        observer.positions=pos;observer.data={}
        x=torch.tensor([enc['input_ids']],device=device)
        final=model.model(input_ids=x,use_cache=False).last_hidden_state
        allfield=np.stack([observer.data[i] for i in range(depth+1)])
        if key=='qwen4':
            original=field(r,r['split']=='confirmation')
            assert np.array_equal(allfield,original['layers']),('Q4 all-layer replay differs',r['sample_id'])
        post=bits(final[0,pos]);check=False
        if len(captured)<2:
            repeat=model.model(input_ids=x,use_cache=False).last_hidden_state
            assert torch.equal(final,repeat) and np.array_equal(allfield,np.stack([observer.data[i] for i in range(depth+1)]))
            check=True
        dtype=model.get_input_embeddings().weight.dtype
        incoming=[]
        for q in (pos[2],pos[4]):
            # Calling only embedding triggers a hook: take allfield before this point.
            observer.positions=[0]
            incoming.append(bits(model.get_input_embeddings()(torch.tensor([[enc['input_ids'][q+1]]],device=device))[0,0]))
        packet={'early':allfield[observer.early],'mid':allfield[observer.mid],'late':allfield[depth],
            'postnorm':post,'incoming_embedding':np.stack(incoming),'positions':np.array(pos),'all_layer_energy':np.mean(unbits(allfield).astype(float)**2,axis=-1).astype(np.float32)}
        path=out/'fields'/f'{r["sample_id"]}.npz';npz(path,**packet)
        m={k:r[k] for k in ('sample_id','source_group','language','genre','split')}|{'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],
            'positions':pos,'exact_endpoint':align,'same_shape_repeat_bitwise':check,'field_sha':sha(path),'all_layer_array_identity':array_identity(allfield),
            'own_anchor_has_later_piece_with_same_char_end':[any(b==enc['offset_mapping'][q][1] and j>q for j,(a,b) in enumerate(enc['offset_mapping'])) for q in (pos[2],pos[4])],
            'alignment_limit':'Q4 preserves exact original native anchor index. Equal char offsets may include overlapping multibyte pieces, so equal endpoint is not proof of identical decoded prefix information across tokenizers.'}
        save(out/'rows'/f'{r["sample_id"]}.json',m)
        group=r['split'];ff=unbits(allfield).astype(float)
        if group not in strata:strata[group]={'count':0,'sum':np.zeros((depth+1,6,width)), 'square':np.zeros((depth+1,6,width))}
        strata[group]['sum']+=ff;strata[group]['square']+=ff*ff;strata[group]['count']+=1
        return packet,m
    def pack(collected):
        raw={k:[] for k in ('early','late','new_embedding','next_late')};meta=[]
        for z,m in collected:
            for j,q in enumerate((2,4)):
                for name,v in [('early',z['early'][q]),('late',z['late'][q]),('next_late',z['late'][q+1]),('new_embedding',z['incoming_embedding'][j])]:raw[name].append(unbits(v))
                meta.append({**m,'anchor':j})
        return {k:np.stack(v) for k,v in raw.items()},meta
    def logprob(h,post=False):
        x=torch.as_tensor(np.asarray(h),dtype=torch.bfloat16,device=device)
        norm=x if post else model.model.norm(x)
        return model.lm_head(norm).float().log_softmax(-1)
    try:
      with torch.inference_mode():
        for i,r in enumerate(mainrows):
            captured.append(collect(r))
            if i<2 or (i+1)%16==0:print('JOINT_SCALE_MAIN',key,i+1,160,'seconds',round(time.monotonic()-start,1),flush=True)
            assert time.monotonic()-start<5400
        raw,meta=pack(captured);tr=np.array([i for i,m in enumerate(meta) if m['split']=='train']);va=np.array([i for i,m in enumerate(meta) if m['split']=='validation'])
        scales={k:float(np.mean(np.sum(raw[k][tr].astype(float)**2,1))) for k in ('early','late','new_embedding')};dd=dots(raw,raw,scales)
        for name in ('current_linear','current_quadratic','temporal_linear','temporal_bilinear'):
            kd,kind=kernel(dd,name);target=raw['next_late'] if name.startswith('temporal') else raw['late']
            models[name],choices[name],grid=select(kd,kind,target,tr,va,[(0,width)])
            npz(out/'models'/f'{name}.npz',**models[name]);save(out/'models'/f'{name}.json',{'choice':choices[name],'grid':grid})
        chosen={s:min([n for n in choices if n.startswith(s)],key=lambda n:choices[n]['validation_normalized_MSE']) for s in ('current','temporal')}
        x=np.stack([unbits(z['early'][0]) for z,m in captured if m['split']=='train']).astype(float)
        y=np.stack([unbits(z['mid'][0]) for z,m in captured if m['split']=='train']).astype(float)
        mx,my=x.mean(0),y.mean(0);variance=((x-mx)**2).mean(0);cov=((x-mx)*(y-my)).mean(0)
        common=(my-mx).astype(np.float32);aa=(cov/(1.01*variance+1e-8)).astype(np.float32);bb=(my-aa*mx).astype(np.float32)
        npz(out/'first_boundary_fit.npz',common=common,a=aa,b=bb)
        immutable(out/'frozen.json',{'timestamp':stamp(),'selected':chosen,'choices':choices,'scales':scales,'train_indices':tr.tolist(),
            'model_sha':{n:sha(out/'models'/f'{n}.npz') for n in choices},'boundary_fit_sha':sha(out/'first_boundary_fit.npz'),
            'fresh_model_outputs_seen':False,'fresh_ids':[r['sample_id'] for r in freshrows]})
        fresh=[]
        for i,r in enumerate(freshrows):
            value=collect(r);captured.append(value);fresh.append(value)
            if (i+1)%16==0:print('JOINT_SCALE_FRESH',key,i+1,64,'seconds',round(time.monotonic()-start,1),flush=True)
            assert time.monotonic()-start<5400
        new,fr=pack(fresh);cross=dots(new,{k:v[tr] for k,v in raw.items()},scales);reports={}
        for name in models:
            kd,kind=kernel(cross,name);p=predict(models[name],Bank.gram(kd,kind,choices[name]['mix']))
            target=new['next_late'] if name.startswith('temporal') else new['late'];mse=np.mean((p.astype(float)-target)**2,1);per=[]
            for begin in range(0,len(p),8):
                lq,lp=logprob(target[begin:begin+8]),logprob(p[begin:begin+8]);kl=(lq.exp()*(lq-lp)).sum(-1);agree=lq.argmax(-1)==lp.argmax(-1)
                for j,i in enumerate(range(begin,min(begin+8,len(p)))):per.append({'sample_id':fr[i]['sample_id'],'source_group':fr[i]['source_group'],'language':fr[i]['language'],'anchor':fr[i]['anchor'],'MSE':float(mse[i]),'KL':float(kl[j]),'agreement':bool(agree[j])})
            reports[name]={'MSE':paired_summary(mse,[m['source_group'] for m in fr]),'anchor_MSE':float(mse.mean()),
                'KL':paired_summary([r['KL'] for r in per],[r['source_group'] for r in per]),'argmax_agreement':float(np.mean([r['agreement'] for r in per]))}
            compressed_json(out/f'{name}_fresh_rows.json.gz',per);npz(out/f'{name}_full_coordinate_MSE.npz',value=np.mean((p.astype(float)-target)**2,0).astype(np.float32))
            if name in chosen.values():npz(out/f'{name}_fresh_prediction.npz',prediction=p)
            print('JOINT_SCALE_RESULT',key,name,reports[name]['anchor_MSE'],reports[name]['KL']['mean'],flush=True)
        boundary=[];floor=[]
        for z,m in fresh:
            xx,yy=unbits(z['early'][0]),unbits(z['mid'][0]);pp={'identity':xx,'common':xx+common,'affine':xx*aa+bb}
            boundary.append({**{k:m[k] for k in ('sample_id','source_group','language')},'methods':{n:float(np.mean((v.astype(float)-yy)**2)) for n,v in pp.items()}})
            lq=logprob(unbits(z['postnorm'][[2,4]]),True);lp=logprob(unbits(z['late'][[2,4]]))
            floor.extend((lq.exp()*(lq-lp)).sum(-1).cpu().tolist())
        for s,v in strata.items():npz(out/f'all_layer_{s}_full_coordinate_moments.npz',mean=(v['sum']/v['count']).astype(np.float32),second_moment=(v['square']/v['count']).astype(np.float32),count=np.array(v['count']))
        save(out/'boundary_fresh.json',{'rows':boundary,'summary':{n:paired_summary([r['methods'][n] for r in boundary],[r['source_group'] for r in boundary]) for n in ('identity','common','affine')}})
        result={'timestamp':stamp(),'model':key,'model_path':str(ROOT/'models/hf'/MODELS[key]),'depth':depth,'width':width,'early':observer.early,'mid':observer.mid,
            'selected_before_own_fresh':chosen,'reports':reports,'boundary':read(out/'boundary_fresh.json')['summary'],
            'raw_to_saved_postnorm_BF16_head_floor_KL':float(np.mean(floor)),
            'exact_char_endpoint_fraction':float(np.mean([a for z,m in captured for a in m['exact_endpoint']])),
            'sources':len(captured),'train_validation_fresh':[128,32,64],'dtype':str(model.dtype),'quantized':False,
            'scope':'Own native coordinates, own fit. Within-model results and selected layer ratios are not coordinate alignment, universal model-size law or complete reasoning/knowledge/syntax replication.',
            'retention':'Every selected-source early/mid/final/postnorm6 positions plus all-layer full-coordinate moments and per-source full-field array identity.'}
        save(out/'result.json',result)
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    ledger('joint_scale_'+key,time.monotonic()-start,sources=224);guard();print('JOINT_SCALE_COMPLETE',key,usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):main(a.model)
