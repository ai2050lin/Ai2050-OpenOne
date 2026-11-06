"""Native prequestion writer capture and independent heldout pathway attribution."""
import argparse
import json
import shutil
import time

import numpy as np

import mlp_functional_common as C
from mlp_fidelity_v2_collect_v4 import first_word, greedy
from server.mechanism_trace_service import parameter
from shared.dist_protocol.mechanism_trace import GpuLease, unbits
from shared.dist_protocol.natural_register_runner import snapshot


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert snapshot(C.MODEL)==C.identity()['model_snapshot']
    model=AutoModelForCausalLM.from_pretrained(C.MODEL,local_files_only=True,dtype=torch.bfloat16,
                                            device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);tok.padding_side='left'
    eos=model.generation_config.eos_token_id
    return model,tok,{eos} if isinstance(eos,int) else set(eos)


def bit_array(value):
    import torch
    return value.detach().contiguous().view(torch.uint16).cpu().numpy()


def cells(rows):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ['split','frame','r','t','q'])
        groups.setdefault(key,[]).append(row)
    return [dict(zip(['split','frame','r','t','q'],key)) | {'n':len(group),'correct':sum(r['correct'] for r in group)}
            for key,group in sorted(groups.items())]


def collect(part):
    import torch
    torch.set_num_threads(4)
    corpus=C.read(C.OUT/'controlled-inputs.json');pr=C.read(C.OUT/'controlled-prereg.json')
    assert C.digest(corpus)==pr['corpus_sha256']
    result=C.OUT/f'{part}-result.json'
    if result.exists():return
    if part=='trace':assert C.read(C.OUT/'pilot-result.json')['passed']
    rows=corpus['rows'];total=32 if part=='pilot' else len(rows)//8
    contract={**C.identity(),'schema':'functional_upstream_execution.v1','part':part,'backgrounds':total,
              'source_sha256':C.file_sha(__file__),'corpus_sha256':C.digest(corpus),'generation_tokens':4,
              'prequestion_native_arrays':'Q0 only, exact full prefix checked on Q1. Native bits, all blocks and original order.'}
    C.save(C.OUT/f'{part}-execution-prereg.json',contract)
    shapes={'head':(total*4,36,32,128),'mlp':(total*4,35,2560),'residual':(total*4,37,2560)} if part=='trace' else {}
    folder=C.OUT/part;folder.mkdir(exist_ok=True)
    state=folder/'resume.json';done=C.read(state)['backgrounds'] if state.exists() else 0
    record_file=folder/'outcomes.jsonl'
    responses=[json.loads(x) for x in record_file.read_text(encoding='utf-8').splitlines()[:done*8]] if done else []
    needed=sum(np.prod(s)*2 for k,s in shapes.items() if not (folder/f'{k}.npy').exists())
    assert shutil.disk_usage(folder).free>needed+2*2**30,'insufficient disk for native upstream fields'
    arrays={k:np.lib.format.open_memmap(folder/f'{k}.npy',mode='r+' if done else 'w+',dtype=np.uint16,shape=s) for k,s in shapes.items()}
    start=time.time();context={};checks=[]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        handles=[]
        if arrays:
            def take(x):return x[:,context['position']].detach().clone()
            def o_pre(_,args,b):
                if context.get('capture'):context['head'][b]=take(args[0]).reshape(4,32,128)
            def o_post(_,args,value,b):
                if context.get('capture'):context['attention'][b]=take(value)
            def y_post(_,args,value,b):
                if context.get('capture'):context['mlp'][b]=take(value)
            def r_pre(_,args):
                if context.get('capture'):context['r']=take(args[0])
            for b,l in enumerate(model.model.layers):
                handles.extend([l.self_attn.o_proj.register_forward_pre_hook(lambda m,a,b=b:o_pre(m,a,b)),
                                l.self_attn.o_proj.register_forward_hook(lambda m,a,o,b=b:o_post(m,a,o,b))])
                if b<35:handles.append(l.mlp.register_forward_hook(lambda m,a,o,b=b:y_post(m,a,o,b)))
            handles.append(model.model.layers[35].post_attention_layernorm.register_forward_pre_hook(r_pre))

        for b in range(done,total):
            batch_outputs=[];prefirst=None;record0=[]
            for q in [0,1]:
                batch=rows[b*8+q:(b+1)*8:2]
                position=batch[0]['prequestion'];before=batch[0]['before_selector']
                assert all(r['prequestion']==position and r['before_selector']==before for r in batch)
                ids=torch.tensor([r['tokens'] for r in batch],device='cuda')
                context={'capture':bool(arrays and q==0),'position':position,'head':{},'attention':{},'mlp':{}}
                out=model(input_ids=ids,use_cache=True,output_hidden_states=bool(arrays),logits_to_keep=1)
                context['capture']=False
                if arrays:
                    native=torch.stack([h[:,position] for h in out.hidden_states[:36]],dim=1)
                    presel=torch.stack([h[:,before] for h in out.hidden_states[:36]],dim=1)
                    assert torch.equal(presel[::2],presel[1::2]),('selector leaked into causal prefix',b,q)
                    if q==0:
                        prefirst=native
                        for layer in range(35):
                            written=(native[:,layer]+context['attention'][layer])+context['mlp'][layer]
                            assert torch.equal(written,native[:,layer+1]),('native additive residual mismatch',b,layer)
                        assert torch.equal(native[:,35]+context['attention'][35],context['r'])
                        sl=slice(b*4,(b+1)*4)
                        arrays['head'][sl]=bit_array(torch.stack(list(context['head'].values()),1))
                        arrays['mlp'][sl]=bit_array(torch.stack(list(context['mlp'].values()),1))
                        arrays['residual'][sl]=bit_array(torch.cat([native,context['r'][:,None]],1))
                    else:
                        assert torch.equal(prefirst,native),('question leaked into causal prefix',b)
                logits=out.logits[:,-1]
                if b==done and q==0 and arrays:
                    clean=model(input_ids=ids,use_cache=True,logits_to_keep=1).logits[:,-1]
                    assert torch.equal(clean,logits), 'observer canary failed'
                    checks.append({'native_observers_logit_equal':True})
                generated=greedy(model,out.past_key_values,logits,eos,limit=contract['generation_tokens'])
                for item,ids_out in zip(batch,generated):
                    text=tok.decode(ids_out,skip_special_tokens=True);word=first_word(text)
                    batch_outputs.append({k:v for k,v in item.items() if k!='tokens'}|{'generated':text,'generated_ids':ids_out,'word':word,'correct':word==item['expected']})
                del out
            responses.extend(batch_outputs)
            if (b+1)%16==0 or b+1==total:
                for a in arrays.values():a.flush()
                with record_file.open('w',encoding='utf-8') as f:
                    f.writelines(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in responses)
                C.save(state,{'backgrounds':b+1},False)
                print(part,b+1,'/',total,'seconds',round(time.time()-start,1),flush=True)
        for h in handles:h.remove()
        assert snapshot(C.MODEL)==contract['model_snapshot']
    by_cell=cells(responses)
    if part=='pilot':
        assert len(by_cell)==16 and all(r['n']==16 for r in by_cell)
        passed=all(r['correct']>=14 for r in by_cell)
    else:
        held=[r for r in by_cell if r['split']!='discovery']
        assert len(held)==32 and all(r['n']==80 for r in held)
        passed=all(r['correct']/r['n']>=.90 for r in held)
    native=[{'path':f'{part}/{k}.npy','shape':list(s),'dtype':'BF16 encoded uint16','sha256':C.file_sha(folder/f'{k}.npy')} for k,s in shapes.items()]
    C.save(result,{**contract,'status':'complete','passed':passed,'cells':by_cell,'rows':responses,'native':native,'checks':checks,
                   'native_identity':'bitwise for every Q0 case: R_(b+1) = BF16(BF16(R_b + attention_b) + MLP_b); incoming R35 = BF16(R35_block_input + attention35)' if arrays else None,
                   'prequestion_and_preselector_controls_bitwise':bool(arrays),'seconds':time.time()-start})
    print(part,'behavior gate',passed,'errors',sum(not r['correct'] for r in responses),flush=True)


def contrasts(x,attribute):
    # Native order [R,T]. Industry polarity reverses when R=1.
    d=x[:,1::2]-x[:,::2]
    if attribute=='industry':d[:,1]*=-1
    return d


def analyze():
    from scipy.stats import t as student_t
    import torch
    torch.set_num_threads(6)
    trace=C.read(C.OUT/'trace-result.json');pr=C.read(C.OUT/'controlled-prereg.json')
    folder=C.OUT/'trace';n=trace['backgrounds']
    by_name={}
    for meta in trace['native']:
        path=C.OUT/meta['path'];assert C.file_sha(path)==meta['sha256']
        by_name[path.stem]=np.load(path,mmap_mode='r')
    limits={'discovery':slice(0,128),'confirm':slice(128,288),'lockbox':slice(288,n)}
    hs=by_name['residual'];incoming=unbits(np.asarray(hs[:,36])).reshape(n,4,2560)
    directions={};observations={};rows=[];projected=np.empty((n,4,2,36,33),np.float32);projected[:,:,:,35,32]=0
    d_all=[];split_scores={};energies=np.zeros((36,33),np.float64)
    for attr in ['location','industry']:
        d=contrasts(incoming[limits['discovery']],attr).mean(axis=(0,1));d/=np.linalg.norm(d)
        score=incoming@d;threshold=float(score[limits['discovery']].mean());d_all.append(d)
        directions[attr]={'direction':d.tolist(),'threshold':threshold}
        labels=np.asarray([0,1,0,1] if attr=='location' else [0,1,1,0],bool)
        split_scores[attr]={split:{'classification_accuracy':float(np.mean((score[ix]>threshold)==labels)),
                            'paired_accuracy_by_assignment':(contrasts(score[ix],attr)>0).mean(0).tolist()}
                            for split,ix in limits.items()}
    D=np.stack(d_all)
    for b in range(36):
        weight=parameter(C.MODEL,f'model.layers.{b}.self_attn.o_proj.weight',(slice(None),slice(None)))
        w=torch.from_numpy(weight).reshape(2560,32,128).permute(1,2,0).contiguous()
        z=unbits(np.asarray(by_name['head'][:,b])).reshape(n,4,32,128)
        coefficient=(torch.from_numpy(D)@torch.from_numpy(weight)).reshape(2,32,128).numpy()
        projected[:,:,:,b,:32]=np.einsum('nqhd,ahd->nqah',z,coefficient)
        # Exact Euclidean output RMS over discovery, using the real output projection for each query head.
        zz=torch.from_numpy(z[:128].reshape(-1,32,128)).permute(1,0,2)
        for lo in range(0,32,8):
            out=zz[lo:lo+8]@w[lo:lo+8]
            energies[b,lo:lo+8]=out.square().sum(-1).mean(1).numpy()
        if b<35:
            y=unbits(np.asarray(by_name['mlp'][:,b])).reshape(n,4,2560)
            projected[:,:,:,b,32]=y@D.T
            energies[b,32]=np.mean(np.sum(y[:128]**2,axis=-1))
        if (b+1)%6==0:print('projected through block',b,flush=True)

    selected={};null_rng=np.random.default_rng(C.SEED)
    for index,attr in enumerate(['location','industry']):
        proj=projected[:,:,index];d=D[index]
        delta=contrasts(proj,attr);dis=delta[limits['discovery']].mean((0,1));dis[35,32]=-np.inf
        heads=np.argwhere(dis[:,:32]>0)
        heads=sorted((tuple(map(int,p)) for p in heads),key=lambda p:(-dis[p],p))[:2]
        mlps=sorted((b for b in range(35) if dis[b,32]>0),key=lambda b:(-dis[b,32],b))[:2]
        picked=heads+[(b,32) for b in mlps]
        picks=[]
        for b,j in picked:
            evidence={}
            for split in ['confirm','lockbox']:
                values=delta[limits[split],:,b,j];nn=len(values)
                mean=values.mean(0);hw=student_t.ppf(1-.05/(2*2*4*2*2),nn-1)*values.std(0,ddof=1)/np.sqrt(nn)
                evidence[split]={'mean_by_R':mean.tolist(),'CI_by_R':np.stack([mean-hw,mean+hw],-1).tolist(),
                                 'positive_fraction_by_R':(values>0).mean(0).tolist(),'passed':bool(np.all(mean-hw>0))}
            picks.append({'block':b,'head':j if j<32 else None,'kind':'attention' if j<32 else 'mlp',
                          'discovery_mean':float(dis[b,j]),'discovery_output_rms':float(np.sqrt(energies[b,j])),
                          'heldouts':evidence,'replicated':trace['passed'] and all(v['passed'] for v in evidence.values())})
        selected[attr]=picks
        projection_sum=proj.sum((2,3));embed=unbits(np.asarray(hs[:,0])).reshape(n,4,2560)@d
        actual=incoming@d
        error=actual-embed-projection_sum
        for split,ix in limits.items():
            vectors=contrasts(actual[ix],attr).mean(1)
            signs=null_rng.choice([-1.,1.],size=(4000,len(vectors)))
            null=signs@vectors/len(vectors)
            split_scores[attr][split]['block_sign_permutation_p']=float((1+np.sum(null>=vectors.mean()))/(1+len(null)))
        observations[attr]={'incoming':split_scores[attr],'embedding_pair_effect_max_abs':float(np.abs(contrasts(embed,attr)).max()),
                            'projection_residual_rmse':float(np.sqrt(np.mean(error**2))),
                            'incoming_projection_rms':float(np.sqrt(np.mean(actual**2)))}
        for b in range(36):
            for j in range(33 if b<35 else 32):
                rows.append({'attribute':attr,'block':b,'head':j if j<32 else None,'kind':'attention' if j<32 else 'mlp',
                             'discovery_output_rms':float(np.sqrt(energies[b,j])),
                             **{split:delta[ix,:,b,j].mean(0).tolist() for split,ix in limits.items()}})
    primary=selected['location'];head_control=[];mlp_control=[];used={(p['block'],p['head']) for p in primary}
    for p in primary:
        b=p['block'];j=p['head'];rms=p['discovery_output_rms']
        if j is not None:
            pool=[(b,k) for k in range(32) if (b,k) not in used and .75*rms<=np.sqrt(energies[b,k])<=1.25*rms]
        else:
            pool=[(k,None) for k in range(max(0,b-5),min(35,b+6)) if (k,None) not in used and .75*rms<=np.sqrt(energies[k,32])<=1.25*rms]
        if not pool:continue
        chosen=pool[int(null_rng.integers(len(pool)))];used.add(chosen)
        r={'block':chosen[0],'head':chosen[1],'rms':float(np.sqrt(energies[chosen[0],32 if chosen[1] is None else chosen[1]]))}
        (head_control if j is not None else mlp_control).append(r)
    native=C.OUT/'writer-projections.npz'
    np.savez_compressed(native,projected=projected,energies=energies,directions=D)
    C.save(C.OUT/'path-analysis.json',{**C.identity(),'schema':'functional_upstream_paths.v1','source_sha256':C.file_sha(__file__),
          'trace_sha256':C.file_sha(C.OUT/'trace-result.json'),'behavior_gate':trace['passed'],'directions':directions,
          'selected':selected,'observations':observations,'all_writer_effects':rows,'projection_sha256':C.file_sha(native),
          'random':head_control+mlp_control,'random_matched':len(head_control)+len(mlp_control)==len(primary),
          'path_equation':'Head: z_(b,h) dot (O_(b,h)^T d); MLP: y_b dot d. FP32 split sum differs from native BF16 additions, quantified per attribute.',
          'scope':'query-head writes at prequestion token, no per-source K/V attribution; heldout coding and intervention are separate evidence.'})
    for name in selected:print(name,[(p['block'],p['head'],p['replicated']) for p in selected[name]],observations[name],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['pilot','trace','analyze'])
    args=parser.parse_args()
    analyze() if args.part=='analyze' else collect(args.part)
