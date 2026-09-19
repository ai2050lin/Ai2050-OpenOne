"""Frozen native all-coordinate operation atlas; no donor or weight intervention.

Durability unit: one language-family cell (512 conditions). Completed cell files
are hash-checked and skipped. A partial cell is re-observed, but its durable
natural outputs are reused, never selected by success. Only predefined examples
persist full token fields; every case contributes every coordinate to maps.
"""
import argparse, gc, itertools, shutil, time
from collections import defaultdict
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2671_native_mlp_field import unbits, bits
from phase2677_padded_native_runtime import PaddedCapture, padded_inputs, native_pack
from phase2677_source_role_material import evaluate
from phase2679_native_source_ledger import real_token_data, pack_case, exact_bits
from phase2685_native_qkv_capture import NativeQKVCapture
from phase2685_native_attention_contract import LAYERS
from phase2683_explicit_answer_audit import extract, audit_records
from phase2662_symmetric_mapping_contract import load_native

OUT=RESULT/'phase2687_role_qkv_field'
CONTRACT=RESULT/'phase2686_independent_role_contract'
ATT=RESULT/'phase2685_native_attention_contract'
MLP_LAYERS=(23,26,27,28)
OPERATIONS=('roster_order','mention_order','target_index','form')
FUNCTIONS=('truth','mapped_truth','name','cloze')
FACTORS=('form','roster_order','mention_order','target_index','output_function')
SHAPES={'h':(37,2,2560),'a':(36,2,9728),'attention_x':(8,2,2560),
        'linear_q':(8,2,32,128),'linear_k':(8,2,8,128),'linear_v':(8,2,8,128),
        'normalized_q':(8,2,32,128),'normalized_k':(8,2,8,128),
        'query_rope':(8,2,32,128),'key_rope':(8,2,8,128),'probability':(8,2,32,256)}
FLOOR=8*1024**3


def source_arrays(source,up):
    a=pack_case(source)
    for l,data in up.items():
        for key,value in data.items():a[f'L{l}__upstream_{key}']=exact_bits(value)
    return a


def budget_for(rows):
    D,K=2560,9728; boundary=2*(37*2*D+36*2*K)
    native=sum(boundary+2*len(r['prompt_ids'])*37*D for r in rows if r['published'])
    native+=sum(2*len(r['prompt_ids'])*len(MLP_LAYERS)*(3*K+4*D) for r in rows if r['parameter_published'])
    source=0
    for r in rows:
        if r['parameter_published']:
            n=len(r['prompt_ids'])
            # All source factors, nine upstream arrays, exact BF16 and FP64 mask.
            source+=len(LAYERS)*(2*(2*(5*D+3*K+2*4096)+n*(2*1024+64+2560+6144+5120+4096+256))+16*n+8)
    coords=sum(int(np.prod(s)) for s in SHAPES.values())
    # pos/neg and four-function concordant pos/neg uint16, sums/abs FP64.
    maps=16*4*coords*(4*2+2*8)+16*4*int(np.prod(SHAPES['probability']))*4
    moments=16*16*(37*D+36*(3*K+2*D)+8*(2560+6144+5120+4096+256+1024))
    return {'published_native':native,'published_QKV':source,'allcoordinate_operation_maps':maps,
            'all_actual_token_coordinate_moments':moments,'headers_records_reserve':256*1024**2,
            'total':native+source+maps+moments+256*1024**2}


def prepare():
    assert read(CONTRACT/'analysis/final.json')['all_checks_passed']
    c=read(CONTRACT/'protocol/frozen.json');rows=read(CONTRACT/'material/initial.json')
    for split in ('initial','confirmation'):
        assert sha(CONTRACT/f'material/{split}.json')==c['material'][split]['case_sha256']
    assert sha(TESTS/'phase2683_explicit_answer_audit.py')==c['scoring']['parser_sha256']
    a=read(ATT/'protocol/frozen.json')
    assert sha(TESTS/'phase2685_native_qkv_capture.py')==a['capturer_sha256']
    assert sha(TESTS/'phase2685_native_attention_math.py')==a['math_sha256']
    b=budget_for(rows);confirmation=budget_for(read(CONTRACT/'material/confirmation.json'))
    free=shutil.disk_usage(RESULT).free
    contract={'cases':8192,'base_instances':128,'conditions_per_base':64,'groups':16,'cases_per_group':512,
        'material_sha256':c['material']['initial']['case_sha256'],'field_total':256,'natural_cache_token_budget':32,
        'noop_cases':64,'qkv_layers':LAYERS,'fulltoken_MLP_layers':MLP_LAYERS,'shapes':SHAPES,
        'source_raw_cases':16,'allH_raw_cases':64,'fullMLP_raw_cases':16,
        'storage_budget':{'free_before':free,'initial_uncompressed_upper':b,'confirmation_uncompressed_upper':confirmation,
            'later_scalar_crossmodel_reserve':2*1024**3,'floor':FLOOR,
            'required':b['total']+confirmation['total']+2*1024**3+FLOOR},
        'observation':'Every actual token/all coordinates of H/x/gate/up/a/down and QKV/norm/RoPE processed. Token sums/sumsq retain physical coordinates, NOT arbitrary token correlations. Two real queries, allheads and allreal sources; no all-query attention matrix claim.',
        'operation_maps':'For4frozenedges: positive/negative counts, signed/absolute sums on everycoordinate; zero=count-pos-neg. Additional all4function strictlysame-sign counts, not a semantic-closure gate. Retain everyfamily/language background. No TopK or donor.',
        'source_position_caveat':'P maps align absolute source index, not lexical/semantic tokens. Compare only shared actualsource positions; virtual tail excluded. Future actualtoken probabilities zero are retained. Source raw exact16predeclaredtruth examples permits detailed token inspection.',
        'precision':'Native nonquantized BF16. FP64 analysis accumulators only. Fixed256padded field and natural cache generation are two numerical protocols, never one exact native trajectory.',
        'durability':'16 cell commits with hashes. Partialcell observations replay; durable case behavior reused. No success-based material selection. Published fields must reproduce exact before reuse.',
        'denominator_correction':'2686 text256base*32wasarithmeticerror. Actual128base*64=8192; all8192prompts/4096operationedges unchanged. Append correction next completedphase, preserve old MEMO.',
        'parser_sha256':c['scoring']['parser_sha256'],
        'code_sha256':{n:sha(TESTS/n) for n in ('phase2687_role_qkv_field.py','phase2677_padded_native_runtime.py','phase2685_native_qkv_capture.py')}}
    path=OUT/'protocol/frozen.json'
    if path.exists():
        old=read(path)
        for k in ('material_sha256','code_sha256','parser_sha256'):assert old[k]==contract[k],f'Frozen code/material changed: {k}'
        return rows,old
    assert free>contract['storage_budget']['required'],contract['storage_budget']
    save(path,contract)
    print('2687 STORAGE',contract['storage_budget'],flush=True)
    return rows,contract


def boundaries(pack,source,up,row):
    qq=[row['body_end_token'],row['task_end_token']];n=len(row['prompt_ids'])
    out={'h':pack['h'],'a':pack['a']}
    for key in ('attention_x','linear_q','linear_k','linear_v','normalized_q','normalized_k'):
        vals=np.stack([up[l][key][qq] for l in LAYERS]).reshape(SHAPES[key]);out[key]=exact_bits(vals)
    out['query_rope']=exact_bits(np.stack([up[l]['query_post_rope_full'][qq] for l in LAYERS]))
    out['key_rope']=exact_bits(np.stack([source[l]['actual_key_post_rope'][qq] for l in LAYERS]))
    p=np.zeros(SHAPES['probability'],np.float64)
    for li,l in enumerate(LAYERS):p[li,:,:,:n]=source[l]['actual_probability']
    out['probability']=exact_bits(p)
    for k,v in out.items():assert v.shape==SHAPES[k] and v.dtype==np.uint16
    return out


def qkv_moments(up,source):
    result={}
    for k in next(iter(up.values())):
        vals=[up[l][k] for l in LAYERS]
        for suffix,fn in (('sum',lambda x:x.sum(0)),('sumsq',lambda x:(x*x).sum(0))):
            result['qkv__'+k+'__'+suffix]=np.stack([fn(x) for x in vals])
    for suffix,fn in (('sum',lambda x:x.sum(0)),('sumsq',lambda x:(x*x).sum(0))):
        result['qkv__key_rope__'+suffix]=np.stack([fn(source[l]['actual_key_post_rope']) for l in LAYERS])
    return result


def add_arrays(dst,src):
    for k,v in src.items():
        assert np.isfinite(v).all(),k
        if k not in dst:dst[k]=np.zeros_like(v)
        dst[k]+=v


def init_maps():
    out={}
    for axis in OPERATIONS:
        for key,shape in SHAPES.items():
            for name,dtype in (('positive',np.uint16),('negative',np.uint16),('sum',np.float64),('sumabs',np.float64),('all4_positive',np.uint16),('all4_negative',np.uint16)):
                out[f'{axis}__{key}__{name}']=np.zeros(shape,dtype)
        for name in ('valid_count','all4_valid_count'):
            out[f'{axis}__probability__{name}']=np.zeros(SHAPES['probability'],np.uint16)
    return out


def operation_maps(rows,data,maps):
    assert len(rows)==len(data)==64
    for axis in OPERATIONS:
        grouped=defaultdict(list)
        for r in rows:grouped[tuple(r[k] for k in FACTORS if k!=axis)].append(r)
        assert len(grouped)==32
        four={}
        for rr in grouped.values():
            aa,bb=sorted(rr,key=lambda r:r[axis]);a=data[aa['case_index']];b=data[bb['case_index']]
            base=tuple(aa[k] for k in FACTORS if k not in (axis,'output_function'))
            if base not in four:four[base]={'positive':{},'negative':{},'valid_n':256,'functions':set()}
            f=four[base];f['functions'].add(aa['output_function'])
            n=min(len(aa['prompt_ids']),len(bb['prompt_ids']));f['valid_n']=min(f['valid_n'],n)
            for key in SHAPES:
                d=unbits(b[key]).astype(np.float64)-unbits(a[key]).astype(np.float64)
                if key=='probability':
                    d[...,n:]=0;maps[f'{axis}__{key}__valid_count'][...,:n]+=1
                pos=d>0;neg=d<0;stem=f'{axis}__{key}__'
                maps[stem+'positive']+=pos;maps[stem+'negative']+=neg
                maps[stem+'sum']+=d;maps[stem+'sumabs']+=np.abs(d)
                if key not in f['positive']:f['positive'][key]=pos.copy();f['negative'][key]=neg.copy()
                else:f['positive'][key]&=pos;f['negative'][key]&=neg
        assert len(four)==8 and all(v['functions']==set(FUNCTIONS) for v in four.values())
        for v in four.values():
            for key in SHAPES:
                maps[f'{axis}__{key}__all4_positive']+=v['positive'][key]
                maps[f'{axis}__{key}__all4_negative']+=v['negative'][key]
            maps[f'{axis}__probability__all4_valid_count'][...,:v['valid_n']]+=1


def check_same_body(rows,data):
    g=defaultdict(list)
    for r in rows:g[tuple(r[k] for k in FACTORS if k!='output_function')].append(r)
    checks=0
    for rr in g.values():
        assert len(rr)==4 and len({tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in rr})==1
        for key in ('h','a','attention_x','linear_q','linear_k','linear_v','normalized_q','normalized_k','query_rope','key_rope'):
            a=data[rr[0]['case_index']][key][:,0]
            assert all(np.array_equal(a,data[r['case_index']][key][:,0]) for r in rr[1:]),(rr[0]['case_id'],key)
        checks+=1
    return checks


def exact_save(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as old:assert set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items()),f'Published replay changed: {path}'
    else:np.savez_compressed(path,**arrays)


@torch.inference_mode()
def qualify(model,tok,rows):
    out=[]
    for r in [r for r in rows if r['published']]:
        inp=padded_inputs(model,r['prompt_ids'],tok.eos_token_id,total=256)
        base=model.model(**inp).last_hidden_state.detach().cpu()
        cap=PaddedCapture(model,MLP_LAYERS)
        try:
            with NativeQKVCapture(model,LAYERS) as qc:
                cap.reset(r['body_end_token'],True,r['task_end_token']);qc.reset(r['body_end_token'],r['task_end_token'])
                cap.enabled=qc.enabled=True
                obs=model.model(**inp).last_hidden_state.detach().cpu();cap.enabled=qc.enabled=False
                pp=cap.pack();ss=real_token_data(qc.pack(),len(r['prompt_ids']));uu=qc.upstream_pack()
                bb=boundaries(pp,ss,uu,r)
                for li,l in enumerate(LAYERS):
                    assert np.array_equal(unbits(bb['h'][l]),ss[l]['residual_before_attention'])
                    assert np.array_equal(unbits(bb['a'][l]),ss[l]['mlp_a'])
                assert pp['full__h'].shape==(37,len(r['prompt_ids']),2560)
        finally:cap.close()
        restored=model.model(**inp).last_hidden_state.detach().cpu()
        assert torch.equal(base,obs) and torch.equal(base,restored)
        out.append({'case_id':r['case_id'],'baseline_observed_restored_exact':True,'allH_MLP_QKV_bridge':True})
        if len(out)%8==0:print('2687 COMBINED256 NOOP',len(out),64,flush=True)
    assert len(out)==64
    save(OUT/'analysis/native256_preflight.json',{'all_checks_passed':True,'records':out,'source_sha256':sha(Path(__file__))})


@torch.inference_mode()
def natural(model,tok,row,padded_state):
    device=model.get_input_embeddings().weight.device
    ids=torch.tensor([row['prompt_ids']],device=device)
    output=model.model(input_ids=ids,use_cache=True)
    state=output.last_hidden_state[0,-1]
    diff=state.float().cpu()-padded_state.float().cpu()
    logits=model.lm_head(state).float();cache=output.past_key_values
    logp=torch.log_softmax(logits,dim=-1);chosen=int(logits.argmax());native_id=chosen
    probe=[{'word':word,'ids':tok.encode(word,add_special_tokens=False)} for word in row['common_readout_words']]
    for x in probe:
        assert x['ids'];x['first_token_probability']=float(logp[x['ids'][0]].exp())
    generated=[];generation_logprobs=[]
    for step in range(32):
        generated.append(chosen);generation_logprobs.append(float(torch.log_softmax(logits,dim=-1)[chosen]))
        if chosen==tok.eos_token_id or step==31:break
        output=model.model(input_ids=torch.tensor([[chosen]],device=device),past_key_values=cache,use_cache=True)
        cache=output.past_key_values;logits=model.lm_head(output.last_hidden_state[0,-1]).float();chosen=int(logits.argmax())
    text=tok.decode(generated,skip_special_tokens=True)
    record={k:row[k] for k in ('case_index','case_id','family','language','unit','content_instance','form','roster_order','mention_order','target_index','output_function','target','alternate')}
    record.update(generated=text,generated_ids=generated,generated_token_logprobs=generation_logprobs,
        native_id=native_id,eos=tok.eos_token_id in generated,final_answer_available=True,
        padded_natural_final_exact=bool((diff==0).all()),padded_natural_final_max_abs=float(diff.abs().max()),
        candidate_first_token_probabilities=probe,**evaluate(row,text))
    record['explicit_final']=extract(text,(row['target'],row['alternate']),True)
    # Padded field does not supply the starting token of natural generation.
    return record,bits(state)


@torch.inference_mode()
def collect(model,tok,rows,contract):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['family'],r['language'])].append(r)
    assert len(grouped)==16;t0=time.monotonic()
    cap=PaddedCapture(model,MLP_LAYERS)
    try:
        with NativeQKVCapture(model,LAYERS) as qc:
            for (family,lang),rr in grouped.items():
                stem=family+'_'+lang;commit=OUT/f'analysis/cell_{stem}.json'
                if commit.exists():
                    c=read(commit);assert c['all_checks_passed'] and c['cases']==512
                    for file,digest in c['files'].items():assert sha(OUT/file)==digest
                    print('2687 VERIFIED COMMITTED CELL',stem,flush=True);continue
                maps=init_maps();moments={};records=[];tokens=0;prefixchecks=0
                for offset in range(0,512,64):
                    block=rr[offset:offset+64];data={}
                    assert len({(r['unit'],r['content_instance']) for r in block})==1
                    for r in block:
                        if shutil.disk_usage(OUT).free<FLOOR:raise RuntimeError('8GiB disk floor; retain completed work, do not delete unrelated data')
                        n=len(r['prompt_ids']);cap.reset(r['body_end_token'],r['published'],r['task_end_token']);qc.reset(r['body_end_token'],r['task_end_token'])
                        cap.enabled=qc.enabled=True
                        output=model.model(**padded_inputs(model,r['prompt_ids'],tok.eos_token_id,total=256))
                        cap.enabled=qc.enabled=False;padded_state=output.last_hidden_state[0,r['task_end_token']].detach().clone()
                        pp=cap.pack();ss=real_token_data(qc.pack(),n);uu=qc.upstream_pack()
                        for l in LAYERS:
                            assert np.array_equal(unbits(pp['h'][l]),ss[l]['residual_before_attention'])
                            assert np.array_equal(unbits(pp['a'][l]),ss[l]['mlp_a'])
                        data[r['case_index']]=boundaries(pp,ss,uu,r)
                        mm=cap.moment_pack();mm.update(qkv_moments(uu,ss));add_arrays(moments,mm);tokens+=n
                        casepath=OUT/f'behavior/case_{r["case_index"]:04d}.json'
                        if casepath.exists():
                            record=read(casepath);assert record['case_id']==r['case_id'] and record['material_sha256']==contract['material_sha256']
                        else:
                            record,natural_state=natural(model,tok,r,padded_state)
                            record['material_sha256']=contract['material_sha256']
                            if r['published']:exact_save(OUT/f'field/natural_{r["case_index"]:04d}.npz',{'natural_final_state':natural_state})
                            save(casepath,record)
                        if r['published']:exact_save(OUT/f'field/case_{r["case_index"]:04d}.npz',native_pack(pp,True,r['parameter_published']))
                        if r['parameter_published']:exact_save(OUT/f'source/case_{r["case_index"]:04d}.npz',source_arrays(ss,uu))
                        records.append(record);del pp,ss,uu,mm,output,padded_state
                        cap.reset(0,False);qc.reset(0,0)
                    prefixchecks+=check_same_body(block,data);operation_maps(block,data,maps);del data
                    complete=sum(512 for _ in (OUT/'analysis').glob('cell_*.json'))+len(records)
                    save(OUT/'analysis/progress.json',{'stage':'formal','conditions':complete,'total':8192,'current_cell':stem,'cell_cases':len(records),
                        'elapsed_seconds_this_process':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free})
                    print('2687 NATIVEQKV NATURAL',complete,8192,stem,flush=True)
                assert prefixchecks==128 and len(records)==512
                mapfile=OUT/f'maps/operations_{stem}.npz';momentfile=OUT/f'maps/alltoken_{stem}.npz'
                mapfile.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mapfile,**maps);np.savez_compressed(momentfile,**moments)
                recordfile=OUT/f'analysis/records_{stem}.json';save(recordfile,records)
                summary={'cases':512,'actual_tokens':tokens,'samebody_fourfunction_checks':128,'pairs_each_operation':256,
                    'fourfunction_groups_each_operation':64,'behavior':audit_records(rr,records)['groups']}
                save(commit,{'all_checks_passed':True,**summary,'files':{str(p.relative_to(OUT)):sha(p) for p in (mapfile,momentfile,recordfile)}})
                del maps,moments,records;gc.collect()
    finally:cap.close()


def audit(rows,contract):
    records=[];families=[];global_counts={};files=[]
    for family,lang in dict.fromkeys((r['family'],r['language']) for r in rows):
        stem=family+'_'+lang;c=read(OUT/f'analysis/cell_{stem}.json');assert c['all_checks_passed'] and c['cases']==512
        for f,h in c['files'].items():assert sha(OUT/f)==h
        records.extend(read(OUT/f'analysis/records_{stem}.json'));families.append(c)
        with np.load(OUT/f'maps/operations_{stem}.npz') as z:
            assert len(z.files)==len(OPERATIONS)*(6*len(SHAPES)+2)
            for axis in OPERATIONS:
                for key,shape in SHAPES.items():
                    p=z[f'{axis}__{key}__positive'];n=z[f'{axis}__{key}__negative']
                    p4=z[f'{axis}__{key}__all4_positive'];n4=z[f'{axis}__{key}__all4_negative']
                    valid=z[f'{axis}__{key}__valid_count'] if key=='probability' else 256
                    valid4=z[f'{axis}__{key}__all4_valid_count'] if key=='probability' else 64
                    assert p.shape==shape and (p+n<=valid).all() and (p4+n4<=valid4).all()
                    assert (p>=4*p4).all() and (n>=4*n4).all()
                    for suffix in ('sum','sumabs'):
                        a=z[f'{axis}__{key}__{suffix}'];assert a.shape==shape and np.isfinite(a).all()
                    assert (np.abs(z[f'{axis}__{key}__sum'])<=z[f'{axis}__{key}__sumabs']+1e-8).all()
                    if key in ('h','a'):
                        for sign,a in (('positive',p4),('negative',n4)):
                            kk=axis+'__'+key+'__'+sign
                            if kk not in global_counts:global_counts[kk]=np.zeros(shape,np.uint16)
                            global_counts[kk]+=a
        with np.load(OUT/f'maps/alltoken_{stem}.npz') as z:
            for k in z.files:assert np.isfinite(z[k]).all()
    records.sort(key=lambda r:r['case_index']);assert [r['case_index'] for r in records]==list(range(8192))
    explicit=audit_records(rows,records);save(OUT/'analysis/explicit_answer_audit.json',explicit)
    save(OUT/'analysis/records.json',records)
    np.savez_compressed(OUT/'maps/global_fourfunction_counts.npz',**global_counts)
    fullgate={k:np.argwhere(v==1024).tolist() for k,v in global_counts.items()}
    for r in rows:
        for folder,prefix,flag in (('field','case',r['published']),('field','natural',r['published']),('source','case',r['parameter_published'])):
            if not flag:continue
            path=OUT/f'{folder}/{prefix}_{r["case_index"]:04d}.npz';assert path.exists()
            with np.load(path) as z:
                if folder=='field' and prefix=='case':
                    assert z['full__h'].shape==(37,len(r['prompt_ids']),2560)
                    if r['parameter_published']:assert z['full__a'].shape==(4,len(r['prompt_ids']),9728)
                for k in z.files:
                    if k.endswith('actual_mask'):continue
                    assert np.isfinite(unbits(z[k]) if z[k].dtype==np.uint16 else z[k]).all()
            files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'case_index':r['case_index']})
    save(OUT/'analysis/published_manifest.json',files)
    summary={'conditions':8192,'base_instances':128,'conditions_per_base':64,'allactual_tokens':sum(c['actual_tokens'] for c in families),
        'operation_pairs_each':4096,'same_body_fourfunction_checks':2048,'global_fourfunction_groups_each_operation':1024,
        'global_strict_direction_addresses_H_checkpoint_or_MLP_layer_query_coordinate':fullgate,
        'behavior':explicit['groups'],'padded_natural_state_different':sum(not r['padded_natural_final_exact'] for r in records),
        'padded_natural_max_abs':max(r['padded_natural_final_max_abs'] for r in records),'published_files':len(files)}
    checks={'8192actual_native_and_natural':True,'16completecells_hashed':len(families)==16,'64combined_noops':read(OUT/'analysis/native256_preflight.json')['all_checks_passed'],
        'all_coordinate_maps_finite_counted':True,'2048samebody_prefix_checks':summary['same_body_fourfunction_checks']==2048,
        'published64H16MLP16QKV64natural':len(files)==144,'frozen_material':sha(CONTRACT/'material/initial.json')==contract['material_sha256'],
        'parser_frozen_before_outputs':sha(TESTS/'phase2683_explicit_answer_audit.py')==contract['parser_sha256']}
    assert all(checks.values());save(OUT/'analysis/scientific_checks.json',{'all_checks_passed':True,'checks':checks,'summary':summary})
    return checks,summary


def finalize(checks,summary):
    finish(2687,'8192独立因素语言条件：原生八层QKV、全层H/MLP操作图谱与自然输出双账',OUT,
        {'provenance':str(Path(__file__)),'checks':checks,'summary':summary},
        '正式输出前冻结材料和评分。64真实例先做固定256形状的baseline/双采集器/restored比对；随后每个条件处理所有真实token、全部H/MLP坐标及八层全部QKV/headnorm/RoPE坐标。自然cachegreedy32token另行前向，不用固定场首token启动。',
        r'D_{a,j}(x)=X_j(x_{a=1})-X_j(x_{a=0});\quad C^+_{a,j}=\sum_x[D_{a,j}(x)>0],\ C^-_{a,j}=\sum_x[D_{a,j}(x)<0];\quad A_{a,j}=\sum_x|D_{a,j}(x)|;\quad C^{4+}_{a,j}=\sum_b\prod_{f=1}^4[D_{a,j}(b,f)>0].',
        'C00164例联合采集无操作；C0028192原生BF16固定场+自然32token缓存生成；C003四种外部操作各4096边全坐标符号/绝对量/相消及1024四功能组；C004全部真实token逐坐标和/平方和；C0052048同正文四功能精确前缀；C00664全H/16全MLP/16全QKV来源原场，另64自然末状态；C007固定旧严格/有限归一化及末行明确答案解析分账。',
        '真实单参数路径的上游地图已经具备基础外部操作索引。观察差是同坐标两个独立前向的比较，不是把另一材料的激活搬入模型。名单、事实位置、关系目标和表达各自保留完整响应，部分复用不因全局门失败删除。',
        '更正2686文字分母：8×2×4×2=128基础实例，每实例64条件，不是256×32；实际8192材料与4096配对边完全不变。P地图按绝对来源位置对齐而非同语义token，缺失来源不计，真实未来token零概率保留。token矩不保留全部跨token关联，仅16来源例有完整原场。固定场与自然生成的BF16数值差不是语义差；答案匹配不验证解释或一般推理。全局同号也不是语义特异齿轮。',
        '继续2688逐Wq/Wk/Wv完整输入坐标项与相消账本，2689真实单标量内生headnorm/softmax验证，再按冻结合同完成2690独立扩大和2691顺序跨模型；不把观测图谱或已知算术当作机制闭合。')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('prepare','run','audit'),default='run');args=parser.parse_args()
    rows,c=prepare()
    if args.action=='prepare':return
    assert not (OUT/'analysis/final.json').exists()
    if args.action=='run':
        model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
        save(OUT/'protocol/runtime.json',{'dtype':str(model.dtype),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':False,'device_map':getattr(model,'hf_device_map',None)})
        qualify(model,tok,rows);collect(model,tok,rows,c)
        del model;gc.collect();torch.cuda.empty_cache()
    checks,summary=audit(rows,c);finalize(checks,summary)


if __name__=='__main__':main()
