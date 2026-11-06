"""Document-disjoint long-context native-neuron audit; no model activations here."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared.dist_protocol.mechanism_trace import digest,file_sha
from tools.power_analysis import check_phase
OUT=ROOT/'tests/glm5/result/native_mlp_long_v1';PUBLIC=ROOT/'research/native_mlp_long/v1'
RUN='NMLP-LONG-20261104-v1';MODEL=Path('D:/develop/model/Qwen/Qwen3-4B');SEED=28112026
LEGACY=Path('D:/ai2050_native/_census/corpus/2ca5f0a756d36c8a')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d,immutable=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
    if immutable and p.exists():
        if p.read_text(encoding='utf-8')!=b:raise ValueError('immutable file differs: '+str(p))
    else:p.write_text(b,encoding='utf-8')
def sha_tokens(t):return hashlib.sha256(np.asarray(t,np.int32).tobytes()).hexdigest()

def fetch():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);man=read(LEGACY/'manifest.json');fs=HfFileSystem(token=False)
    for s in man['sources']:
        if not s['source'].startswith('wikipedia'):continue
        dest=OUT/(s['source']+'-fresh.json');
        if dest.exists():continue
        fn=s['row_groups'][0]['file'];start=max(x['row_group'] for x in s['row_groups'] if x['file']==fn)+1;docs=[];used=[]
        with fs.open(f'datasets/{s["repo"]}@{s["revision"]}/{fn}','rb',block_size=8*1024**2) as fh:
            pf=pq.ParquetFile(fh)
            for rg in range(start,pf.num_row_groups):
                columns=[n for n in ['id','url','title','text'] if n in pf.schema_arrow.names]
                data=pf.read_row_group(rg,columns=columns).to_pylist();used.append(rg)
                for row,r in enumerate(data):
                    ids=tok.encode(r['text'],add_special_tokens=False)
                    if len(ids)<1025:continue
                    docs.append({'source':s['source'],'repo':s['repo'],'revision':s['revision'],'license':s['license'],'file':fn,
                        'row_group':rg,'row':row,'id':str(r.get('id','')),'url':r.get('url',''),'title':r.get('title',''),
                        'tokens':ids[:2048],'original_token_length':len(ids)})
                print(json.dumps({'source':s['source'],'row_group':rg,'long_docs':len(docs)}),flush=True)
                if len(docs)>=800:break
        assert len(docs)>=512
        save(dest,{'source':s,'unused_row_groups':used,'docs':docs,'note':'row groups beyond prior20M census; not guaranteed unseen in model pretraining'})

def build():
    man=read(LEGACY/'manifest.json');legacy_hashes=set();training=[]
    assert file_sha(ROOT/'models/hf/qwen3-4b/tokenizer.json')==file_sha(MODEL/'tokenizer.json')
    for s in man['sources']:
        path=LEGACY/s['file'];assert file_sha(path)==s['file_sha256']
        z=np.load(path);tokens=z['tokens'];starts=z['doc_starts'];ends=np.r_[starts[1:]-1,len(tokens)-1]
        candidates=[]
        for index,(a,b) in enumerate(zip(starts,ends)):
            if b-a<1025:continue
            ids=tokens[a:min(b,a+2048)];h=sha_tokens(ids[:1024]);legacy_hashes.add(h)
            if s['source'].startswith('fineweb'):
                candidates.append({'source':s['source'],'repo':s['repo'],'revision':s['revision'],'license':s['license'],
                    'legacy_stream_offset':int(a),'id':f'{s["source"]}:{index}','tokens':ids.tolist(),'prefix_sha256':h,'split':'discovery'})
        if candidates:
            unique={r['prefix_sha256']:r for r in candidates};ordered=sorted(unique.values(),key=lambda r:digest({'seed':SEED,'id':r['id']}))
            assert len(ordered)>=512;training.extend(ordered[:512])
    held=[];newhash=set();excluded=0
    for source in ['wikipedia-en','wikipedia-zh']:
        docs=read(OUT/(source+'-fresh.json'))['docs'];chosen=[]
        for r in sorted(docs,key=lambda r:digest({'seed':SEED,'id':r['id'],'source':source})):
            h=sha_tokens(r['tokens'][:1024])
            if h in legacy_hashes or h in newhash:excluded+=1;continue
            newhash.add(h);chosen.append({**r,'prefix_sha256':h})
            if len(chosen)==512:break
        assert len(chosen)==512
        for i,r in enumerate(chosen):r['split']='confirm' if i<256 else 'lockbox'
        held.extend(chosen)
    docs=training+held;assert len(docs)==2048
    # Conservative near-duplicate check: 5-token shingle overlap, candidates from
    # 16 MinHashes/4 bands. Fail instead of silently mixing near-duplicate splits.
    rng=np.random.default_rng(SEED);aa=rng.integers(1,2**63,16,dtype=np.uint64)|np.uint64(1);bb=rng.integers(0,2**63,16,dtype=np.uint64)
    bands={};shingles=[];near=[]
    for i,r in enumerate(docs):
        t=np.asarray(r['tokens'][:1024],np.uint64);h=np.zeros(1020,np.uint64)
        with np.errstate(over='ignore'):
            for k in range(5):h=h*np.uint64(1000003)+t[k:k+1020]
            h=np.unique(h);sig=(h[None,:]*aa[:,None]+bb[:,None]).min(1)
        candidates=set()
        for k in range(4):candidates.update(bands.get((k,tuple(sig[k*4:k*4+4])),[]))
        for j in candidates:
            overlap=len(np.intersect1d(h,shingles[j],assume_unique=True));sim=overlap/(len(h)+len(shingles[j])-overlap)
            if sim>=.85 and docs[j]['split']!=r['split']:near.append({'a':j,'b':i,'jaccard':sim})
        for k in range(4):bands.setdefault((k,tuple(sig[k*4:k*4+4])),[]).append(i)
        shingles.append(h)
    assert not near,near[:3]
    raw=np.full((2048,2048),151643,np.int32)
    for i,r in enumerate(docs):raw[i,:len(r['tokens'])]=r['tokens'];r['doc_index']=i;r['doc_id']=r['source']+':'+str(r['id'])
    np.save(OUT/'input-tokens.npy',raw)
    pairs={}
    seen=np.bincount(raw[:1024,:1024].ravel(),minlength=151936)
    for split in ['confirm','lockbox']:
        pp=[]
        for source in ['wikipedia-en','wikipedia-zh']:
            ix=[r['doc_index'] for r in docs if r['split']==split and r['source']==source]
            from scipy.optimize import linear_sum_assignment
            left,right=ix[:128],ix[128:];matches=((raw[left,None,512:1024]==raw[None,right,512:1024])&(seen[raw[left,None,512:1024]]>=20)).sum(-1)
            rr,cc=linear_sum_assignment(np.where(matches>0,0.,1e6)+rng.random(matches.shape))
            assert np.all(matches[rr,cc]>0),'no complete text-only matched pairing'
            for a,b in [(left[i],right[j]) for i,j in zip(rr,cc)]:
                positions=np.flatnonzero((raw[a,512:1024]==raw[b,512:1024])&(seen[raw[a,512:1024]]>=20))+512
                assert len(positions)>0,(a,b)
                pos=int(positions[int(digest({'seed':SEED,'a':a,'b':b})[:8],16)%len(positions)])
                pp.append({'a':a,'b':b,'position':pos,'token_id':int(raw[a,pos]),'pair_id':split+f':{a}:{b}'})
        assert len(pp)==256;pairs[split]=pp
    units=[]
    for l in range(36):
        units.extend({'layer':l,'unit':int(u),'selection':'stratified_random'} for u in rng.choice(9728,12,replace=False))
    prior=read(ROOT/'tests/glm5/result/phase2794_arbitration_v4/selection.json')['units']+[[30,9517],[30,80]]
    for l,u in prior:
        if not any(r['layer']==l and r['unit']==u for r in units):units.append({'layer':l,'unit':u,'selection':'prior_purposive'})
    save(OUT/'documents.json',{'docs':[{k:v for k,v in r.items() if k!='tokens'} for r in docs],'pairs':pairs,'units':units,
        'legacy_exact_prefix_exclusions':excluded,'cross_split_near_duplicates':near,'near_duplicate_check':'16 MinHash/4bands candidates, exact5-token Jaccard>=.85; not exhaustive all-pair similarity',
        'input_tokens_sha256':file_sha(OUT/'input-tokens.npy')})
    power=check_phase(.25,256,p_d=.5,alpha=.05/432);assert power['pass']
    prereg={'schema':'native_mlp_long_prereg.v1','kind':'real','run_id':RUN,'seed':SEED,'model_snapshot':read(ROOT/'research/phase2794/report.research.json')['model_snapshot'],
        'documents_sha256':file_sha(OUT/'documents.json'),'input_tokens_sha256':file_sha(OUT/'input-tokens.npy'),'natural_docs':2048,'input_tokens':2097152,
        'splits':'1024 discovery fineweb en/zh,512 confirm and512lockbox fresh wikipedia en/zh rowgroups not used by priorcensus; exactprefix overlap with old corpus excluded; model pretraining exposure unknown',
        'sampling':'432 random native units,12/layer; earlier32+2 purposed units separate; not the whole population explained',
        'collection':'1024-token plain documents, no chat template; first32 positions excluded from statistics; all350208 native G/V/A moments, regimes andextrema; complete selected G/V/A at all input positions andall37 hidden states at final position retained',
        'primary':'for432 random units, predict A ordering of same-token, same-absolute-position paired fresh documents atposition>=512; pairs frozen fromtext before activations, independent doc pairs',
        'predictor':'discovery token means + position-bin residual + Ridge lambda100 onprefix token-category counts at16/64/256/512 andpreceding4 categories; training samplesevery8positions after32. No hidden state/future token/sourceID input.',
        'criteria':{'expected_delta':.25,'power':power,'minimum_delta_vs_random_tie_baseline':.15,'alpha_per_unit':.05/432,'confirm_and_lockbox_both_required':True,'min_nonzero_pair_fraction':.8},
        'negative_control':'same-token shuffle of discovery residuals before fitting context; samefrozen controls andp-value thresholds; original-token baseline is tied byconstruction',
        'pole_analysis':'G andA positive/negative doc extrema, actualG/V andinput h; strongest doc examples andshoulder are retrieval diagnostics, not semantic labels; read/write alignment descriptive, not a pass/fail monosemanticity proof',
        'intervention':'after confirm select16 highest ranking gains, atmostone perlayer, excludinglockbox; strictcontrol A-RMS/down-norm/product ratios[.8,1.25],otherwise skipcandidate; atmatchedposition on256lockbox A-documents,alpha.5/1 andrandom16alpha1; accuracy/fullvocabKL/true-nexttokenloss retained, not semantic specificity',
        'intervention_power':check_phase(.1,256,p_d=.2),'capability':'repeat2200formal keyed monitor for selected16alpha1;2pp NI;140reference-only excluded',
        'storage':'all-unit aggregate statistics only; selected-unit arrays losslessBF16; rare-pole reader vectors retained; do not claim all natural-token全模块原场'}
    save(PUBLIC/'prereg.research.json',prereg);print(json.dumps({'docs':2048,'units':len(units),'pairs':256,'power':power},ensure_ascii=False))
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    fetch() if len(sys.argv)>1 and sys.argv[1]=='fetch' else build()
