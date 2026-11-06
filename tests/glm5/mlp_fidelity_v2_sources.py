"""New pinned natural documents, row groups not consumed by previous studies."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import digest,file_sha
import numpy as np

def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);fs=HfFileSystem(token=False);old=P.read(P.LONG/'documents.json')['docs'];used={(d['source'],str(d['id'])) for d in old}
    corpus=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_code_v2/corpus.json');used.update((d['background_source']['source'],str(d['background_source']['id'])) for d in corpus['items']);prior=np.load(P.LONG/'input-tokens.npy',mmap_mode='r')
    import hashlib
    h=lambda ids:hashlib.sha256(np.asarray(ids[:1024],np.int32).tobytes()).hexdigest()
    prefixes={h(x) for x in prior};prefixes.update(h(d['tokens']) for source in ['en','zh'] for d in P.read(P.LONG/f'wikipedia-{source}-fresh.json')['docs'])
    selected=[]
    for lang in ['en','zh']:
        dest=P.OUT/f'new-source-{lang}.json'
        if dest.exists():selected+=P.read(dest)['docs'];continue
        meta=P.read(P.LONG/f'wikipedia-{lang}-fresh.json')['source'];fn=meta['row_groups'][0]['file'];docs=[];groups=[]
        with fs.open(f'datasets/{meta["repo"]}@{meta["revision"]}/{fn}','rb',block_size=8*1024**2) as fh:
            pf=pq.ParquetFile(fh)
            for rg in range(7,pf.num_row_groups):
                rr=pf.read_row_group(rg,columns=[n for n in ['id','url','title','text'] if n in pf.schema_arrow.names]).to_pylist();groups.append(rg)
                for row,d in enumerate(rr):
                    if (meta['source'],str(d.get('id',''))) in used:continue
                    ids=tok.encode(d['text'],add_special_tokens=False)
                    if len(ids)<1100 or h(ids) in prefixes:continue
                    prefixes.add(h(ids));docs.append({'source':meta['source'],'repo':meta['repo'],'revision':meta['revision'],'license':meta['license'],'file':fn,'row_group':rg,'row':row,'id':str(d.get('id','')),'title':d.get('title',''),'url':d.get('url',''),'language':lang,'tokens':ids[:1100],'prefix_sha256':h(ids)})
                print(json.dumps({'language':lang,'row_group':rg,'eligible':len(docs)}),flush=True)
                # Natural256 + controlled256 perlanguage. Controls alternatebackgroundlanguages;
                # taskinstructions areEnglish inboth, explicitly notChinese-tasktransfer.
                if len(docs)>=512:break
        assert len(docs)>=512;docs=sorted(docs,key=lambda d:digest({'seed':P.SEED,'source':d['source'],'id':d['id']}))[:512]
        P.save(dest,{'source':meta,'row_groups':groups,'docs':docs,'fetch_script_sha256':file_sha(__file__)});selected+=docs
    natural=[];controlled=[]
    for lang in ['en','zh']:
        rows=[d for d in selected if d['language']==lang]
        natural +=[{**d,'split':'confirm' if i<128 else 'lockbox'} for i,d in enumerate(rows[:256])]
        controlled+=rows[256:512]
    # Equal source-language representation in every controlled split and frame.
    controlled=[v for pair in zip([r for r in controlled if r['language']=='en'],[r for r in controlled if r['language']=='zh']) for v in pair]
    P.save(P.OUT/'fresh-documents.json',{'schema':'mlp_fidelity_fresh_documents.v1','kind':'real','run_id':P.RUN,'natural':natural,'controlled':controlled,'source_scope':'newpinnedrowgroups beyond6; allcachedold1024prefixesexcluded; notpretrainingunseen','source_sha256':file_sha(__file__)})
    print(json.dumps({'natural':len(natural),'controlled':len(controlled)}),flush=True)
if __name__=='__main__':main()
