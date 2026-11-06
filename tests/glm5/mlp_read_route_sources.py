"""Enroll backgrounds from the next unused groups of the pinned Wikimedia corpus."""
import mlp_read_route_common as C
from mlp_functional_sources import prefix_hash


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    C.OUT.mkdir(parents=True,exist_ok=True)
    if (C.OUT/'documents.json.gz').exists():return
    previous=C.read(C.PREVIOUS/'targeted/inputs.json')['docs']
    other=C.read(C.PREVIOUS/'documents.json')['docs']
    seen={r['prefix_sha256'] for r in previous+other}
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    fs=HfFileSystem(token=False);cohorts=[]
    for lang in ['en','zh']:
        cache=C.OUT/f'source-{lang}.json.gz'
        if cache.exists():cohorts.append(C.read_gz(cache));continue
        old=[r for r in previous if r['language']==lang];meta=old[0]
        start=max(r['row_group'] for r in old)+1;rows=[]
        with fs.open(f'datasets/{meta["repo"]}@{meta["revision"]}/{meta["file"]}','rb',block_size=8*1024**2) as stream:
            parquet=pq.ParquetFile(stream)
            for group in range(start,parquet.num_row_groups):
                for i,r in enumerate(parquet.read_row_group(group,columns=['id','title','url','text']).to_pylist()):
                    ids=tok.encode(r['text'],add_special_tokens=False)
                    if len(ids)<1040:continue
                    fp=prefix_hash(ids)
                    if fp in seen:continue
                    seen.add(fp)
                    rows.append({**{k:meta[k] for k in ['source','repo','revision','license','file','language']},
                                 'id':str(r['id']),'title':r['title'],'url':r['url'],'row':i,'row_group':group,
                                 'tokens':ids[:1040],'prefix_sha256':fp})
                print(lang,group,len(rows),flush=True)
                if len(rows)>=C.N_SOURCE//2:break
        assert len(rows)>=C.N_SOURCE//2,(lang,len(rows))
        rows=sorted(rows,key=lambda r:C.digest([C.SEED,'background',lang,r['id']]))[:C.N_SOURCE//2]
        C.write_gz(cache,rows);cohorts.append(rows)
    docs=[r for pair in zip(*cohorts) for r in pair]
    assert len({r['prefix_sha256'] for r in docs})==C.N_SOURCE
    C.write_gz(C.OUT/'documents.json.gz',{**C.identity(),'schema':'mlp_read_route_sources.v1','docs':docs,'source_sha256':C.file_sha(__file__)})
    print('frozen distinct new backgrounds',len(docs),flush=True)


if __name__=='__main__':main()
