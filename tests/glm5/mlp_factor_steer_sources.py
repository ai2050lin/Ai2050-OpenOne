"""Read later untouched row groups from the same pinned Wikimedia snapshot."""
import mlp_factor_steer_common as C
from mlp_functional_sources import prefix_hash


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    C.OUT.mkdir(parents=True,exist_ok=True)
    dest=C.OUT/'documents.json.gz'
    if dest.exists():return
    old=C.read_gz(C.PREVIOUS/'documents.json.gz')['docs'];seen={v['prefix_sha256'] for v in old}
    fs=HfFileSystem(token=False);tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);sources=[]
    for lang in ['en','zh']:
        cache=C.OUT/f'new-{lang}.json.gz'
        if cache.exists():sources.append(C.read_gz(cache));continue
        part=[v for v in old if v['language']==lang];meta=part[0];start=max(v['row_group'] for v in part)+1;docs=[]
        with fs.open(f'datasets/{meta["repo"]}@{meta["revision"]}/{meta["file"]}','rb',block_size=8*1024**2) as stream:
            table=pq.ParquetFile(stream)
            for group in range(start,table.num_row_groups):
                for i,v in enumerate(table.read_row_group(group,columns=['id','title','url','text']).to_pylist()):
                    ids=tok.encode(v['text'],add_special_tokens=False)
                    if len(ids)<1040:continue
                    fp=prefix_hash(ids)
                    if fp in seen:continue
                    seen.add(fp)
                    docs.append({**{k:meta[k] for k in ['source','repo','revision','license','file','language']},
                                 'id':str(v['id']),'title':v['title'],'url':v['url'],'row_group':group,'row':i,
                                 'tokens':ids[:1040],'prefix_sha256':fp})
                print(lang,group,len(docs),flush=True)
                if len(docs)>=C.TOTAL//2:break
        assert len(docs)>=C.TOTAL//2
        docs=sorted(docs,key=lambda r:C.digest([C.SEED,lang,r['id']]))[:C.TOTAL//2]
        C.write_gz(cache,docs);sources.append(docs)
    docs=[r for pair in zip(*sources) for r in pair]
    assert len({r['prefix_sha256'] for r in docs})==C.TOTAL
    C.write_gz(dest,{**C.identity(),'schema':'factor_steer_backgrounds.v1','docs':docs,'source_sha256':C.file_sha(__file__)})
    print('new independent backgrounds',len(docs),flush=True)


if __name__=='__main__':main()
