"""Fetch a second independent background cohort after the strict format gate failure."""
import mlp_functional_common as C
from mlp_functional_sources import prefix_hash


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    out=C.OUT/'revision';out.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);fs=HfFileSystem(token=False)
    enrolled=[]
    for lang in ['en','zh']:
        cache=out/f'{lang}-source.json'
        if cache.exists():
            enrolled.append(C.read(cache)['docs']);continue
        old=C.read(C.OUT/f'new-{lang}.json')['docs'];meta=old[0]
        start=max(d['row_group'] for d in old)+1
        docs=[];seen={d['prefix_sha256'] for d in old}
        with fs.open(f'datasets/{meta["repo"]}@{meta["revision"]}/{meta["file"]}','rb',block_size=8*1024**2) as stream:
            parquet=pq.ParquetFile(stream)
            for group in range(start,parquet.num_row_groups):
                for row,item in enumerate(parquet.read_row_group(group,columns=['id','url','title','text']).to_pylist()):
                    ids=tok.encode(item['text'],add_special_tokens=False)
                    if len(ids)<1100:continue
                    fp=prefix_hash(ids)
                    if fp in seen:continue
                    seen.add(fp)
                    docs.append({**{k:meta[k] for k in ['source','repo','revision','license','file','language']},
                                 'id':str(item['id']),'url':item['url'],'title':item['title'],'row_group':group,'row':row,
                                 'tokens':ids[:1100],'prefix_sha256':fp})
                print(lang,group,len(docs),flush=True)
                if len(docs)>=176:break
        assert len(docs)>=176
        docs=sorted(docs,key=lambda d:C.digest([C.SEED,'format-revision',d['id']]))[:176]
        C.save(cache,{'docs':docs,'source_sha256':C.file_sha(__file__)})
        enrolled.append(docs)
    docs=[d for pair in zip(*enrolled) for d in pair]
    assert len({d['prefix_sha256'] for d in docs})==352
    C.save(out/'documents.json',{**C.identity(),'kind':'real','docs':docs,'source_sha256':C.file_sha(__file__)})
    print('352 revision backgrounds frozen',flush=True)


if __name__=='__main__':main()
