"""Independent bilingual backgrounds from later row groups of the frozen Wikipedia source."""
import mlp_kv_common as C
from mlp_functional_sources import prefix_hash


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    C.OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    fs=HfFileSystem(token=False);sources=[];previous=C.read(C.PREVIOUS/'revision/documents.json')['docs']
    seen={r['prefix_sha256'] for r in previous}
    for lang in ['en','zh']:
        old=C.read(C.PREVIOUS/f'new-{lang}.json')['docs'];seen.update(r['prefix_sha256'] for r in old)
        start=1+max(r['row_group'] for r in previous if r['language']==lang)
        file=old[0]['file'];repo=old[0]['repo'];revision=old[0]['revision'];cache=C.OUT/f'source-{lang}.json'
        if cache.exists():sources.append(C.read(cache)['docs']);continue
        docs=[]
        with fs.open(f'datasets/{repo}@{revision}/{file}','rb',block_size=8*1024**2) as stream:
            table=pq.ParquetFile(stream)
            for group in range(start,table.num_row_groups):
                for i,row in enumerate(table.read_row_group(group,columns=['id','url','title','text']).to_pylist()):
                    tokens=tok.encode(row['text'],add_special_tokens=False)
                    if len(tokens)<1040:continue
                    fp=prefix_hash(tokens)
                    if fp in seen:continue
                    seen.add(fp)
                    docs.append({**{k:old[0][k] for k in ['source','repo','revision','license','file','language']},
                                 'id':str(row['id']),'url':row['url'],'title':row['title'],'row_group':group,'row':i,
                                 'tokens':tokens[:1040],'prefix_sha256':fp})
                print(lang,group,len(docs),flush=True)
                if len(docs)>=176:break
        assert len(docs)>=176,(lang,start,len(docs))
        docs=sorted(docs,key=lambda r:C.digest([C.SEED,lang,r['id']]))[:176]
        C.save(cache,{'docs':docs,'script_sha256':C.file_sha(__file__)});sources.append(docs)
    docs=[r for pair in zip(*sources) for r in pair]
    assert len({r['prefix_sha256'] for r in docs})==352
    C.save(C.OUT/'documents.json',{**C.identity(),'schema':'mlp_kv_sources.v1','docs':docs,'source_sha256':C.file_sha(__file__),
                                  'scope':'earlier project source row groups excluded; pretraining overlap unknown'})
    print('352 new background documents',flush=True)


if __name__=='__main__':main()
