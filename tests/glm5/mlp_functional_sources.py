"""Fetch independent pinned row groups beyond both earlier Wikipedia rounds."""
import hashlib

import numpy as np
import mlp_functional_common as C


def prefix_hash(ids):
    return hashlib.sha256(np.asarray(ids[:1024],dtype='<i4').tobytes()).hexdigest()


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    C.OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    old_dir=C.ROOT/'tests/glm5/result/native_mlp_long_v1'
    previous=C.read(C.PREVIOUS/'fresh-documents.json')
    exposed=previous['natural']+previous['controlled']
    excluded={prefix_hash(x) for x in np.load(old_dir/'input-tokens.npy',mmap_mode='r')}
    used={(r['source'],str(r['id'])) for r in exposed}
    excluded.update(prefix_hash(r['tokens']) for r in exposed)
    for lang in ['en','zh']:
        archived=C.read(old_dir/f'wikipedia-{lang}-fresh.json')['docs']
        used.update((r['source'],str(r['id'])) for r in archived)
        excluded.update(prefix_hash(r['tokens']) for r in archived)
    fs=HfFileSystem(token=False)
    natural, backgrounds=[],[]
    for lang in ['en','zh']:
        cache=C.OUT/f'new-{lang}.json'
        if cache.exists():
            rows=C.read(cache)['docs']
        else:
            meta=C.read(C.PREVIOUS/f'new-source-{lang}.json')
            source=meta['source']; file=source['row_groups'][0]['file']; rows=[]
            start=max(meta['row_groups'])+1
            with fs.open(f'datasets/{source["repo"]}@{source["revision"]}/{file}','rb',block_size=8*1024**2) as f:
                parquet=pq.ParquetFile(f)
                for group in range(start,parquet.num_row_groups):
                    for row,d in enumerate(parquet.read_row_group(group,columns=['id','url','title','text']).to_pylist()):
                        key=source['source'],str(d['id'])
                        if key in used:continue
                        ids=tok.encode(d['text'],add_special_tokens=False)
                        if len(ids)<1100:continue
                        fingerprint=prefix_hash(ids)
                        if fingerprint in excluded:continue
                        used.add(key);excluded.add(fingerprint)
                        rows.append({'source':key[0],'id':key[1],'repo':source['repo'],'revision':source['revision'],
                                     'file':file,'row_group':group,'row':row,'url':d['url'],'title':d['title'],
                                     'license':source['license'],'language':lang,'tokens':ids[:1100],'prefix_sha256':fingerprint})
                    print(lang,group,len(rows),flush=True)
                    if len(rows)>=512:break
            assert len(rows)>=512,(lang,start,len(rows))
            rows=sorted(rows,key=lambda d:C.digest({'seed':C.SEED,'lang':lang,'id':d['id']}))[:512]
            C.save(cache,{'docs':rows,'start_row_group':start,'source_sha256':C.file_sha(__file__)})
        natural.extend({**r,'split':'confirm' if i<128 else 'lockbox'} for i,r in enumerate(rows[:256]))
        backgrounds.append(rows[256:])
    controlled=[r for pair in zip(*backgrounds) for r in pair]
    assert len({r['prefix_sha256'] for r in natural+controlled})==1024
    C.save(C.OUT/'fresh-documents.json',{**C.identity(),'schema':'mlp_functional_sources.v1',
           'natural':natural,'controlled':controlled,'source_sha256':C.file_sha(__file__),
           'scope':'new pinned Wikipedia row groups, distinct source IDs and 1024-token hashes; unknown pretraining overlap'})
    print('Frozen 512 natural documents and 512 disjoint controlled backgrounds.',flush=True)


if __name__=='__main__':
    main()
