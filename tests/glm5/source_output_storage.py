"""Read-only proof that an old offload directory is exactly reconstructible."""
import hashlib,json,math,time
from pathlib import Path
import source_output_core as S
from source_output_core import C

def digest(stream,n=None):
    h=hashlib.sha256()
    while n is None or n>0:
        chunk=stream.read(8<<20 if n is None else min(n,8<<20))
        if not chunk:break
        h.update(chunk)
        if n is not None:n-=len(chunk)
    assert n is None or n==0
    return h.hexdigest()

def main():
    root=Path('D:/ai2050_offload/qwen3-14b');assert root.resolve()==root and not root.is_junction()
    model=C.ROOT/'models/hf/Qwen3-14B';index=C.read(root/'index.json');weights=C.read(model/'model.safetensors.index.json')['weight_map']
    files=set(p.name for p in root.iterdir());assert files=={'index.json'}|{n+'.dat' for n in index}
    headers={};rows=[];start=time.monotonic()
    for i,(name,meta) in enumerate(index.items()):
        file=root/(name+'.dat');shard=model/weights[name]
        if shard.name not in headers:
            with shard.open('rb') as f:n=int.from_bytes(f.read(8),'little');headers[shard.name]=(8+n,json.loads(f.read(n)))
        offset,header=headers[shard.name];tensor=header[name];assert tensor['dtype']=='BF16' and meta['dtype']=='bfloat16' and tensor['shape']==meta['shape']
        lo,hi=tensor['data_offsets'];assert file.stat().st_size==hi-lo==2*math.prod(meta['shape'])
        with shard.open('rb') as f:f.seek(offset+lo);expected=digest(f,hi-lo)
        with file.open('rb') as f:actual=digest(f)
        assert actual==expected,('Offload bytes differ from retained model',name)
        rows.append({'name':name,'path':str(file),'bytes':file.stat().st_size,'mtime_ns':file.stat().st_mtime_ns,'sha256':actual,
            'retained_model_shard':str(shard),'tensor_byte_offset':offset+lo,'shape':meta['shape'],'dtype':'BF16'})
        if (i+1)%8==0:print(json.dumps({'stage':'offload_reconstruction_audit','done':i+1,'total':len(index),'seconds':round(time.monotonic()-start,1)}),flush=True)
    C.save('storage-offload-audit.json',{'schema':'source_output_storage_audit.v1','execution_only':True,'target':str(root),
        'passed':True,'bytes':sum(r['bytes'] for r in rows)+(root/'index.json').stat().st_size,
        'index':index,'index_sha256':C.sha(root/'index.json'),'retained_model_index_sha256':C.sha(model/'model.safetensors.index.json'),
        'reconstruction':'Write each recorded BF16 byte range from its retained model shard to name.dat; write the archived index mapping as index.json.',
        'records':rows,'contains_research_activations':False,'deletion_performed':False})
    print(json.dumps({'passed':True,'tensors':len(rows),'GiB':sum(r['bytes'] for r in rows)/2**30}),flush=True)

if __name__=='__main__':main()
