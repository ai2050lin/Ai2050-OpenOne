"""Independent streamed checkpoint SHA and direct BF16 scalar byte audit."""
import sys,struct,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from phase2620_native_coordinate_contract import read,save,sha,RESULT
from server.native_qkv_parameter_query import native_weight,MODEL,query

OUT=RESULT/'phase2693_qkv_campaign_delivery';LINK=RESULT/'phase2692_linked_native_ledger'

def locate(key):
    index=read(MODEL/'model.safetensors.index.json')['weight_map'];path=MODEL/index[key]
    with path.open('rb') as f:
        length=int.from_bytes(f.read(8),'little');import json
        meta=json.loads(f.read(length))[key]
    assert meta['dtype']=='BF16'
    return path,8+length+meta['data_offsets'][0],meta['data_offsets'][1]-meta['data_offsets'][0],meta['shape']

def scalar(key,indices):
    path,start,size,shape=locate(key);offset=0
    assert len(indices)==len(shape)
    for i,n in zip(indices,shape):assert 0<=i<n;offset=offset*n+i
    with path.open('rb') as f:f.seek(start+2*offset);raw=f.read(2)
    return struct.unpack('<f',b'\0\0'+raw)[0]

def main(published=False):
    matrices=[];points=[]
    for layer in (0,5,17,23,26,27,28,35):
        manifest=read(LINK/f'weights/L{layer}_checkpoint_manifest.json')
        for kind,r in manifest.items():
            path,start,size,shape=locate(r['checkpoint_key']);assert shape==r['shape']
            h=hashlib.sha256();remaining=size
            with path.open('rb') as f:
                f.seek(start)
                while remaining:
                    b=f.read(min(8*1024**2,remaining));assert b;h.update(b);remaining-=len(b)
            assert h.hexdigest()==r['native_bytes_sha256']
            matrices.append({'layer':layer,'key':r['checkpoint_key'],'complete_native_bytes':size,'sha256':h.hexdigest()})
            indices=[(0,),(shape[0]-1,)] if len(shape)==1 else [(0,0),(0,shape[1]-1),(shape[0]-1,0),(shape[0]-1,shape[1]-1),(shape[0]//2,shape[1]//2)]
            for idx in indices:
                a=scalar(r['checkpoint_key'],idx);b=float(native_weight(r['checkpoint_key'],idx));assert a==b
                points.append({'key':r['checkpoint_key'],'index':idx,'exact_native_value':a})
        # Whole query uses explicit final-unit/coordinate addresses, checked by
        # independently seeking checkpoint bytes, not by its mmap helper.
        q=query(case=0,layer=layer,kind='v',output_row=1023,input_coordinate=2559,token=82,
                query_position=1,source_token=82,head=31,head_coordinate=127,checkpoint=36,unit=9727,output_coordinate=2559)
        for name,key,idx in [('actual_gate_j_k',f'model.layers.{layer}.mlp.gate_proj.weight',(9727,2559)),
                             ('actual_up_j_k',f'model.layers.{layer}.mlp.up_proj.weight',(9727,2559)),
                             ('actual_down_out_j',f'model.layers.{layer}.mlp.down_proj.weight',(2559,9727))]:
            assert q['values'][name]==scalar(key,idx)
    assert len(matrices)==48
    from phase2693_campaign_terminal import binding
    prefix='' if published else 'staged_'
    save(OUT/f'analysis/{prefix}checkpoint_byte_audit.json',{'all_checks_passed':True,**binding(not published),
        'complete_checkpoint_arrays':matrices,'direct_scalar_points':points,'full_query_lastMLPunit_checks':8,
        'independent_reference':'Phase2692 native tensor byte SHA was obtained via safetensors PyTorch CPU; this test streams native file bytes and direct scalar offsets without torch.',
        'code_sha256':sha(Path(__file__)),'not_causal_semantic_validation':True})
    print('2693 COMPLETE WEIGHT HASHES',len(matrices),'SCALARS',len(points),'FULL QUERY LASTUNIT',8,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--published',action='store_true');args=parser.parse_args()
    main(args.published)
