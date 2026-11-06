"""Stream all36 native arrays into unsorted, signed role-response profiles."""
import sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from shared.dist_protocol.mechanism_trace import unbits,digest,file_sha
from deploy.research_library import Library
def rounded(x):
    x=np.asarray(x);return np.array([float(format(v,'.7g')) for v in x.ravel()],np.float64).reshape(x.shape).tolist()
def main():
    c=P.read(P.OUT/'corpus.json');families=list(P.FAMILIES);splits=['discovery','confirm','lockbox'];s=np.zeros((3,3,36,4,3,9728),np.float64)
    head=np.zeros((3,3,36,4,32,128),np.float64);hidden=np.zeros((3,3,4,37,2560),np.float64);count=np.zeros((3,3),np.int64)
    checked_files=checked_bytes=0
    for index,item in enumerate(c['items']):
        folder=P.OUT/'native'/digest(item['id'])[:20];path=folder/'manifest.json'
        while not path.is_file():time.sleep(2)
        d=P.read(path);assert d['model_snapshot']==c['model_snapshot'] and d['sample_id']==item['id']
        assert d['trace_id']==digest({k:v for k,v in d.items() if k!='trace_id'})
        for f in d['files']:
            p=folder/f['path'];assert p.stat().st_size==f['bytes'] and file_sha(p)==f['sha256'];checked_files+=1;checked_bytes+=f['bytes']
        fi=families.index(item['family']);si=splits.index(item['split']);sign=1 if item['semantic'] else -1
        h=unbits(np.load(folder/'hidden_readouts.npy'));hidden[fi,si]+=sign*h.transpose(1,0,2)
        for l in range(36):
            with np.load(folder/f'block_{l:02d}.npz') as z:
                s[fi,si,l]+=sign*np.stack([unbits(z[k]) for k in ['gate','value','activation']],axis=1)
                head[fi,si,l]+=sign*unbits(z['head_z'])
        if item['semantic']:count[fi,si]+=1
        if (index+1)%128==0:print(json.dumps({'profiled':index+1,'total':len(c['items']),'verifiedGiB':checked_bytes/2**30}),flush=True)
    s/=count[:,:,None,None,None,None];head/=count[:,:,None,None,None,None];hidden/=count[:,:,None,None,None]
    np.savez(P.OUT/'profiles-native.npz',GVA_delta=s.astype(np.float32),head_z_delta=head.astype(np.float32),hidden_delta=hidden.astype(np.float32),count=count)
    while not (P.OUT/'observations.json').exists():time.sleep(2)
    obs=P.read(P.OUT/'observations.json');lib=Library(P.ROOT/'distributed_data/research_library');refs={}
    for fi,family in enumerate(families):
        refs[family]={}
        for l in range(36):
            doc={'schema':'research_native_function_profile.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'family':family,'block':l,'counts':dict(zip(splits,count[fi].tolist())),
                'readout_labels':['选择键之前（因果负对照）','选择键当前位置','完整问题后的共同冒号','助手回答边界'],
                'splits':{split:{'GVA_delta':rounded(s[fi,si,l]),'head_delta_norm':rounded(np.linalg.norm(head[fi,si,l],axis=-1)),
                    'embedding_delta':rounded(hidden[fi,si,:,0]),'hidden_delta':rounded(hidden[fi,si,:,l+1])} for si,split in enumerate(splits)},
                'meaning':'positive semantic-role minus negative role paired mean, raw coordinates unsorted; includes model errors; descriptive not FWER-confirmed semantic neurons',
                'precision':'7 significant decimal digits for viewer only; complete BF16 inputs andfloat32 derived NPZ retained', 'provenance':obs['provenance']}
            path=P.PUBLIC/'profiles'/f'{family}-B{l:02d}.research.json';P.save(path,doc);refs[family][str(l)]=lib.ingest(doc,str(path.relative_to(P.ROOT)),kind='derived_snapshot')['sha256']
    P.save(P.OUT/'profile-index.json',{'kind':'real','run_id':P.RUN,'refs':refs,'verified_files':checked_files,'verified_bytes':checked_bytes,'all_native_hashes_verified':True,'provenance':obs['provenance']})
    print(json.dumps({'layers':36,'families':3,'all_hashes_verified':True,'bytes':checked_bytes}),flush=True)
if __name__=='__main__':main()
