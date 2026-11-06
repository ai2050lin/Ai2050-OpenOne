"""Discovery-only long-visible-prefix poles; never read heldout activations."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as OLD
from shared.dist_protocol.mechanism_trace import unbits,file_sha
ROOT=OLD.ROOT;OUT=ROOT/'tests/glm5/result/native_mlp_function_v1';PUBLIC=ROOT/'research/native_mlp_function/v1'
RUN='NMLP-FUNCTION-20261104-v1';MODEL=OLD.MODEL;SEED=4112026
read=OLD.read;save=OLD.save
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True);a=read(OLD.OUT/'analysis.json');data=read(OLD.OUT/'documents.json');tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    raw=np.load(OLD.OUT/'input-tokens.npy',mmap_mode='r');selected=[]
    for lo,hi in [(6,13),(14,21),(22,29),(30,35)]:
        pool=sorted([c for c in a['cards'] if c['selection']=='stratified_random' and lo<=c['layer']<=hi],key=lambda c:-c['poles']['0:Gpos']['coherence'])
        selected.append(pool[0]);selected.append(next(c for c in pool[1:] if c['layer']!=pool[0]['layer']))
    records=[]
    for c in selected:
        l,u=c['layer'],c['unit'];j=[r['unit'] for r in data['units'] if r['layer']==l].index(u);examples={}
        arrays={k:np.load(OLD.OUT/f'L{l:02d}-{k}.npy',mmap_mode='r') for k in ['G','V','A']}
        for key in ['G','A']:
            x=unbits(np.asarray(arrays[key][:1024,512:,j]))
            for sign in [1,-1]:
                positions=(x*sign).argmax(1)+512;vals=x[np.arange(1024),positions-512];order=np.argsort(-vals*sign)[:12];ee=[]
                for di in order:
                    p=int(positions[di]);ee.append({'doc_index':int(di),'position':p,'visible_tokens':p+1,'current_token_id':int(raw[di,p]),'current_token':tok.decode([int(raw[di,p])]),
                        'prefix':tok.decode(raw[di,:p+1]),'GVA':{k:float(unbits(v[di,p,j])) for k,v in arrays.items()},'doc':data['docs'][di]})
                examples[key+('pos' if sign==1 else 'neg')]=ee
        records.append({'layer':l,'unit':u,'selection':'top discovery-only Gpos h coherence, two distinct layers per prespecified layer group; purposive not random population sample',
            'old_discovery_G_coherence':c['poles']['0:Gpos']['coherence'],'examples':examples})
    result={'kind':'real','run_id':RUN,'model_snapshot':a['model_snapshot'],'units':records,'source_sha256':file_sha(OLD.OUT/'analysis.json'),
        'only_discovery_used':True,'minimum_visible_tokens':513,'provenance':a['provenance']}
    save(OUT/'discovery.json',result)
    for r in records:
        print(f'B{r["layer"]}/U{r["unit"]}',flush=True)
        for pole,ee in r['examples'].items():
            for e in ee[:3]:print(json.dumps({'pole':pole,'token':e['current_token'],'GVA':e['GVA'],'tail':e['prefix'][-250:]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
