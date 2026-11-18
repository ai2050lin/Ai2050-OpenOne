"""Post-discovery grouping of prefix differences by padding/length, without new model calls."""
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
from source_binding_native_v2 import unbits

def main():
    rows=D.read(D.OUT/'qwen14-materials.json')['splits']['discovery'];root=D.OUT/'qwen14/discovery-native'
    k=np.load(root/'kpre.npy',mmap_mode='r');early=np.load(root/'early.npy',mmap_mode='r');L=k.shape[2];groups={};saved=[];start=time.monotonic()
    widths={chunk:max(len(r['tokens']) for r in rows[chunk:chunk+128]) for chunk in range(0,len(rows),128)}
    for i,r in enumerate(rows):groups.setdefault((r['family'],r['order']),[]).append(i)
    for gi,(key,ii) in enumerate(groups.items()):
        a=ii[0];ra=rows[a]
        for i in ii[1:]:
            r=rows[i];same_len=len(ra['tokens'])==len(r['tokens']);same_width=widths[a//128*128]==widths[i//128*128]
            category='same_length_same_width' if same_len and same_width else 'same_length_changed_width' if same_len else 'changed_length'
            for j in range(6):
                pa=ra['records'][str(j)]['value_span'][-1];pb=r['records'][str(j)]['value_span'][-1]
                assert ra['tokens'][:pa+1]==r['tokens'][:pb+1]
                ia=pa+L-len(ra['tokens']);ib=pb+L-len(r['tokens']);x=unbits(k[a,:,ia]);y=unbits(k[i,:,ib])
                rel=np.linalg.norm(x-y,axis=-1)/np.maximum(np.linalg.norm(x,axis=-1),1e-8)
                hx=unbits(early[a,ia]);hy=unbits(early[i,ib]);hr=float(np.linalg.norm(hx-hy)/max(np.linalg.norm(hx),1e-8))
                saved.append({'a':a,'b':i,'record':j,'category':category,'relative_key_error':rel.tolist(),'key_equal':[bool(np.array_equal(k[a,bi,ia],k[i,bi,ib])) for bi in range(2)],'early_relative_error':hr})
        if gi%32==0:D.log('prefix-numerics',gi+1,len(groups),start)
    summary={}
    for cat in sorted({s['category'] for s in saved}):
        rs=[s for s in saved if s['category']==cat];err=np.array([s['relative_key_error'] for s in rs]);eq=np.array([s['key_equal'] for s in rs])
        summary[cat]={'record_pair_count':len(rs),'nonidentical_by_head':(~eq).sum(0).tolist(),'mean_relative_key_error':err.mean(0).tolist(),'max_relative_key_error':err.max(0).tolist(),
            'mean_hs20_relative_error':float(np.mean([s['early_relative_error'] for s in rs]))}
    k._mmap.close();early._mmap.close();D.save('prefix-numerics-audit.json',{'summary':summary,'rows':saved,
        'scope':'Posthoc numerical diagnosis on exposed discovery. Identical token prefixes verified; subsequent suffix length changes the left-padding offset, not the declared RoPE position. No semantic attribution or gate repair.'})
    print(summary,flush=True)

if __name__=='__main__':main()
