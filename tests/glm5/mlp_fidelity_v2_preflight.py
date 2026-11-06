"""Frozen material leakage and code/readout checks before formal collection."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    fresh=P.read(P.OUT/'fresh-documents.json');old=np.load(P.LONG/'input-tokens.npy',mmap_mode='r');rows=[(f'old:{i}',r[:1024]) for i,r in enumerate(old)]+[(f'newnatural:{i}',r['tokens'][:1024]) for i,r in enumerate(fresh['natural'])]+[(f'newcontrolled:{i}',r['tokens'][:1024]) for i,r in enumerate(fresh['controlled'])]
    rng=np.random.default_rng(P.SEED);aa=rng.integers(1,2**63,16,dtype=np.uint64)|np.uint64(1);bb=rng.integers(0,2**63,16,dtype=np.uint64);bands={};shingles=[];near=[]
    for i,(label,ids) in enumerate(rows):
        t=np.asarray(ids,np.uint64);h=np.zeros(1020,np.uint64)
        with np.errstate(over='ignore'):
            for k in range(5):h=h*np.uint64(1000003)+t[k:k+1020]
            h=np.unique(h);sig=(h[None]*aa[:,None]+bb[:,None]).min(1)
        candidates=set()
        for k in range(4):candidates.update(bands.get((k,tuple(sig[k*4:k*4+4])),[]))
        for j in candidates:
            if i<2048:continue
            overlap=len(np.intersect1d(h,shingles[j],assume_unique=True));jac=overlap/(len(h)+len(shingles[j])-overlap)
            if jac>=.85:near.append({'a':rows[j][0],'b':label,'jaccard':jac})
        for k in range(4):bands.setdefault((k,tuple(sig[k*4:k*4+4])),[]).append(i)
        shingles.append(h)
    assert not near,near
    c=P.read(P.OUT/'controlled-corpus.json');assert all(r['split']=='discovery' for r in c['items'][:2048])
    for split in ['discovery','confirm','lockbox']:
        rr=[r for r in c['items'] if r['split']==split]
        for frame in [0,1]:
            ss=[r for r in rr if r['frame']==frame];assert sum(r['background_language']=='en' for r in ss)==sum(r['background_language']=='zh' for r in ss)
    P.save(P.OUT/'material-preflight.json',{'kind':'real','run_id':P.RUN,'passed':True,'near_duplicate_candidates':near,'method':'16MinHash/4bands candidate retrieval, exact5tokenJaccard>=.85; notexhaustive all-pair guarantee','documents_checked':len(rows),'corpus_sha256':file_sha(P.OUT/'controlled-corpus.json')});print({'passed':True,'docs':len(rows)},flush=True)
if __name__=='__main__':main()
