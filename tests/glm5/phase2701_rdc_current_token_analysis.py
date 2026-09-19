"""Earlier-state forecast of actual current native probability groups and EOS; visible-history controls."""
from rdc_continuity_common import *
from rdc_feature_extractors import fit_predict
OUT=CAMPAIGN/'g_generation'

def main():
    rows=read(OUT/'material.json');assert len(list((OUT/'prefix_commits').glob('*.json')))==320
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),'inputs':'H12/H24/H36 currentquery full2560; observed step and last input token controls','target':'fixed shortlist + other full probability mass; actualargmax category and EOS probability','fits':'A1 shared across steps, A1 separate stage, step0-only transfer; validation-only regularization','limits':'Probability regression unconstrained: report MSE, raw score argmax and out-of-range frequency; do not label regression scores calibrated probabilities.'})
    hs=[];target=[];actual=[]
    for r in rows:
      with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:hs.append(unbits(z['h'][:,0]))
      b=read(OUT/f'behavior/{r["sample_id"]}.json');target.append(b['group_probability']);actual.append(b['argmax_group'])
    h=np.stack(hs);y=np.array(target);actual=np.array(actual);meta=read(OUT/'shortlist.json');eos=meta['token_ids'].index(meta['eos_id'])
    steps=np.array([r['generation_step'] for r in rows]);historyids=meta['token_ids'];last=np.array([historyids.index(r['visible_last_token']) if r['visible_last_token'] in historyids else len(historyids) for r in rows])
    control=np.concatenate([np.eye(8)[steps],np.eye(len(historyids)+1)[last],np.eye(2)[[int(r['language']=='zh') for r in rows]]],axis=1)
    tr,va,te=[np.array([i for i,r in enumerate(rows) if r['word_split']==s]) for s in ('train','validation','test')]
    results=[]
    def evaluate(name,blocks,train,val,test,algorithm='A1_linear'):
      if not len(train) or not len(val) or not len(test):return
      score,p,params=fit_predict(blocks,y,train,val,test,algorithm,False)
      m=dict(algorithm=name,representation=name,split='prefix_heldout',target='native_current_distribution',**score,accuracy=float(np.mean(p.argmax(1)==actual[test])),eos_mse=float(np.mean((p[:,eos]-y[test,eos])**2)),out_of_range_fraction=float(np.mean((p<0)|(p>1))))
      m['by_step']=[{'step':int(s),'n':int(mask.sum()),'mse':float(np.mean((p[mask]-y[test][mask])**2)),'argmax_correct':int(np.sum(p[mask].argmax(1)==actual[test][mask])),'eos_mse':float(np.mean((p[mask,eos]-y[test][mask,eos])**2))} for s in sorted(set(steps[test])) if (mask:=steps[test]==s).any()]
      results.append(m);npz(OUT/f'predictions/{name}.npz',prediction=p,target=y[test],test_indices=test,actual_argmax=actual[test])
      npz(OUT/f'models/{name}.npz',**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
    evaluate('train_mean',[control],tr,va,te,'A0_mean')
    evaluate('visible_history',[control],tr,va,te)
    for l in (12,24,36):
      evaluate(f'H{l}_shared',[h[:,l]],tr,va,te)
      evaluate(f'H{l}_with_history',[h[:,l],control],tr,va,te)
      evaluate(f'H{l}_step0_transfer',[h[:,l]],tr[steps[tr]==0],va[steps[va]==0],te)
      for step in sorted(set(steps)):evaluate(f'H{l}_stage{step}',[h[:,l]],tr[steps[tr]==step],va[steps[va]==step],te[steps[te]==step])
    counts=[{'step':int(s),'n':int((steps==s).sum()),'shortlist_argmax_covered':int(np.sum(actual[steps==s]<len(historyids))),'native_eos_argmax':int(np.sum(actual[steps==s]==eos))} for s in sorted(set(steps))]
    prefixes=[read(p) for p in (OUT/'prefix_commits').glob('*.json')]
    save(OUT/'result.json',{'phase':2701,'timestamp':stamp(),'prefixes':320,'states':len(rows),'all_eos_prefixes':sum(p['eos'] for p in prefixes),'stage_counts':counts,'results':results,'test_states':len(te),'limits':read(OUT/'protocol.json')['limits']})
    npz(OUT/'features/current_state.npz',h=h,probabilities=y,actual_argmax=actual,steps=steps,visible_history=control)
    announce('g_generation',state='analysis_complete',completed=len(rows),total=len(rows),prefixes=320)

if __name__=='__main__':main()
