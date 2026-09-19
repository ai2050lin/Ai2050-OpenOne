"""Complete-vocabulary validation, frozen choices, and held-out evidence (never future selection)."""
import gc
from rdc_relation_common import *
from rdc_relation_estimators import KINDS
from phase2716_rdc_relation_dynamics import TEMPORAL
from phase2712_rdc_prefix_native_probability import checkpoint


def references(meta,scope,fresh=False):
    material={r['sample_id']:r for r in rows(fresh)};actual=[];post=[];observed=[];capture=[]
    for m in meta:
        r=material[m['sample_id']];z=load_field(r,fresh);j=3*m['anchor']+int(scope=='temporal')
        actual.append(unbits(z['h36'][j]));post.append(unbits(z['postnorm'][j]));observed.append(r['prompt_ids'][m['position']+1+int(scope=='temporal')])
        capture.append(read(BASE/('fresh' if fresh else 'main')/f'behavior/{m["sample_id"]}.json'))
    return np.stack(actual),np.stack(post),observed,capture


class Readout:
    def __init__(self):
        import torch
        self.torch=torch;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cfg=read(ROOT/'models/hf/qwen3-4b/config.json');key='model.embed_tokens.weight' if cfg['tie_word_embeddings'] else 'lm_head.weight'
        self.wb=checkpoint(key).cuda();self.w=self.wb.float();self.gamma=checkpoint('model.norm.weight').float().cuda();self.eps=cfg['rms_norm_eps'];self.vocab=cfg['vocab_size']
    def logprob(self,h):
        t=self.torch;h=t.as_tensor(np.array(h),dtype=t.float32,device='cuda');return ((h*t.rsqrt(h.square().mean(-1,keepdim=True)+self.eps)*self.gamma)@self.w.T).log_softmax(-1)
    def native(self,post):
        t=self.torch;return (t.as_tensor(post,dtype=t.bfloat16,device='cuda')@self.wb.T).float().log_softmax(-1)
    def evaluate(self,pred,post,observed,meta,folder,name,scope,split,capture=None):
        t=self.torch;terms_sum=np.zeros(self.vocab,np.float64);per=[];argdiff=[];nlldiff=[]
        with t.inference_mode():
            for begin in range(0,len(pred),16):
                end=min(begin+16,len(pred));lp=self.logprob(pred[begin:end]);lq=self.native(post[begin:end]);terms=lq.exp()*(lq-lp);kl=terms.sum(-1);terms_sum+=terms.sum(0).cpu().numpy();pa=lp.argmax(-1);qa=lq.argmax(-1)
                for j,i in enumerate(range(begin,end)):
                    m=meta[i];y=observed[i];row={k:m[k] for k in ('sample_id','source_group','language','genre','anchor')}|{'KL':float(kl[j]),'argmax_agreement':bool(pa[j]==qa[j]),'predicted_argmax':int(pa[j]),'native_argmax':int(qa[j]),'predicted_observed_token_NLL':-float(lp[j,y]),'native_observed_token_NLL':-float(lq[j,y])};per.append(row)
                    if capture is not None and scope=='current':
                        cp=capture[i];k=m['anchor'];argdiff.append(int(qa[j])!=cp['native_argmax'][k]);nlldiff.append(abs(row['native_observed_token_NLL']-cp['observed_next_token_NLL'][k]))
        report={'route':name,'scope':scope,'split':split,'anchors':len(pred),'source_units':len({r['source_group'] for r in meta}),
          'KL':float(np.mean([r['KL'] for r in per])),'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in per])),
          'predicted_observed_token_NLL':float(np.mean([r['predicted_observed_token_NLL'] for r in per])),'native_observed_token_NLL':float(np.mean([r['native_observed_token_NLL'] for r in per]))}
        for group in ('language','genre'):
            report['by_'+group]={v:{'n':len(rr),'KL':float(np.mean([r['KL'] for r in rr])),'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in rr]))} for v in sorted({r[group] for r in per}) if (rr:=[r for r in per if r[group]==v])}
        if argdiff:report['native_capture_batch2_vs_reference_batch16']={'argmax_disagreement_fraction':float(np.mean(argdiff)),'observed_NLL_max_abs_difference':float(max(nlldiff))}
        assert abs(float(terms_sum.sum()/len(pred))-report['KL'])<1e-4
        save(folder/f'{split}_{scope}_{name}.json',{'summary':report,'rows':per});npz(folder/f'{split}_{scope}_{name}.npz',all_vocabulary_mean_KL_contribution=(terms_sum/len(pred)).astype(np.float32))
        print('RELATION_PROBABILITY',split,scope,name,report['KL'],report['argmax_agreement'],flush=True)
        return report
    def close(self):
        del self.w,self.wb,self.gamma;gc.collect();self.torch.cuda.empty_cache()


def freeze(result):
    choices={'current_MSE':read(BASE/'rules/result.json')['validation_MSE_winner'],'temporal_MSE':read(BASE/'dynamics/result.json')['validation_MSE_winner']}
    choices['current_KL']=min(KINDS,key=lambda k:(next(r['KL'] for r in result if r['split']=='validation' and r['scope']=='current' and r['route']==k),KINDS.index(k)))
    choices['temporal_KL']=min(TEMPORAL,key=lambda k:(next(r['KL'] for r in result if r['split']=='validation' and r['scope']=='temporal' and r['route']==k),TEMPORAL.index(k)))
    files={}
    for name in ('material.json','fresh_material.json','rules/scales.json','rules/rows.json','rules/protocol.json','rules/result.json','dynamics/rows.json','dynamics/temporal_scales.json','dynamics/result.json','dynamics/protocol.json','prefix_parser/model.npz','prefix_parser/protocol.json','prefix_parser/frozen.json'):
        files[name]=sha(BASE/name)
    for directory in ('rules','dynamics'):
        for p in (BASE/directory).rglob('*.npz'):files[str(p.relative_to(BASE))]=sha(p)
    for p in (BASE/'main/commits').glob('*.json'):files[str(p.relative_to(BASE))]=sha(p)
    source={p.name:snapshot(p) for p in [ROOT/'tests/glm5/rdc_relation_common.py',ROOT/'tests/glm5/rdc_relation_estimators.py',ROOT/'tests/glm5/phase2716_rdc_relation_dynamics.py',Path(__file__),ROOT/'tests/glm5/phase2715_rdc_prefix_relations.py',ROOT/'tests/glm5/phase2715_rdc_relation_capture.py']}
    immutable(BASE/'frozen.json',{'timestamp':stamp(),'choices':choices,'files':files,'source_code':source,
      'confirmation_sources':128,'confirmation_model_outputs_seen':False,'selection':'MSE choices from joint H23/H36 or temporal H36 validation; KL choices among these already MSE-tuned routes using complete vocabulary validation only.',
      'limits':'KL selection does not jointly optimize every mixture/ridge for KL; primary comparison is explicitly different target objectives, not a proof of a globally optimal rule.'})
    print('RELATION_RULES_FROZEN',choices,flush=True)


def main():
    out=BASE/'probability';out.mkdir(parents=True,exist_ok=True);meta=read(BASE/'rules/rows.json')
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'vocabulary':151936,
      'candidate':'Predicted raw H36 -> own RMS -> real final gamma -> actual complete FP32 readout; TF32 disabled.',
      'reference':'Saved actual native BF16 postnorm through actual BF16 head, batch16. Original capture batch2 argmax and observed NLL retained as a separate shape audit.',
      'arithmetic_floor':'Actual raw H36 with same FP32 norm/head versus BF16 reference; not a learned predictor.',
      'selection':'Validation complete-vocabulary KL chooses routes before fresh capture. Test is only evaluation. Natural following token NLL is teacher-forced, not free-generation correctness.',
      'all_coordinates_and_vocabulary_retained_or_exactly_reconstructible':True})
    rd=Readout();reports=[]
    try:
      for split in ('validation','test'):
        selected=[m for m in meta if m['split']==split]
        for scope,names,directory in [('current',KINDS,'rules'),('temporal',TEMPORAL,'dynamics')]:
            actual,post,observed,capture=references(selected,scope)
            candidates=[]
            for name in names:
                with np.load(BASE/directory/name/'predictions.npz') as z:p=z[split]
                candidates.append((name,p[:,-2560:]))
            with np.load(BASE/directory/names[0]/'model.npz') as z:mean=z['mean'][-2560:]
            candidates.extend([('training_mean',np.broadcast_to(mean,actual.shape)),('actual_H36_FP32_oracle',actual)])
            if scope=='current':
                for name in ('current','full_quadratic'):
                    with np.load(BASE/f'dynamics/layer23_36/{name}_rolled.npz') as z:candidates.append(('layer_roll_'+name,z[split]))
            for name,pred in candidates:
                cp=out/f'{split}_{scope}_{name}.json'
                reports.append(read(cp)['summary'] if cp.exists() else rd.evaluate(pred,post,observed,selected,out,name,scope,split,capture))
    finally:rd.close()
    save(out/'result.json',{'timestamp':stamp(),'reports':reports,'full_vocabulary':151936,'protocol_sha':sha(out/'protocol.json')});freeze(reports);guard(60*1024**2)


if __name__=='__main__':main()
