"""Matched-source, matched-training-count, full-native-coordinate model comparison."""
from tokenizers import Tokenizer
from rdc_prefix_estimators import *
from phase2711_rdc_prefix_capture import selected_rows
OUT=CAMPAIGN/'scale_analysis'
MODEL_DIR={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def main():
    material=selected_rows('qwen14',False)
    immutable(OUT/'protocol.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),
      'source_units':64,'unit_splits':{'train':32,'validation':16,'test':16},'anchors_per_unit':2,
      'materials':[r['sample_id'] for r in material],
      'models':'Qwen3-4B, Qwen3-14B, GLM4-9B nonquantized BF16; each model refit on same32source units, no main512-data advantage.',
      'checkpoints':'Native H0,Hfloor(depth/3),Hdepth and all-prefix mean at the early checkpoint. Layer fractions and character-enclosing token alignment are functional heuristics, not index correspondence.',
      'rules':['early_linear','full_linear','graph_interaction','train_mean','copy_early'],
      'scope':'Exploratory engineering replication after Qwen4 findings; only16 held-out source units per model. Model scale, architecture, tokenizer and training are confounded.',
      'inputs':'All current early coordinates, current embedding, full-prefix early mean, safe-decoded prefix graph. No target/target norm/UD/future token.'})
    reports=[];identity=[];alignments=[]
    for run in MODEL_DIR:
        assert read(CAMPAIGN/run/'status.json')['state']=='captured'
        runtime=read(CAMPAIGN/run/'runtime.json');depth=runtime['depth'];width=runtime['width'];early=depth//3
        tok=Tokenizer.from_file(str(ROOT/'models/hf'/MODEL_DIR[run]/'tokenizer.json'))
        rows=[];pack={k:[] for k in ('h0','h12','history12','graph','hash_graph','target')};seen={};mismatch=0
        for m in material:
            r=read(CAMPAIGN/run/f'rows/{m["sample_id"]}.json')
            with np.load(CAMPAIGN/run/f'fields/{m["sample_id"]}.npz') as z:
                h=unbits(z['h']);history=z[f'H{early}_prefix_mean']
                for k in (0,3):
                    p=r['positions'][k];g=prefix_graph(tok.decode(r['prompt_ids'][:p+1],skip_special_tokens=False),p,r['language'])
                    for key,value in [('h0',h[0,k]),('h12',h[early,k]),('history12',history[k]),('graph',descriptor(g)),('hash_graph',np.zeros(549)),('target',h[depth,k])]:pack[key].append(value.copy())
                    tid=r['prompt_ids'][p]
                    if tid in seen:mismatch+=int(not np.array_equal(h[0,k],seen[tid]))
                    else:seen[tid]=h[0,k].copy()
                    rows.append({q:r[q] for q in ('sample_id','source_group','language','genre','split')}|{'anchor':k//3,'position':p,'token_id':tid,'prefix':g['observed_prefix']})
                    alignments.append({'model':run,'sample_id':r['sample_id'],'anchor':k//3,'position':p,'own_token_end':r['token_offsets'][p][1],
                      'qwen4_token_end':m['token_offsets'][m['positions'][k]][1],'own_width':width})
        data={k:np.stack(v).astype(np.float32) for k,v in pack.items()};tr,va,te=splits(rows)
        assert tuple(map(len,(tr,va,te)))==(64,32,32);assert mismatch==0
        bank=KernelBank(data,tr);save(OUT/run/'scales.json',bank.serial_scales());save(OUT/run/'rows.json',rows)
        # Inputs/targets preserved for recomputation and native-coordinate client extension.
        npz(OUT/run/'features.npz',**data);grams={}
        for name in ('early_linear','full_linear','graph_interaction','train_mean','copy_early'):
            if name=='train_mean':pred=np.broadcast_to(data['target'][tr].mean(0),data['target'][te].shape);meta={'baseline':True}
            elif name=='copy_early':pred=data['h12'][te];meta={'baseline':True}
            else:
                gram=bank.gram(name,np.arange(len(rows)),np.arange(len(rows)));grams[name]=gram.astype(np.float32)
                pred,_,meta=fit(gram,tr,va,te,data['target'],[(0,width)],OUT/run/f'models/{name}.npz')
            report,arr=errors(data['target'][te],pred,data['target'][tr],[rows[i] for i in te])
            centered=np.mean((data['target'][te]-data['target'][tr].mean(0))**2)
            reports.append({'model':run,'rule':name,'depth':depth,'early_layer':early,'native_width':width,**meta,**report,
              'mse_over_test_error_of_frozen_train_mean':report['mse']/float(centered)})
            npz(OUT/run/f'predictions/{name}.npz',prediction=np.asarray(pred,np.float32),test=te,**arr)
            print('MATCHED_SCALE',run,name,report['mse'],reports[-1]['mse_over_test_error_of_frozen_train_mean'],flush=True)
        npz(OUT/run/'grams.npz',**grams,train=tr,validation=va,test=te)
        identity.append({'model':run,'distinct_anchor_token_ids':len(seen),'repeated_identity_mismatches':mismatch})
    save(OUT/'alignment.json',alignments)
    save(OUT/'result.json',{'timestamp':stamp(),'phase':2713,'source_units_per_model':64,'train_validation_test_source_units':[32,16,16],
      'reports':reports,'embedding_identity':identity,'alignment_exact_character_end_fraction':{m:float(np.mean([r['own_token_end']==r['qwen4_token_end'] for r in alignments if r['model']==m])) for m in MODEL_DIR},
      'comparison':'Raw MSE has model-dependent scales; also report error/target-energy and error/frozen-training-mean baseline. No cross-model coordinate isomorphism or universality claim.',
      'limits':['Only16 test units, two correlated anchors each.','Different widths/depth/tokenization/pretraining; not isolated causal effect of model size.','Matching character endpoints does not create an identical cross-tokenizer next-token target.','Exploratory matched-scale re-fit, not zero-shot transfer of Qwen4 parameters.']})
    status('scale_analysis',state='complete',models=3,matched_source_units=64);guard();print('MATCHED_SCALE_COMPLETE',flush=True)


if __name__=='__main__':main()
