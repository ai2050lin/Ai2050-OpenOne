"""Frozen coordinate-envelope rule tested on1024 new internal fields, streamed all coordinates."""
import gc,itertools
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr,tensor_output
from phase2636_precision_engine import load_precision
from phase2643_matched_dual_adjoint_engine import MATERIAL
from phase2647_matched_operation_delivery import OUT,MAPS

DEST=OUT/'third_field_confirmation'
UNITS=tuple(range(2,10))

class Capture:
    def __init__(self,model):
        self.hooks=[];self.reset()
        for block in model.model.layers:
            self.hooks.append(block.register_forward_hook(self.hidden));self.hooks.append(block.mlp.down_proj.register_forward_pre_hook(self.mlp))
    def reset(self):self.h=[];self.a=[]
    def hidden(self,m,inp,out):self.h.append(arr(tensor_output(out)[0,self.pos]))
    def mlp(self,m,inp):self.a.append(arr(inp[0][0,self.pos]))
    def close(self):
        for h in self.hooks:h.remove()

@torch.inference_mode()
def run():
    if (DEST/'analysis/completion.json').exists():raise RuntimeError('third set already tested')
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['unit'] in UNITS]
    assert len(cases)==1024 and all(r['field_set']=='behavior_only' for r in cases)
    contract={'timestamp':datetime.now().astimezone().isoformat(),'units':UNITS,'cases':[r['case_id'] for r in cases],
        'selection':'all8families x2languages x8new entity-pairs x2forms x2targets x2orders; no behavioral success selection',
        'frozen_rule':'target RMS>order RMS AND target RMS>form RMS, strictly, separately per coordinate/group; signed version requires nonzero third target mean sign equal initial group mean',
        'discovery_artifact_sha256':sha(OUT/'field/allcoordinate_response_envelopes.npz'),'initial_maps':[{'path':str(p),'sha256':sha(p)} for p in sorted((MAPS['initial']/'field').glob('*_fullcoordinate_maps.npz'))],
        'validation_boundary':'Internal fields for these units have not been captured before;4096 natural BF16 behavior includes them. Prospective INTERNAL FIELD confirmation, not an entirely unseen behavioral benchmark.',
        'precision':'same-value FP32 nonquantized; full3anchor coordinates, all37h and36MLP, no donor, no weight edits',
        'storage':'stream all coordinates into complete pergroup contrast means/RMS; no percase raw field files persisted; third fields not promised as alltoken archive'}
    save(DEST/'protocol/frozen.json',contract);model,info=load_precision('fp32');save(DEST/'protocol/model.json',info);cap=Capture(model)
    table={(r['family'],r['language'],r['unit'],r['form'],r['target_index'],r['mention_order']):r for r in cases}
    totals={};row_summaries=[];n_done=0;DEST.joinpath('field').mkdir(parents=True,exist_ok=True)
    for fam,lang in sorted({(r['family'],r['language']) for r in cases}):
        sums={};counts={}
        for unit in UNITS:
            cube={}
            for form,v,o in itertools.product((0,1),repeat=3):
                r=table[(fam,lang,unit,form,v,o)];cap.pos=[r['entity_spans']['a'][-1],r['entity_spans']['b'][-1],len(r['prompt_ids'])-1];cap.reset()
                em=model.get_input_embeddings()(torch.tensor([r['prompt_ids']],device='cpu')).to('cuda:0')
                result=model.model(inputs_embeds=em,use_cache=False)
                cube[(form,v,o)]={'h':np.stack([arr(em[0,cap.pos])]+cap.h).astype('float64'),'mlp':np.stack(cap.a).astype('float64')}
                n_done+=1;del result,em
            for kind,axis in [('target',1),('order',2),('form',0)]:
                for a in itertools.product((0,1),repeat=3):
                    if a[axis]!=0:continue
                    b=list(a);b[axis]=1;b=tuple(b)
                    for metric in ('h','mlp'):
                        d=cube[a][metric]-cube[b][metric];key=kind+'__'+metric
                        if key not in sums:sums[key]=[np.zeros_like(d),np.zeros_like(d)];counts[key]=0
                        sums[key][0]+=d;sums[key][1]+=d*d;counts[key]+=1
            del cube
        maps={k+'__'+suffix:(sums[k][0]/counts[k] if suffix=='mean' else np.sqrt(sums[k][1]/counts[k])).astype('float32') for k in sums for suffix in ('mean','rms')}
        with np.load(MAPS['initial']/f'field/{fam}_{lang}_fullcoordinate_maps.npz') as init:
            for metric in ('h','mlp'):
                t=maps['target__'+metric+'__rms'];o=maps['order__'+metric+'__rms'];f=maps['form__'+metric+'__rms']
                dom=(t>0)&(t>o)&(t>f);mean=maps['target__'+metric+'__mean'];ref=init['target__'+metric+'__mean']
                signed=dom&(mean!=0)&(ref!=0)&(np.sign(mean)==np.sign(ref))
                for suffix,value in [('dominant',dom),('dominant_same_sign',signed)]:
                    key=metric+'__'+suffix
                    if key not in totals:totals[key]=np.zeros_like(value,dtype='int16')
                    totals[key]+=value
                row_summaries.append({'group':fam+'/'+lang,'metric':metric,'dominant_fraction_by_layer_position':dom.mean(-1).tolist(),'dominant_same_sign_fraction_by_layer_position':signed.mean(-1).tolist()})
        np.savez(DEST/f'field/{fam}_{lang}_fullcoordinate_maps.npz',**maps)
        save(DEST/'analysis/progress.json',{'cases':n_done,'total':len(cases)});print('prospective envelope',n_done,'/',len(cases),flush=True)
    cap.close();del model,cap;gc.collect();torch.cuda.empty_cache();np.savez(DEST/'field/coordinate_group_coverage.npz',**totals)
    summary={}
    with np.load(OUT/'field/allcoordinate_response_envelopes.npz') as prior:
        for metric,l in [('h',36),('mlp',35)]:
            a=prior[metric+'__both_dominant'][l,2]==16;s=prior[metric+'__both_dominant_same_sign'][l,2]==16
            t=totals[metric+'__dominant'][l,2]==16;ts=totals[metric+'__dominant_same_sign'][l,2]==16
            summary[metric]={'frozen_amplitude_candidates':int(a.sum()),'confirmed_all16_amplitude':int((a&t).sum()),'frozen_signed_candidates':int(s.sum()),
                'confirmed_all16_signed':int((s&ts).sum()),'third_all16_amplitude_total':int(t.sum()),'third_all16_signed_total':int(ts.sum())}
    checks={'all1024_untouched_internal_fields':n_done==1024,'all16_family_language_groups':len(row_summaries)==32,'same28_weight_values':all(info['all28_weight_values_exact'].values()),
        'no_percase_raw_fields':not list(DEST.rglob('case_*.npz'))}
    save(DEST/'analysis/per_group_results.json',row_summaries);result={'summary':summary,'checks':checks,'all_checks_passed':all(checks.values()),'status':'third internal-field validation completed, not semantic closure'}
    save(DEST/'analysis/completion.json',result);print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':run()
