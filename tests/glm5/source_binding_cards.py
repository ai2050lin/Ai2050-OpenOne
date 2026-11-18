"""Native-index feature cards, including missing evidence and counterexamples."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_native import unbits

def run(model):
    root=D.OUT/model;c=D.read(root/'candidates.json');identity=D.read(D.OUT/f'{model}-identity.json')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];mass=np.load(root/'discovery-native/source_mass.npy',mmap_mode='r')
    units=np.load(root/'discovery-native/units.npy',mmap_mode='r');cards=[]
    def evidence(tag):
        e={}
        for split in ['confirm','lockbox_frame']:
            p=root/f'{split}-prediction.json'
            if p.exists():e[split]=next(x for x in D.read(p)['candidates'] if x['id']==tag)
        return e
    def causal_evidence(kind):
        result={}
        for split in ['confirm','lockbox_frame']:
            path=root/f'{split}-causal.json'
            if path.exists():
                record=D.read(path);arms=['correct_edge','wrong_edge','sparse_qk','random_qk'] if kind=='head' else ['mlp_selected','mlp_matched']
                result[split]={'artifact_sha256':D.sha(path),'scope':'group intervention; not isolated feature',
                    'arms':{arm:record['summary'][arm] for arm in arms if arm in record['summary']},'skipped_invalid_controls':record.get('skipped_invalid_controls',[])}
        return result
    def residual_cases(tag,method):
        result={}
        for split in ['confirm','lockbox_frame']:
            path=root/f'{split}-predictions.npz'
            if path.exists():
                with np.load(path) as data:
                    err=np.square(data[f'{tag}_{method}_write']-data[f'{tag}_actual']).sum(1);rs=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
                    result[split]=[{'id':rs[i]['id'],'component_squared_error':float(err[i])} for i in np.argsort(err)[-3:]]
        return result
    for x in c['heads']:
        b,h=x['block'],x['head'];tag=f'B{b}H{h}';a=mass[:,b,h];target=np.array([r['target_record'] for r in rows]);correct=a[np.arange(len(rows)),target]
        bad=np.flatnonzero(a[:,:4].argmax(1)!=target).tolist()
        if model=='qwen14':numerator=sorted(set(x['keep'])|{(j+64)%128 for j in x['keep']})
        else:numerator=sorted(set(x['keep'])|{j^1 for j in x['keep'] if j<64})
        cards.append({'id':f'{model}:{identity["model_snapshot"]}:{tag}:postRoPE:source_binding_v1','kind':'native_attention_candidate','native_index':x,
            'discovery_contexts':[{'id':rows[i]['id'],'content':rows[i]['content'],'correct_record_mass':float(correct[i])} for i in np.argsort(correct)[-3:]],
            'counterexamples_discovery':[rows[i]['id'] for i in bad[:12]],'counterexample_count':len(bad),
            'parameters':{'post_RoPE_coordinates':x['keep'],'Q_numerator_rows':[h*128+j for j in numerator],'K_numerator_rows':[x['kv_head']*128+j for j in numerator],
                'V_rows':list(range(x['kv_head']*128,(x['kv_head']+1)*128)),'O_columns':list(range(h*128,(h+1)*128)),
                'qualification':'post-RoPE coordinates mix paired pre-RoPE rows (Qwen halves; GLM adjacent pairs in rotary half). Qwen Q/K RMSNorm denominator depends on all128 head coordinates.32 dot-product coordinates do not mean only32 weight rows were needed; no isolated scalar cause claim.'},
            'heldout_evidence':evidence(tag),'causal_scope':'all four candidates jointly; no claim of per-head necessity',
            'causal_results':causal_evidence('head'),'largest_heldout_residual_cases':residual_cases(tag,'selected32'),
            'upstream':'actual prefix K/V and last-query Q at this block','downstream':'original O projection and subsequent original blocks',
            'unembedding_readout':'no separate static directional readout; end-to-end full-vocabulary intervention KL recorded',
            'evidence_level':'candidate natural computation + heldout component prediction + group intervention if present; no automatic formal promotion'})
    for x in c['mlps']:
        b=x['block'];tag=f'B{b}MLP';a=unbits(units[:,b]);sel=x['selected'];energy=np.square(a[:,sel]).sum(1);unitstats=[]
        for j in sel:unitstats.append({'unit':j,'mean':float(a[:,j].mean()),'std':float(a[:,j].std()),'positive_fraction':float((a[:,j]>0).mean())})
        cards.append({'id':f'{model}:{identity["model_snapshot"]}:{tag}:SwiGLU:source_binding_v1','kind':'native_MLP_unit_group','block':b,
            'units':unitstats,'typical_contexts':[{'id':rows[i]['id'],'content':rows[i]['content']} for i in np.argsort(energy)[-3:]],
            'parameters':{'gate_and_up_rows':sel,'down_columns':sel,'GLM_note':'GLM fused gate_up has gate row j and up row j+intermediate_size'},
            'equation':'write_hat=discovery_mean_write + W_down[:,S](SiLU(G_S)*V_S - discovery_mean_A_S)',
            'heldout_evidence':evidence(tag),'causal_scope':'both selected MLP blocks jointly,256units perblock; cannot infer individual neuron necessity',
            'causal_results':causal_evidence('mlp'),'largest_heldout_residual_cases':residual_cases(tag,'selected256'),
            'matched_control_valid':x['matched_control_valid'],'matched_rms_ratio':x['matched_rms_ratio'],
            'upstream':'actual normalized same-layer post-attention hidden state; original gate/up projections',
            'downstream':'original Down and later blocks','unembedding_readout':'no separate static unit interpretation; end-to-end full-vocabulary intervention KL if present',
            'counterexamples':'heldout reconstruction residuals retained in predictions arrays; no semantic naming of units',
            'evidence_level':'variance-selected native group, not an independently learned SAE feature'})
    D.save(f'{model}/feature-cards.json',{'cards':cards,'candidate_sha256':D.sha(root/'candidates.json'),'formal_evidence_promotion':0},model)

if __name__=='__main__':run(sys.argv[1])
