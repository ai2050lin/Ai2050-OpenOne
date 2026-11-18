"""Posthoc native-coordinate lineage for the B39 numerical counterfactual."""
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_binding_native_v2 import unbits

def run():
    import torch
    model='qwen14';split='confirm';root=D.OUT/model;rt=Runtime(model);timer=time.monotonic()
    fit=D.read(root/'formation-fit.json');selected=next(c['selected'] for c in fit['mlps'] if c['block']==39)
    u=np.load(root/f'{split}-native/units.npy',mmap_mode='r')
    w=rt.weight('model.layers.39.mlp.down_proj.weight','cpu')[:,selected].float()
    indices=[];fractions=[]
    for lo in range(0,len(u),64):
        a=torch.from_numpy(unbits(u[lo:lo+64,1,selected]));delta=-(a@w.T);energy=delta.square();value,idx=energy.max(1)
        indices.extend(idx.tolist());fractions.extend((value/energy.sum(1)).tolist())
    hist=np.bincount(indices,minlength=rt.dim);dominant=int(hist.argmax());u._mmap.close()
    D.save(f'{model}/{split}-matching-coordinate-lineage.json',{'posthoc':True,'model_snapshot':D.read(D.OUT/f'{model}-identity.json')['model_snapshot'],
        'block':39,'space':'MLP down_proj output / residual coordinates; not MLP intermediate unit indices',
        'original_weight':'model.layers.39.mlp.down_proj.weight','input_columns':selected,'modal_dominant_output_row':dominant,
        'modal_fraction_of_samples':float(hist.max()/len(indices)),'per_sample_dominant_row':indices,'per_sample_dominant_energy_fraction':fractions,
        'source_sha256':D.sha(Path(__file__)),'scope':'Natural-state selected-set suppression direction only. A dominant numerical coordinate is not evidence of a semantic concept or single-parameter necessity.'},model)
    D.log('qwen14/confirm/matching-lineage',len(indices),len(indices),timer)

if __name__=='__main__':run()
