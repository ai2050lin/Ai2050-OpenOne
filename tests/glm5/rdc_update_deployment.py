"""Original in-memory BF16 checkpoints with one reversible declared MLP update."""
from rdc_update_common import *

class Deployer:
    def __init__(self,model):
        self.model=model;self.weights={b:dict(model.model.layers[b].mlp.named_parameters()) for b in (16,35)}
        self.original={b:{k:v.detach().clone() for k,v in w.items()} for b,w in self.weights.items()};self.changed=None;self.block=None

    def reset(self):
        for b,weights in self.weights.items():
            for k,v in weights.items():v.copy_(self.original[b][k])

    def select(self,branch):
        import torch
        self.reset();self.changed=None
        if branch=='native':return {'branch':branch,'actual_norm':0.}
        middle=branch.startswith('middle_');b=16 if middle else 35;self.block=b
        path=BASE/'middle_training'/branch[7:]/'parameter_deltas.npz' if middle else BASE/'learning/native_deltas'/f'{branch}.npz'
        self.changed={};norm=0.
        with np.load(path) as z:
            for k,v in self.weights[b].items():
                alias=k if middle else {'gate_proj.weight':'g','up_proj.weight':'u','down_proj.weight':'d'}[k]
                changed=(self.original[b][k].float()+torch.tensor(z[alias],device=v.device)).to(torch.bfloat16)
                norm+=float((changed.float()-self.original[b][k].float()).double().square().sum());self.changed[k]=changed
        if not middle:assert abs(norm**.5/.02-1)<.005
        self.activate(True);return {'branch':branch,'block':b,'actual_norm':norm**.5,'delta_sha256':sha(path)}

    def activate(self,changed):
        if self.changed is None:return
        values=self.changed if changed else self.original[self.block]
        for k,v in self.weights[self.block].items():v.copy_(values[k])

    def restored(self):
        import torch
        return all(torch.equal(v,self.original[b][k]) for b,w in self.weights.items() for k,v in w.items())
