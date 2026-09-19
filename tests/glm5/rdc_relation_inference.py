"""Specialized exact evaluators of the frozen selected full-coordinate rules."""
from rdc_relation_common import *
from rdc_relation_estimators import load_model,predict,Bank
from phase2716_rdc_relation_dynamics import data,temporal_dots,temporal_as_bank
from phase2715_rdc_prefix_relations import PrefixRelations


class CurrentRule:
    def __init__(self,kind='full_quadratic'):
        self.kind=kind;self.info=read(BASE/f'rules/{kind}/result.json');self.model=load_model(BASE/f'rules/{kind}')
        self.scale=read(BASE/'rules/scales.json')['current'];material={r['sample_id']:r for r in rows()};self.train=[]
        for m in read(BASE/'rules/rows.json'):
            if m['split']=='train':self.train.append(unbits(load_field(material[m['sample_id']])['h12'][m['position']]))
        self.train=np.stack(self.train).astype(float)
        assert kind=='full_quadratic' or self.info['mix']==0,'This evaluator specializes the actual frozen zero-history rule; use full Bank for nonzero-history retraining.'
    def __call__(self,full_current_h12):
        x=np.asarray(full_current_h12).reshape(-1,2560).astype(float);dot=x@self.train.T/self.scale;gram=(1+dot)**2 if self.kind=='full_quadratic' else 1+dot
        return predict(self.model,gram)


class TemporalRule:
    def __init__(self,kind='previous_embedding_bilinear',training=None):
        self.kind=kind;self.info=read(BASE/f'dynamics/{kind}/result.json');self.model=load_model(BASE/f'dynamics/{kind}');self.scales=read(BASE/'dynamics/temporal_scales.json');self.parser=PrefixRelations()
        raw,meta=data(rows()) if training is None else training;train=np.array([i for i,m in enumerate(meta) if m['split']=='train']);self.train={k:v[train] for k,v in raw.items() if k in self.scales}
    def __call__(self,previous_h36,new_embedding,known_prefix_ids,language):
        raw={'h36':np.asarray(previous_h36).reshape(1,-1),'embedding':np.asarray(new_embedding).reshape(1,-1),'descriptor':self.parser.descriptor(known_prefix_ids,language)[None]}
        dots,_=temporal_dots(raw,scales=self.scales,other=self.train);dd,kk=temporal_as_bank(dots,self.kind)
        return predict(self.model,Bank.gram(dd,kk,self.info['mix']))[0]


def check_frozen():
    frozen=read(BASE/'frozen.json')
    for name,digest in frozen['files'].items():assert sha(BASE/name)==digest,('Frozen artifact changed',name)
    return frozen
