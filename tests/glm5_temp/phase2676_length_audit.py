"""Keep length confounds separate for each external operation; no new model forward."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2676_native_mlp_delivery'


def main():
    material=RESULT/'phase2670_native_mlp_contract/material/cases.json';rows=read(material)
    axes=('family','language','unit','content_instance','form','target_index','mention_order','probe_index','polarity','mapping');summary={}
    for label,axis in [('factual_target','target_index'),('query_entity','probe_index'),('question_polarity','polarity'),('answer_mapping','mapping')]:
        groups={}
        for r in rows:groups.setdefault(tuple(r[k] for k in axes if k!=axis),{})[r[axis]]=r
        assert len(groups)==4096 and all(set(g)=={0,1} for g in groups.values());scope={};different=[]
        for g in groups.values():
            a,b=g[0],g[1];lens=[len(a['prompt_ids']),len(b['prompt_ids'])];key=a['family']+'/'+a['language'];d=abs(lens[0]-lens[1]);v=scope.setdefault(key,{'pairs':0,'equal_length':0,'maximum_length_difference':0})
            v['pairs']+=1;v['equal_length']+=d==0;v['maximum_length_difference']=max(v['maximum_length_difference'],d)
            if d:different.append({'cases':[a['case_index'],b['case_index']],'lengths':lens,'family_language':key})
        summary[label]={'pairs':4096,'equal_length':4096-len(different),'by_family_language':scope,'different_pairs':different}
    save(OUT/'analysis/operation_length_audit.json',{'summary':summary,'material_sha256':sha(material),'all_checks_passed':True,
        'scope':'Equal total token length removes this one execution-shape difference only; it does not remove lexical/semantic confounds or prove deterministic numerics. Body prefix invariance and task-boundary factualcontrast are separate controls.'})
    print({k:{p:v[p] for p in ('pairs','equal_length')} for k,v in summary.items()})


if __name__=='__main__':main()
