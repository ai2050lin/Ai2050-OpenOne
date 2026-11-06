"""Calibration revision: distinct referents and unambiguous active-entry question."""
import copy
import native_mlp_function_materials as B
from native_mlp_function_discovery import OUT as DISCOVERY
B.OUT=B.ROOT/'tests/glm5/result/native_mlp_function_v2';B.PUBLIC=B.ROOT/'research/native_mlp_function/v2';B.RUN='NMLP-FUNCTION-20261104-v2'
B.FAMILIES=copy.deepcopy(B.FAMILIES)
QUESTIONS={'conditional':'According only to the active entry, does the action require a particular detected condition?',
 'universal':'According only to the active entry, does every member of its set have a lid?',
 'location':'According only to the active entry, is its named entity a geographic place?',
 'enumeration':'According only to the active entry, does the colon introduce an enumeration of items?'}
for family,d in B.FAMILIES.items():
    d['question']=QUESTIONS[family]
    for key,suffix in [('positive','amber'),('negative','birch')]:
        d[key]=[[text.replace('{name}','{name}_'+suffix) for text in group] for group in d[key]]
ROOT=B.ROOT;OUT=B.OUT;PUBLIC=B.PUBLIC;RUN=B.RUN;MODEL=B.MODEL;SEED=B.SEED;read=B.read;save=B.save;FAMILIES=B.FAMILIES
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'discovery.json',read(DISCOVERY/'discovery.json'))
    B.build()
