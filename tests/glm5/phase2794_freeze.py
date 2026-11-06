"""Freeze before any confirmatory outputs. Failed calibration versions retained."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from shared.dist_protocol.mechanism_trace import digest,file_sha
from tools.power_analysis import check_phase
def main():
    pilot=json.loads((P.OUT/'pilot-result.json').read_text(encoding='utf-8'));assert pilot['passed'],'discovery calibration failed'
    c=json.loads((P.OUT/'formal-corpus.json').read_text(encoding='utf-8'));assert len(c['items'])==704
    power=check_phase(.2,64,p_d=.3,alpha=.05);assert power['pass']
    deep=[x['id'] for x in c['items'] if (x['family_id'] in [e['id'] for e in c['entities'][:2]] and x['split']=='unseen_combination' and x['frame']==2)
          or (x['family_id']==c['entities'][0]['id'] and x['split']=='length_panel')]
    assert len(deep)==8
    d={'schema':'atomic_arbitration_prereg.v1','kind':'real','run_id':P.RUN,'created_at':time.time(),'model_snapshot':c['model_snapshot'],
        'corpus_sha256':digest(c),'reference_source':P.SOURCE,'reference_sha256':c['reference_file_sha256'],
        'calibration':{'scope':'only16 discovery entities,F0/F1; previous v1/v2/v3 failures retained','passed':pilot['passed'],'pilot_sha256':file_sha(P.OUT/'pilot-result.json')},
        'scope':'single-entity atomic-number arbitration with an explicit verified reference card AND a contradictory fictitious record in both scopes; NOT closed-book factual recall',
        'behavior_gate':'full greedy generated integer equals the complete number, max4 new tokens; each unseen-combination and unseen-entity scope accuracy>=.80; do not discard individual errors',
        'splits':{'discovery':'64 familiar entities x2 frames x2 scopes,1024tokens,256inputs','unseen_combination':'same64 entities,new number assignments andnew source frames2/3,256inputs',
            'unseen_entity':'32 disjoint entities,frames2/3,128inputs','length_panel':'first16 discovery entities,F0,512/2048tokens,64inputs'},
        'H2':{'power':power,'criterion':'on64 familiar entities,F2,1024,bothscopes; joint32 MLP activation removal reduces full-integer accuracy>=.20; loss versus same-block/RMS-matched random32>=.10 and entity bootstrap95% lower>0; behavior gate required'},
        'selection':'discovery only; select32 native MLP units and8heads in blocks18..35 by mean direct expected-first-digit minus opposing-first-digit contribution; actual final RMS fixed. This selects answer-support candidates, NOT scope-specific semantic units.',
        'doses':[0,.25,.5,1],'interventions':'multiply selected final-query activations by1-alpha at every actual generated step; head8 andsame-blockrandom8 secondary; originalweights unchanged',
        'prediction':{'status':'predeclared heldout predictive assessment, no assumed power for continuous KL/MSE','output':'full-vocabulary NEXT token KL on256 unseen-combination inputs; mixture of discovery reality-memory nodes and document-first-digit prototypes, scope weights fit ondiscovery only; cached model not called at prediction time',
            'criterion':'>=10% KL reduction versus scope-blind mixture, entity cluster bootstrap lower of KL gain>0, behavior gate required',
            'internal':'discovery-selected block; Ridge lambda1, entity-by-scope and each explicitly provided source-first-digit-by-scope one-hots; predict all9728 MLP units and2560 next hidden coordinates; compare entity/scope empiricalmean, report raw/standardizedMSE',
            'internal_criterion':'>=5% standardizedMSE reduction and paired entity bootstrap lower>0; no test-state/logit or expected answer label as input; source values are explicitly present in user text',
            'limit':'familiar-entity new combinations; no graph-only new-entity prediction claim, no complete-integer generation prediction claim'},
        'native':'all37 hidden layers and36 blocks complete G/V/A,HeadZ at entity/final; deep8 fulltoken HiddenState/QKV, actual cached generation steps; raw values notTopK',
        'deep_ids':deep,'capability':'unconditionally run new32-MLP alpha1 onthe repeated frozen2200 keyed pool,2pp NI andpaired simultaneousCP95%;140reference-only excluded',
        'capability_pool_sha256':file_sha(P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'),
        'limits':['only one attribute andone localphysical node','author-written repeated long background','scope is system-level; changes early states too','world reference is supplied, not memory-only arbitration','discovery16 entities used forcalibration','single final query pathway andfirstdigit DLA selection','no Goodfire native-neuron completeness claim']}
    P.save(P.PUBLIC/'prereg.research.json',d,True);print({'run_id':P.RUN,'n':704,'deep':8,'power':power})
if __name__=='__main__':main()
