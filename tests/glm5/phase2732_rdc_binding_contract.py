"""Freeze evidence corrections and bounded integrated prospective protocol."""
from collections import Counter
from rdc_binding_common import *

def main():
    start=time.monotonic()
    if (BASE/'contract.json').exists():
        print('BINDING_CONTRACT_EXISTS', flush=True); return
    rows=gzread(LAW/'material.json.gz')
    confirm=gzread(LAW/'confirmation_material.json.gz')
    assert read(LAW/'verification/final.json')['finite_delivery_complete']
    attachments=[Path('C:/Users/Admin/.codex/attachments')/p/'pasted-text.txt' for p in
        ['5604dfd6-7a2f-43c2-be54-c9361bd4da66','468d7d3b-2a08-482f-aea1-5404d826f2d8','4979861b-6bfd-435e-a508-303779f7d3a8']]
    correction=[
      {'id':'A01','status':'retain','claim':'Actual scalar-parameter gradients and restricted continued training exist.','limit':'They do not reconstruct original pretraining or establish a unique semantic encoding.'},
      {'id':'A02','status':'narrow','claim':'Earlier relation-combination exclusion covered UD-parsed English natural windows.','limit':'QA training and unparsed CMRC had no UD absence certificate. No assertion of demonstrated leakage; universal absence was unproved. New strict fit excludes both.'},
      {'id':'A03','status':'narrow','claim':'H12 source means contain contextual information.','limit':'Order has influenced source vectors, but averaging can lose which source/position carries which relation.'},
      {'id':'B01','status':'reject_universal','claim':'Joint gate/value contribution is about 80% universally.','limit':'Old block35 GUM counterexample near .0867 versus block16 .8082; conditional observations cannot become universal constants.'},
      {'id':'B02','status':'unsupported','claim':'Coherence is the primary sculptor; random order cannot learn.','limit':'A restricted continuation comparison is not the historical training cause; label/difficulty/gradient-norm controls are required.'},
      {'id':'B03','status':'unsupported','claim':'Final-layer errors prove intact memory and only a defective mouth.','limit':'Same-history KV equality is an architectural constraint, not factual or semantic correctness.'},
      {'id':'B04','status':'unsupported','claim':'Present gradients recover how knowledge was stored or prove a manifold.','limit':'Present gradients depend on current state and loss; full historical training is not identifiable from them.'},
      {'id':'B05','status':'rename_experiment','claim':'Gold-target parameter updates are zero-shot, without finetuning.','limit':'They are supervised oracle test-time updates. Report separately from a prefix-only repetition proxy and controls.'},
      {'id':'B06','status':'scope','claim':'English/Chinese/code projections establish cross-modal or cross-model isomorphism.','limit':'All three are token text in one native coordinate system. Different model widths are not directly projected.'},
      {'id':'C01','status':'retain_candidate','claim':'Fixed weights yield context-dependent effective computation across layers and tokens.','limit':'Transformer equations alone do not establish the extracted reusable language rule.'},
    ]
    resources={'timestamp':stamp(),'result_ceiling_bytes':12*1024**3,'disk_floor_bytes':12*1024**3,
      'compute_ceiling_seconds':21600,'per_process_ceiling_seconds':7200,
      'scope':'Finite integrated delivery plus one information-bearing automatic follow-up. Pilot gates expansion; ceilings are not promises to spend all resources.'}
    immutable(BASE/'resources.json',resources)
    strict=[r for r in rows if r['kind']=='natural' and r.get('retrospective_ud')]
    unaudited=[r for r in rows if r['split']=='train' and not r.get('retrospective_ud')]
    audit={'timestamp':stamp(),'prior_material_sha':sha(LAW/'material.json.gz'),
      'counts':dict(Counter(r['split']+'/'+r['cohort'] for r in rows)),
      'strict_annotated_natural_rows':len(strict),'unparsed_training_rows':len(unaudited),
      'unparsed_training_cohorts':dict(Counter(r['cohort'] for r in unaudited)),
      'schema':{k:type(v).__name__ for k,v in strict[0].items()},
      'example_ud_word':strict[0]['retrospective_ud'][0],
      'scope':'New fits on old data are retrospective discovery, not new independent confirmation. New controlled programs and connected UD selection are separately frozen.'}
    save(BASE/'prior_material_audit.json',audit)
    protocol={'timestamp':stamp(),'status':'prospective_not_executed','phases':[
      {'phase':2732,'question':'Does preserving source-position/role binding add predictive information?','tasks':[
        'audit attachments and all fit-source exclusion scope',
        'full-coordinate natural source kernels versus query/mean, with soft roles learned only from training prefixes',
        'same-source position/role permutation controls, capacity-matched ridge, grouped uncertainty',
        'freeze connected dependency combinations and multilingual executable programs before confirmation']},
      {'phase':2733,'question':'What native parameter relationships and training changes support or limit the binding rules?','tasks':[
        'factor-exact same-unit gate/up bilinear parameter relationships, all units and coordinates',
        'Alpha: full declared gradient-span comparisons and norm-matched finite updates; no top-rank selection',
        'actual middle-block16 continuation through native suffix with coherent/order controls and two seeds',
        'Beta: oracle-gold versus prefix-only repetition objective versus random/norm controls',
        'Gamma: matched English/Chinese/Python semantic-case gradient relations and projection transfer']},
      {'phase':2734,'question':'Do fixed extracted rules predict connected/new/deeper combinations and own-history behavior?','tasks':[
        'independently frozen connected-graph and program-depth confirmation',
        'sequential nonquantized Qwen4/Qwen14/GLM native capability and parameter-response comparisons',
        'first-token/full answer/stopping and collateral evaluation; learned model versus unmodified original',
        'all-coordinate client data, evidence/formula registry and integrity checks']},
      {'phase':2735,'status':'conditional_automatic_followup','question':'Resolve the strongest new binding/training discontinuity found in2732-2734, within remaining budget.'}],
      'constraints':['Natural forward observations primary; controlled programs supplementary.',
        'No donor-difference mechanism core, no PCA/Top-K coordinate or unit selection.',
        'All gold graph labels are offline training/evaluation labels, never online current-prefix features.',
        'Literal giant Kronecker tensor replaced by algebraically exact factors, not approximate compression.',
        'All pretrained checkpoints read-only; transient changes reset and saved as separate deltas.',
        'Prospective claims distinguish unseen documents, expressions, entities and connected/deeper combinations.',
        'No universal language closure, pretraining-history reconstruction or AGI claim.'],
      'pilot':'Reuse complete old sources; time exact kernel blocks and native middle-suffix backward before final scale freeze.'}
    immutable(BASE/'plan.json',protocol)
    contract={'timestamp':stamp(),'source':snapshot(Path(__file__)),'attachments':[snapshot(p) for p in attachments],
      'memo_prefix_bytes':MEMO.stat().st_size,'memo_prefix_sha':sha(MEMO),
      'prior_final_sha':sha(LAW/'verification/final.json'),'corrections':correction,
      'prior_confirmation_rows':len(confirm),'status':'audit_and_protocol_frozen_experiments_pending'}
    immutable(BASE/'contract.json',contract)
    ledger('contract',time.monotonic()-start)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
