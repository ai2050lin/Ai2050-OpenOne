"""Independent confirmation and fixed-ruler/native continuity campaign."""
from rdc_mechanism_common import *
HISTORY=CAMPAIGN
CAMPAIGN=RESULT/'rdc_continuity_campaign_20260909'

def announce(run,**kw):
    save(CAMPAIGN/run/'status.json',dict(run_id=run,updated_at=stamp(),**kw))

def events(run,kind,**kw):
    path=CAMPAIGN/run/'events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    n=sum(1 for _ in path.open(encoding='utf-8')) if path.exists() else 0
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(cursor=n+1,run_id=run,kind=kind,timestamp=stamp(),**kw),ensure_ascii=False)+'\n')

def freeze_contract():
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    if not (CAMPAIGN/'memo_prefix.json').exists():immutable(CAMPAIGN/'memo_prefix.json',{'bytes':memo.stat().st_size,'sha256':sha(memo)})
    immutable(CAMPAIGN/'plan.json',{'phases':[
      {'phase':2699,'question':'Do old frozen readers survive fresh entities, formulations and relation combinations under matched shape?','tasks':['1024 new balanced bilingual cases in8 families','natural versus globally padded full-coordinate audit','frozen old readers then grouped new controls']},
      {'phase':2700,'question':'What changes under one fixed readout direction, and which native factors reproduce or predict those changes?','tasks':['all37 checkpoints with same ruler','full9728 gate/up/down conditional factors','held-out full unit/state prediction and simple baselines']},
      {'phase':2701,'question':'Can available-prefix states predict current output decisions across generation and task conditions?','tasks':['cached natural generation on320 prefixes','actual next-token shortlist and EOS targets','stage/current-history controls and cross-stage tests']},
      {'phase':2702,'question':'Which surviving results justify automatic next extension?','tasks':['independent replication or serial scale pilot chosen on information value','client and data provenance audit','complete append-only delivery']}],
      'execution_budget':{'cuda_models_concurrent':1,'capture_case_ceiling':1024,'generation_prefix_ceiling':320,'max_generated_tokens':8,'pilot_cases':16,'pilot_max_seconds_per_case':60,'minimum_free_bytes':8*1024**3,'raw_estimate_ceiling_bytes':40*1024**3},
      'route_decisions':'No forced orthogonal pure-semantic dictionary, no assumed XOR locus, no predefined shape-caused reasoning failures. Scale pilot only after actual CPU/offload cost check; no unlimited loop.',
      'evidence_levels':['observation','prospective fixed-reader confirmation','native arithmetic identity','held-out computation prediction','native intervention (not required)']})
    corrections=[
      ['U/V','Annotated lexical/participant spans, not global mean and local difference.'],
      ['t/q','t=support of fixed statement; q=query polarity; external t XOR q is not located native XOR.'],
      ['H24','126/128 is strong limited readability, neither perfect nor a unique logical compilation centre.'],
      ['output ledger','Include H35 dot v and each rounding term; v conditions on observed final norm. Accounting is not prospective prediction.'],
      ['generation','Old target stayed Yes-No at every step; it did not test punctuation/EOS prediction or disjoint parameter populations.'],
      ['shape cause','RMSNorm averages hidden coordinates; causal source mask excludes future tokens. Kernel-level origin of shape-dependent differences was not localized.'],
      ['90 percent','No such measured percentage; withdraw.'],
      ['matched shape','Does not equalize content, semantic role, actual positions or establish pure semantics.'],
      ['largest units','Contribution rank does not define unique semantic gears; all coordinates and units retained.'],
      ['cross-layer ruler','Separately fitted probes cannot alone be stitched into computation flow.'],
      ['knowledge chains','H24-to-new-U relay, edible subspace and shape-caused multihop failure were invented examples, not observations.'],
      ['theory','Keep RDC name; manifold, orthogonal subspaces, universal closure and no-need-for-new-math remain unproved.'],
      ['bilinear expansion','SwiGLU factor expansion is an ideal-arithmetic identity; input-dependent tensor is not a globally bilinear model.'],
      ['independence','Old reuse stress tests are not fresh confirmation; new frozen protocol explicitly separates old readers and new training.']]
    immutable(CAMPAIGN/'review.json',{'timestamp_policy':'Immutable review before new captures','corrections':corrections,'sources':[str(HISTORY/'b_relations/result.json'),str(HISTORY/'c_generation/result.json'),str(HISTORY/'d_generalization/result.json')],
      'attachments':[{'path':p,'sha256':sha(p)} for p in [r'C:\Users\Admin\.codex\attachments\e2a5d577-6fc4-4196-a41b-2bbc87e540c5\pasted-text.txt',r'C:\Users\Admin\.codex\attachments\e4857815-e022-42e9-92f9-d6b151b05551\pasted-text.txt']]})

