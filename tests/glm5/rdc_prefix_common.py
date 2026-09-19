"""Unified prefix atlas: explicit scope, immutable material, no implicit model load."""
import re
import shutil
from rdc_feature_common import ROOT, RESULT, read, save, immutable, sha, bits, unbits, npz, stamp, np, Path, time, json

CAMPAIGN = RESULT / 'rdc_prefix_atlas_20260910'
PRIOR = RESULT / 'rdc_conditional_campaign_20260910'
FLOOR = 8 * 1024**3
CEILING = 3 * 1024**3


def usage():
    return sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())


def guard(expected=0):
    assert shutil.disk_usage(ROOT).free - expected > FLOOR, '8 GiB physical reserve'
    assert usage() + expected < CEILING, '3 GiB new campaign allocation'


def status(run, **kw):
    save(CAMPAIGN/run/'status.json', dict(updated_at=stamp(), run_id=run, **kw))


def freeze():
    memo = ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    if not (CAMPAIGN/'memo_prefix.json').exists():
        immutable(CAMPAIGN/'memo_prefix.json', dict(bytes=memo.stat().st_size, sha256=sha(memo)))
    immutable(CAMPAIGN/'plan.json', {
      'phases': [
        {'phase':2711,'question':'Which full-coordinate organizations recur in one shared natural-prefix corpus?',
         'tasks':['audit all four attachments against original results and primary references',
           '512 naturally occurring English/Chinese sentence-context units; typed prefix graph and token/span identities',
           'all-position all-layer full-coordinate streaming moments; six full-coordinate anchor states per unit and 16 complete panels',
           'embedding/context identity separation; complete coordinate-pair covariance at declared checkpoints; condition and nuisance comparisons',
           'one bidirectionally linked atlas index including legacy controlled/generation reference material']},
        {'phase':2712,'question':'Can shared rules explain coordinates, layer/time updates and complete next-token distributions together?',
         'tasks':['one common train/validation/test split and one rule per algorithm, never per sentence/template',
           'coordinatewise shared cross-layer operators and full-coordinate kernel predictors with prefix-graph/interaction controls',
           'current-to-later-layer and next-input-available temporal forecasts; prefix-only graph versus retrospective annotation separation',
           'full-vocabulary KL/argmax/observed-token metrics and native parameter-constrained readout',
           'language/genre/position/identity/error and low-amplitude coordinate audits; freeze selected rules before confirmation']},
        {'phase':2713,'question':'Which organizations survive frozen new material and nonquantized model changes?',
         'tasks':['128 official held-out natural units after rule freeze, no refitting frozen Qwen4 rules',
           '64 shared natural units on Qwen14 then GLM4 after per-model pilot, model-native indices and widths',
           'full-coordinate replication not cross-model coordinate equality; independent versus exploratory labels',
           'client query, scientific figures, source/numerical integrity, append-only records and bounded same-goal continuation decision']}
      ],
      'resources':{'new_campaign_bytes':CEILING,'minimum_free_bytes':FLOOR,
        'max_cuda_models':1,'qwen4_main_units':512,'qwen4_confirmation_units':128,
        'scale_units_per_model':64,'maximum_text_tokens':96,'anchors_per_unit':2,'anchor_offsets':[0,1,2],
        'all_position_full_panels':16,'pilot_units':4,'max_capture_seconds_per_model':5400,
        'analysis_seconds':7200,'adaptive_rule':'Estimate actual pilot cost before expansion. A smaller completed scope requires an explicit reason and revised manifest; never silently call it full.'},
      'scope':'Natural EWT web genres plus Chinese GSD Wikipedia are the main new material, not all possible language/task families. Prior controlled knowledge/reasoning/grammar/translation/reordering results remain linked reference data, not independent new cases.',
      'causality':'Prefix features use only visible text. Full-sentence UD labels are retrospective annotations and never predictive inputs. Temporal prediction may use the newly observed token, never a still-future token.',
      'retention':'Retain all saved fields for indexed client queries and evidence. All other token states are streamed into full-coordinate moments; those individual raw states are not claimed retained. No PCA/Top-K core; no coordinate deletion or donor transport core.',
      'continuation':'After the integrated stages, continue same-goal research only with a specific information gain inside these finite resources. Do not equate finite delivery with a solved scientific objective.'})
    paths=[str(Path(r'C:\Users\Admin\.codex\attachments')/x/'pasted-text.txt') for x in (
      '44382d7b-3061-4b46-a989-fdc7fa3335b3','c451712b-9dcd-42a7-b9ba-6b7c62f8e578',
      'dce49ca0-e5c0-42d8-bddb-da156abc0f7e','26fc2c92-2a7c-45e9-b6fb-c81325aabc8d')]
    sources=[PRIOR/p for p in ('i_factorial/result.json','j_predictive_gates/result.json','k_long/result.json',
      'l_aligned/result.json','m_order/result.json','n_cached_attention/result.json','o_generalization/result.json',
      'p_token_conditioned/result.json','delivery_audit.json','terminal.json')]
    corrections=[
      ['shared readability','Retain limited high held-out linear readability. It does not establish a manifold, an universal dictionary, native use, or absence of lexical/static coordinates.'],
      ['numerical divergence','The first measured differing operator applies to the tested shapes and 32 numerical-audit cases, not all architectures/inputs.'],
      ['MLP prediction','Failure of the selected factor predictor to beat a direct baseline does not prove insufficient information in H12. Weighted conditional reconstruction is not a causal gate discovery.'],
      ['history','History improved particular forecasts; H36 is not identical to a KV cache. Neither absolute history dominance nor KV causal dominance was measured.'],
      ['output order','Changed generated prefixes and source contributions were observed. Trace is not universally better, and attention mass is not a causal benefit or a measured high-SNR mechanism.'],
      ['cross model','Functional-stage alignment is useful, but topology isomorphism and absence of absolute physical specialization were not proved.'],
      ['transfer','2709 shows substantial domain transfer failure, not that every individual prediction totally fails or that template/position is the uniquely identified cause.'],
      ['2710','Current embedding did not generally fix transfer. The predicted H23 plus native Q/K/V route remains an exploratory candidate, not a confirmed general mechanism.'],
      ['unmeasured examples','Apple/eat edible-plant gates, multi-hop KV congestion, SNR thresholds, and guaranteed repairs are teaching hypotheses or invented mechanisms, not recorded tests.'],
      ['native formula','Q is layer-specific normalized/projected H with architecture-specific Q/K norm and RoPE, not raw embedding or unconstrained early H. Residual, MLP, normalization, current K/V, masking and finite precision cannot be omitted.'],
      ['cache','KV_Encode depends on complete prefix and layer; a logical hop in prefill is not necessarily a generation step or a cache rewrite.'],
      ['probe and paradigm','Do not declare all probes dead or a paradigm fatally disproved by phase count. Integrate observational and predictive comparisons with native constraints.'],
      ['mathematics and brain','No new theorem, universal pure semantic gear, arbitrary-context closure, brain mechanism or AGI follows from these results. New mathematics is neither required nor ruled out.'],
      ['plan integration','A/D typed-prefix shared-library proposal is the main plan. B native-input tools are retained as constraints. C SNR, cache permutation and universal isomorphism claims are not mandatory truth-oriented endpoints.']]
    immutable(CAMPAIGN/'review.json', {'attachments':[{'path':p,'sha256':sha(p)} for p in paths],
      'original_sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sources if p.exists()],
      'corrections':corrections,'method_references':[
        'https://arxiv.org/abs/2406.03707','https://aclanthology.org/D19-1275/',
        'https://arxiv.org/abs/2002.05202','https://arxiv.org/abs/2104.09864',
        'https://www.transformer-circuits.pub/2022/toy_model/',
        'https://www.nature.com/articles/s41593-022-01026-4'],
      'reference_limits':'Predictive sufficient-statistics results have stated model/data assumptions; probing controls and architecture equations are methodological references. Toy superposition and brain correlation do not establish this project\'s proposed mechanism.'})


LEXICONS = {
 'negation':r'\b(?:not|no|never|neither|without|cannot)\b|不|沒|無|未|非',
 'contrast':r'\b(?:but|although|however|yet|whereas)\b|但是|然而|雖然|不過',
 'cause':r'\b(?:because|therefore|thus|since|hence)\b|因為|因此|所以|導致',
 'conditional':r'\b(?:if|unless|whether|otherwise)\b|如果|若|是否|否則',
 'temporal':r'\b(?:then|before|after|when|while|during)\b|之後|之前|當時|後來|同時',
 'conjunction':r'\b(?:and|or|also|both)\b|以及|並且|或者|與|及',
 'pronoun':r'\b(?:i|you|he|she|it|we|they|him|her|them|this|that)\b|我|你|他|她|它|這|該',
 'relation':r'\b(?:is|are|was|were|has|have|belongs|part|type|kind)\b|是|屬於|具有|包含|為',
 'modal':r'\b(?:can|could|may|might|must|should|would|will)\b|可以|可能|必須|應該|將',
 'task_cue':r'\b(?:explain|translate|write|answer|sort|compare|describe)\b|解釋|翻譯|回答|比較|描述',
 'number':r'\d+(?:[.,]\d+)*', 'question':r'[?？]', 'boundary':r'[.!。！;；]',
 'comma':r'[,，:：]', 'quote':r'["“”「」『』]', 'bracket':r'[()（）\[\]【】]'}
GRAPH_NAMES = list(LEXICONS)+['open_quote','open_bracket','characters','token_position','language_zh']


def prefix_graph(text, token_position, language):
    """Causal lexical/punctuation event graph, NOT a complete semantic parser."""
    events=[];values=[]
    for name,pat in LEXICONS.items():
        matches=list(re.finditer(pat,text,re.I));values.append(float(len(matches)))
        events.extend({'type':name,'span':[m.start(),m.end()],'text':m.group()} for m in matches)
    quote_open=(text.count('"')%2)+max(0,text.count('“')-text.count('”'))+max(0,text.count('「')-text.count('」'))
    bracket_open=sum(max(0,text.count(a)-text.count(b)) for a,b in [('(',')'),('（','）'),('[',']'),('【','】')])
    values += [float(quote_open),float(bracket_open),float(len(text)),float(token_position),float(language=='zh')]
    events.sort(key=lambda e:(e['span'][0],e['type']))
    return {'observed_prefix':text,'events':events,'feature_names':GRAPH_NAMES,'features':values,
      'edges':[{'type':'observed_order','source':i,'target':i+1} for i in range(len(events)-1)],
      'unknown':['resolved word senses','semantic roles','coreference','negation scope','unfinished relationships'],
      'status':'prefix_only_cue_graph_not_gold_semantic_graph'}
