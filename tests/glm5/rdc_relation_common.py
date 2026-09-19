"""Natural-prefix relation dynamics: bounded, append-only, full native-coordinate evidence."""
import re
import shutil
from rdc_feature_common import ROOT,RESULT,Path,np,time,json,stamp,sha,read,save,immutable,bits,unbits,npz

BASE=RESULT/'rdc_relation_dynamics_20260910'
PREVIOUS=RESULT/'rdc_prefix_atlas_20260910'
FLOOR=8*1024**3
CEILING=640*1024**2
RELATIONS=('nsubj','obj','iobj','obl','nmod','amod','advmod','advcl','acl','conj','compound','case','mark','det')
UPOS=('ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X')


def usage():return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def guard(expected=0):
    assert usage()+expected<CEILING,('New640MiB allocation',usage(),expected)
    assert shutil.disk_usage(ROOT).free-expected>FLOOR,('Physical8GiB reserve',shutil.disk_usage(ROOT).free,expected)


def status(name,**kw):save(BASE/name/'status.json',{'timestamp':stamp(),**kw})


def canonical(text):return re.sub(r'\W','',re.sub(r'\d+(?:[.,]\d+)*','NUM',text.casefold()))


def normalized(text):return re.sub(r'\W','',text.casefold())


def construction(row):
    """Explicit exact delexicalized whole-sentence skeleton, not every semantic family."""
    return '|'.join(f'{w["upos"]}:{w["relation"]}:{int(np.sign(w["head"]-w["id"])) if w["head"] else 0}' for w in row['retrospective_ud'])


def snapshot(path):
    path=Path(path);digest=sha(path);dest=BASE/'source_snapshots'/f'{path.stem}_{digest[:16]}{path.suffix}'
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    assert sha(dest)==digest
    return {'path':str(path.relative_to(ROOT)),'sha256':digest,'snapshot':str(dest.relative_to(BASE))}


def freeze():
    if (BASE/'plan.json').exists():return
    guard(8*1024**2)
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    immutable(BASE/'memo_prefix.json',{'bytes':memo.stat().st_size,'sha256':sha(memo),'timestamp':stamp()})
    immutable(BASE/'plan.json',{'timestamp':stamp(),'phases':[
      {'phase':2715,'question':'Which full cross-coordinate relation profiles survive source/family separation and stronger controls?',
       'tasks':['Audit both attachments against original numerical artifacts and distinguish algebra from measured mechanisms.',
        'Freeze512 new bilingual natural source units including genuine adjacent English document sentences where available;128 fresh source units reserved before any model outcomes.',
        'Fit a prefix-only token-piece POS/relation posterior using only old training annotations; gold full-sentence relations remain retrospective.',
        'Capture all-token full H12/H23, plus six-position H24/H36/postnorm; no coordinate selection. Pilot gates numerical and storage reliability.',
        'All14 relation types: exact-distance versus exact-distance+POS and same-endpoint-identity control coverage; full D-by-D H12/H12,H12/H23,H23/H23 statistics with exact blockwise reconstruction.']},
      {'phase':2716,'question':'Do relation-bound and full bilinear source rules improve equal-information cross-layer and token updates?',
       'tasks':['Current-state, history-mean, content matching, POS binding, relation binding, relation-source permutation, and complete quadratic/bilinear kernels on identical partitions.',
        'Select mixing including zero-history and regularization on validation, retain matched effective-df controls. No per-template model selection.',
        'Compare full cross-coordinate layer transfer with coordinatewise baselines and rolled predicted-state input.',
        'Fit previous-state/known incoming embedding token updates; evaluate full-vocabulary validation before freezing MSE-selected and KL-selected rules.']},
      {'phase':2717,'question':'Which frozen candidates transfer to new material, native parameter computation and multi-step generation?',
       'tasks':['Capture and evaluate reserved128 new source units only after model/selection freeze.',
        'Full native L23 attention and MLP compilation from predicted H23 for every prefix source, versus actual-past hybrid and actual-state arithmetic oracle.',
        'Native natural continuation, observed-state conditional prediction, and self-fed state-only continuation with explicit information boundaries; full vocabulary and stopping/divergence logs.',
        'Sequential nonquantized scale checks only after actual CPU/GPU memory feasibility, client full-coordinate query, provenance/numerical audits and append-only delivery.']}],
      'resource_plan':{'new_output_ceiling_bytes':CEILING,'physical_floor_bytes':FLOOR,'maximum_cuda_models':1,
        'main_sources':512,'fresh_sources':128,'max_source_tokens':112,'main_fit_anchors_per_source':2,
        'retained_full_token_layers':[12,23],'retained_six_anchor_layers':[24,36,'postnorm'],
        'pilot_sources':4,'capture_limit_seconds_per_run':5400,'analysis_limit_seconds_per_script':3600,
        'next_generation_sources':64,'generation_max_new_tokens':16,'new_budget_reason':'New user-authorized integrated program; previous3GiB campaign is closed and immutable. Reuse original corpus and fields, estimate complete new program before main capture.'},
      'full_coordinate_policy':'Every native coordinate and every requested cross-coordinate pair is retained or exactly reconstructible from retained native fields and full pair indices; no low-rank/PCA/Top-K mechanism definition. Matrices computed blockwise without materializing per-example outer-product tensors.',
      'selection_scope':'Source documents, exact numerical boilerplate, and explicit whole-sentence delexicalized UD skeletons excluded across chosen units. This is not an exhaustive semantic paraphrase/composition split.',
      'observational_priority':True,'same_goal_continuation':'After integrated delivery, continue only a bounded meaningful same-goal extension supported by evidence and actual resources; do not loop through trivial variants.',
      'not_assumed':['Pure second-order semantic gears','Sufficiency of one HiddenState','Failure caused by rank decay','Cross-model isomorphism','Needlessness of linear probes','Universal language closure or AGI']})


def old_material():
    return read(PREVIOUS/'material_stratified.json')+read(PREVIOUS/'confirmation_material.json')+read(PREVIOUS/'full_source_history/fresh_material.json')


def rows(fresh=False):return read(BASE/('fresh_material.json' if fresh else 'material.json'))


def load_field(row,fresh=False):
    with np.load(BASE/('fresh' if fresh else 'main')/f'fields/{row["sample_id"]}.npz') as z:return {k:z[k] for k in z.files}
