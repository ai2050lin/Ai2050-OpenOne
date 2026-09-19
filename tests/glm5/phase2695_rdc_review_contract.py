"""Attachment audit and executable campaign scope, before new tests."""
import shutil
from rdc_mechanism_common import *

def main():
    a=Path('C:/Users/Admin/.codex/attachments/9c0407c0-a216-44c9-b5c9-230367d7bb84/pasted-text.txt')
    b=Path('C:/Users/Admin/.codex/attachments/08a0cd23-25b5-4d6b-9887-ab7e83d5dcdb/pasted-text.txt')
    r=read(PREVIOUS/'s2pilot/result.json');audit=read(PREVIOUS/'s2pilot/confirmation_review.json')
    chosen=next(x for x in r['results'] if x['model_id']=='word__family__H12__A1_linear')
    assert chosen['accuracy']==495/512 and len(read(PREVIOUS/'s2pilot/material.json'))==512
    repairs=[
      ['512 new lexical entries','64 new bilingual entries,512 condition/language realizations.'],
      ['S0 proved every tool works','Full U/V/C cubic calibration failed; restricted-block diagnostic was post-result.'],
      ['No absolute semantic unit can exist','Only the tested global same-sign criterion failed; no universal nonexistence result.'],
      ['A linear concept manifold established','Task-defined family readability established; manifold, dimensionality and unique subspace not established.'],
      ['Concept and logic use disjoint coordinates','S1 joint answer was degenerate; different behavior/readability metrics do not prove disjoint mechanisms.'],
      ['Top100 ledger coordinates are causal gears','No coordinate subset is certified; retain full coordinates and explicit noncausal reader identity.'],
      ['Pure semantic dictionary through orthogonality','Orthogonal/disjoint features are modeling constraints, not tests of semantic purity.'],
      ['MLP(H_l) added alongside attention','Actual sequential pre-norm MLP reads normalized post-attention residual, not H_l directly.'],
      ['Value parameter change passes through softmax directly','At its own attention site V changes payload; P depends on Q/K. Later-state effects may change later routing.'],
      ['Knowledge-chain failure is rotation/rounding accumulation','Unmeasured hypothesis; neither inferred nor encoded as expected result.'],
      ['Rename RDC and declare high-dimensional projection theory','Retain RDC; decomposition identities are not a new language mechanism theorem.'],
      ['One successful lock/rescue closes language encoding','An intervention could support a scoped causal claim, not all language or AGI.'],
      ['No new mathematics is needed','Current evidence neither requires nor rules out new mathematics.']]
    review={'timestamp':stamp(),'status':'reviewed','attachments':{str(p):sha(p) for p in (a,b)},
      'verified_latest':{'s2_cases':512,'H12_A1_correct':495,'s1_joint_target_degenerate':True,
        'readout_ledger_not_native_logit':True,'actual_generation':audit['generation_audit']},
      'kept':['Frozen full-coordinate family readability on new lexical entries and paraphrases.',
          'Exact algebraic reader ledger, actual native-parameter access, and observed condition-specific variation.',
          'Reuse existing fields and connect W_down,gate/up,input coordinates before broad new scans.'],
      'corrections':repairs,
      'older_phase_boundary':'2685-2690 details are historical records; not re-executed this turn or promoted to stronger claims.'}
    immutable(CAMPAIGN/'review.json',review)
    plan={'timestamp':stamp(),'phases':{
      '2695':{'name':'A: readout to actual MLP parameters','work':['1024 prior records: U/V/C and norm controls; conditional and direction-only readers.',
         'Layers11/23/35: all9728 units, true gate/up/down projection ledgers and reader-composed coefficients.',
         'Full-coordinate raw values, cancellation and rounding residuals; no TopK-based selection.']},
      '2696':{'name':'B: balanced relation/role/condition atlas','samples':512,'work':['Eight knowledge/role/grammar tasks,8 base families each,2x2 crossed conditions,2 languages.',
         'Frozen material before model runs; family-level splits; compare linear, interaction, native function and explicit symbolic reference.',
         'Whole-prefill field, actual generation and separate output/content/stop metrics.']},
      '2697':{'name':'C: native output and autoregressive continuation','max_generation_cases':128,'max_generation_steps':4,
        'work':['Natural-cache steps with full last-token coordinates; all attention sources at declared native layers.',
           'Real output rows, observed normalizer, native MLP and attention source ledgers with BF16 rounding residuals.',
           'First-answer divergence/continuation, correct and incorrect cases; scoped predictive validation, not closure by identity.']}},
      'budget':{'new_prefill_cases':512,'generation_prefixes':128,'generation_steps':4,'disk_floor_bytes':8*1024**3,
        'model':'local nonquantized Qwen3-4B BF16 CUDA, one model at a time','free_bytes_before':shutil.disk_usage(ROOT).free,
        'pilot_cases':16,'max_pilot_seconds_per_case':60},
      'client':'Extend existing /rdc with native-unit and parameter paths plus generation-step state; retain evidence if displayed.',
      'continuation':'After these complete, continue same-goal informative followup only within measured resource boundary. Q14 expansion conditional on clear new information.',
      'not_assumed':['semantic manifold','pure disjoint semantic coordinate dictionary','universal absence of local units','rotation-only multi-hop mechanism','full language closure']}
    immutable(CAMPAIGN/'plan.json',plan)
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    immutable(CAMPAIGN/'memo_prefix.json',{'bytes':memo.stat().st_size,'sha':sha(memo)})
    print('REVIEW_PLAN_FROZEN',len(repairs),plan['budget'],flush=True)

if __name__=='__main__':main()
