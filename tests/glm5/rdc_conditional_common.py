"""Prospective conditional-transfer campaign; no implicit model execution."""
from rdc_mechanism_common import *
OLD=RESULT/'rdc_continuity_campaign_20260909'
CAMPAIGN=RESULT/'rdc_conditional_campaign_20260910'


def announce(run,**kw):
    save(CAMPAIGN/run/'status.json',dict(run_id=run,updated_at=stamp(),**kw))


def freeze():
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    if not (CAMPAIGN/'memo_prefix.json').exists():
        immutable(CAMPAIGN/'memo_prefix.json',{'bytes':memo.stat().st_size,'sha256':sha(memo)})
    immutable(CAMPAIGN/'plan.json',{'phases':[
      {'phase':2703,'question':'Which conditional mappings generalize after disentangling surface factors?',
       'tasks':['audit both attachments against actual2699–2702 artifacts','4096 bilingual factorial cases','512 predeclared all-token full-coordinate panels; all4096 all-layer role states','fixed/affine/shared/conditional readers and matched-capacity nuisance controls','first-divergent numerical operator audit on32 balanced cases']},
      {'phase':2704,'question':'Does predicting native gate/up factors improve genuine future-write prediction?',
       'tasks':['all-unit ordinary versus weighted conditional gates and equal-group controls','H12 predicts g/up/a using identical full-coordinate features and Gram matrices','compare derived SiLU(g)*up and real down projection against direct a/down prediction','energy-bin and full-coordinate tails; conditional tensor interpretation bounded as identity']},
      {'phase':2705,'question':'What survives longer multi-operation natural generation?',
       'tasks':['128 prefixes in8 operation families up to128 generated tokens','separate content/order/format/EOS scores','all-layer current-query full coordinates; all native units at declared layers','future full-state and full-vocabulary probability evaluation, no shortlist target']},
      {'phase':2706,'question':'Do functional-stage aligned observations survive model scale and architecture?',
       'tasks':['automatic same-goal extension after2703–2705','serial nonquantized Qwen4/Qwen14/GLM4,256 shared short prompts, up to8 tokens to first content','full-coordinate mappings and gate statistics with within-model controls','client integration, provenance audit, append-only full delivery and retention decision']}],
      'resource_boundary':{'minimum_free_bytes':8*1024**3,'new_disk_ceiling_bytes':30*1024**3,
        'main_cases':4096,'full_token_panel':512,'long_prefixes':128,'long_max_tokens':128,
        'scale_prompts':256,'scale_max_tokens':8,'max_cuda_models':1,
        'pilot_cases':16,'pilot_max_seconds_per_case':60,'scale_model_capture_seconds':5400},
      'adaptation':'Pilot before expansion. Numerical side audit32, not unbounded kernel reverse engineering. Reduce long/scale scope only after actual cost or validity failure is recorded; do not claim unexecuted tasks. Additional same-goal stages require new information and remaining bounded resources, not an infinite queue.',
      'evidence_policy':'Observe -> full-native-coordinate extraction -> grouped held-out prediction -> optional mechanism evidence. No Top-K/PCA core, no donor patch core, no single causal gate stopping rule.',
      'retention':'All-token panel for client and evidence; all4096 full-coordinate role/native data for analysis and client. No disposable raw fields retained without declared use. Final cleanup only verified reproducible unneeded temporaries.'})
    corrections=[
      ['fixed readers','Raw result.json: old H24 UVC support707/1024 in both tracks; H24 C-only support738/1024 matched,740/1024 natural. Previous2702 summary calling738 UVC was a label error. H36 UVC answer865/1024 matched remains a partial success. Failure of selected readers does not rule out all fixed maps.'],
      ['shape','Near-equal matched/natural accuracy bounds a major shape explanation on those samples; first operator and all semantic failure causes were not identified.'],
      ['readability','A probe can compute a task; high readability is not proof that the native model already implemented correct logic.'],
      ['U/V','U,V are lexical participant spans, not global/local context or universal semantic axes.'],
      ['conditional gate','Old mean sigmoid uses actual same-block g/u and more fitted groups. It is reconstruction, not early routing prediction or proven gate clusters.'],
      ['generation','H12 forecast was on short outputs and a ten-token-plus-other target. Shared cross-stage readout worked; separate stage decoders were not shown necessary.'],
      ['ledger','2701 changed directions between Yes/No, period/EOS and EOS/period; observed final norm and finite-precision terms remain required.'],
      ['cross-model','Statistical replication neither establishes nor disproves functional topology isomorphism. Width/index differences alone do not prove different programs.'],
      ['liquid theory','Liquid/flow/manifold/rotation are unmeasured metaphors; retain RDC name. No new mathematical theorem established.'],
      ['invented examples','Apple/eat H12-H24 gates, knowledge-hop KV drift, critical drift thresholds and repair outcomes were not measured; mark examples/hypotheses, not discoveries.'],
      ['KV','Cached per-layer K/V are not the sole condition: current embedding/query, position, weights, normalization and residual state also matter. Logical hops within prefill are not generation steps; old KV is not updated once per logical hop.'],
      ['scalar formula','Input-conditioned SwiGLU expression includes both Wg and Wu, sigmoid and Wd. Ideal identity is not a fixed bilinear semantic law; BF16 rounding separate.'],
      ['new mathematics','Architecture identities do not explain learned linguistic function. Neither necessity nor uselessness of new mathematics follows. Keep linear baselines; no imposed semantic orthogonality.'],
      ['research strategy','Thousands of incomplete phases do not logically prove a fatal paradigm. Reorient using transfer failures and new predictive information, not renaming.'],
      ['scope','Fifty proposed operations and distillation/repair were unexecuted plans. Use controlled validated breadth before expansion; no predetermined closure.'],
      ['historical inventory','AttachmentA historical21-row inventory is not a newly re-audited45-item result. Linked sandbox document was not attached; do not pretend to have read it.']]
    paths=[r'C:\Users\Admin\.codex\attachments\0496568b-db0b-4c91-931d-08df6918e8b8\pasted-text.txt',r'C:\Users\Admin\.codex\attachments\49be648e-4353-4240-8f2b-a6e94bb4e254\pasted-text.txt']
    sources=[OLD/p for p in ('e_confirmation/result.json','f_continuity/result.json','g_generation/result.json','h_scale/paired_behavior.json','delivery_audit.json')]
    immutable(CAMPAIGN/'review.json',{'attachments':[{'path':p,'sha256':sha(p)} for p in paths],
      'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sources if p.exists()],
      'corrections':corrections,'preserve':['limited middle/late-layer readability','native all-unit computation accounting','genuine H12-to-future full-state forecasts','limited cross-model conditional statistics'],
      'method_sources':['https://aclanthology.org/D19-1275/','https://arxiv.org/abs/2002.05202','https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html']})


def splits(rows):
    return [np.array([i for i,r in enumerate(rows) if r['word_split']==s],int) for s in ('train','validation','test')]
