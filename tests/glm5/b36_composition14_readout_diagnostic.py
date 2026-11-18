"""Post-confirmation 2x2 readout decomposition of the observed code reversal.

Uses existing edited final states. Off-diagonal readouts are algebraic diagnostics,
not model behavior under a valid prompt and not held-out mechanism predictions.
"""
import argparse,time
import numpy as np
import b36_composition14 as B
import b36_composition14_tail as T
from recursive_branch14_analysis import inference


def plan():
    B.configure(40*2**20);T.admit();assert not (B.OUT/'readout-diagnostic-plan.json').exists()
    B.save('readout-diagnostic-plan.json',dict(kind='explicit_post_confirmation_exploration',source_sha256=B.file_sha(__file__),
        observed_before_design='ABC/XYZ paired format reversal was already examined. These rows are exposed and cannot become independent confirmation.',
        inputs={f'tail-{s}/summary.json':B.file_sha(B.OUT/f'tail-{s}/summary.json') for s in ['confirm','format']},
        method='For each of140 matched questions, score existing baseline/zero/double final raw states from ABC and XYZ using both code mappings. Same original BF16 RMSNorm and complete lm_head, batch4. Original diagonal candidate scores must match bitwise or stop.',
        quantities='E(state,code)=edited candidate margin minus baseline candidate margin. For each dose, code contribution=.5[(E(A,X)-E(A,A))+(E(X,X)-E(X,A))]; state contribution=.5[(E(X,A)-E(A,A))+(E(X,X)-E(A,X))]. Their sum is total XYZ-ABC difference. Interaction is the finite 2x2 cross difference.',
        limits=['Off-diagonal code scoring is not a valid-task behavioral intervention.','Existing edited downstream states are inputs; not prospective prediction.','State term includes baseline normalization, B36 response and all later state changes; no layer-specific formation inference.',
                'Shapley allocation is an algebraic convention, not unique causal mediation or a new semantic law.','20 existing families/branch, no new independent N; descriptive95% intervals only.'],
        thresholds='No added scientific pass/fail thresholds; exact replay and decomposition equality are technical checks.',budget=dict(peak_new_bytes=40*2**20,reserve_bytes=20*2**30,serial_gpu=True)))


def run():
    B.configure(40*2**20);T.admit();p=B.read(B.OUT/'readout-diagnostic-plan.json');assert p['source_sha256']==B.file_sha(__file__)
    for name,sha in p['inputs'].items():assert B.file_sha(B.OUT/name)==sha
    assert not (B.OUT/'readout-diagnostic.json').exists()
    materials={r['id']:r for r in B.read(B.OUT/'materials.json')['rows']}
    raw={s:B.read(B.OUT/f'tail-{s}/summary.json') for s in ['confirm','format']}
    effects={s:{(r['id'],r['arm']):r for r in data['effects']} for s,data in raw.items()}
    def key(r):return r['family'],r['sequence'],r['mode'],r['title']
    originals={key(materials[i]):materials[i] for i in raw['confirm']['row_ids'] if materials[i]['title']==0}
    pairs=[]
    for i in raw['format']['row_ids']:
        x=materials[i];a=originals[key(x)]
        for field in ['choices','gold','body','frame','input_text','intermediate_states','program_answer','record_answer']:
            assert a[field]==x[field],(field,a['id'],x['id'])
        assert len(a['tokens'])==len(x['tokens']) and a['answer_pos']==x['answer_pos']
        pairs.append((a,x))
    needed={s:{pair[0 if s=='confirm' else 1]['id'] for pair in pairs} for s in raw};states={}
    for split in raw:
        for path in sorted((B.OUT/f'tail-{split}').glob('chunk-*.json')):
            m=B.read(path);wanted=[(j,id_) for j,id_ in enumerate(m['row_ids']) if id_ in needed[split]]
            if not wanted:continue
            assert B.file_sha(path.with_suffix('.npz'))==m['arrays_sha256']
            with np.load(path.with_suffix('.npz')) as data:
                arr=data['final_raw'];arms=raw[split]['arms']
                for j,id_ in wanted:
                    for arm in ['baseline','zero','double']:states[id_,arm]=B.unbits(arr[arms.index(arm),j]).copy()
    R=B.bind();start=time.monotonic();jobs=[(pair_index,state,arm) for pair_index in range(len(pairs)) for state in [0,1] for arm in ['baseline','zero','double']];scored={};maximum_error=0.
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg=B.read(B.MODEL/'config.json');norm=Qwen3RMSNorm(B.DIM,eps=cfg['rms_norm_eps']).cuda()
        norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False);head=R.C.weight('lm_head.weight').cuda()
        for lo in range(0,len(jobs),B.BATCH):
            jj=jobs[lo:lo+B.BATCH];h=np.stack([states[pairs[i][state]['id'],arm] for i,state,arm in jj])
            z=torch.nn.functional.linear(norm(torch.tensor(h,device='cuda',dtype=torch.bfloat16)),head).float()
            for j,(i,state,arm) in enumerate(jj):
                r=pairs[i][state];gold=r['gold'];split='confirm' if state==0 else 'format'
                for code in [0,1]:
                    scores=z[j,pairs[i][code]['choice_ids']].cpu().tolist()
                    if code==state:
                        original=effects[split][r['id'],arm]['scores'];maximum_error=max(maximum_error,max(abs(a-b) for a,b in zip(original,scores)))
                        assert original==scores,('Diagonal BF16 replay mismatch',r['id'],arm,original,scores)
                    scored[i,state,code,arm]=dict(scores=scores,margin=scores[gold]-max(v for k,v in enumerate(scores) if k!=gold))
            if lo%120==0 or lo+len(jj)==len(jobs):B.log('posthoc_readout_replay',lo+len(jj),len(jobs),start)
    rows=[]
    for i,(a,x) in enumerate(pairs):
        for arm in ['zero','double']:
            e={(state,code):scored[i,state,code,arm]['margin']-scored[i,state,code,'baseline']['margin'] for state in [0,1] for code in [0,1]}
            code=.5*((e[0,1]-e[0,0])+(e[1,1]-e[1,0]));state=.5*((e[1,0]-e[0,0])+(e[1,1]-e[0,1]));total=e[1,1]-e[0,0]
            assert abs(code+state-total)<1e-10
            rows.append(dict(id_ABC=a['id'],id_XYZ=x['id'],family=a['family'],branch=a['branch'],depth=a['depth'],mode=a['mode'],arm=arm,
                E_AA=e[0,0],E_AX=e[0,1],E_XA=e[1,0],E_XX=e[1,1],code_component=code,state_component=state,total=total,interaction=e[1,1]-e[1,0]-e[0,1]+e[0,0]))
    summary=[]
    for key_ in sorted({(r['branch'],r['depth'],r['mode'],r['arm']) for r in rows}):
        rr=[r for r in rows if (r['branch'],r['depth'],r['mode'],r['arm'])==key_]
        summary.append(dict(zip(['branch','depth','mode','arm'],key_),n=len(rr),statistics={k:inference([r[k] for r in rr]) for k in ['E_AA','E_AX','E_XA','E_XX','code_component','state_component','total','interaction']}))
    assert p['source_sha256']==B.file_sha(__file__)
    B.save('readout-diagnostic.json',dict(kind='post_confirmation_algebraic_diagnostic_not_validation',rows=rows,summary=summary,pairs=len(pairs),new_independent_samples=0,
        diagonal_max_abs_error=maximum_error,diagonal_replays=len(jobs),decomposition_identity_passed=True,plan_sha256=B.file_sha(B.OUT/'readout-diagnostic-plan.json'),
        source_sha256=B.file_sha(__file__),seconds=time.monotonic()-start,limits=p['limits']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','run']);a=p.parse_args();globals()[a.stage]()
