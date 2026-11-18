"""Native source-to-head-to-neuron causal follow-up with fresh, unselected content families."""

from itertools import batched
from pathlib import Path
import argparse
import time

import numpy as np

import paradigm_routes as X
import paradigm_routes_material as M
import paradigm_routes_report as V
import kv_chain_native as K
from deploy.storage_policy import checked_dir,check_write,staging_file,StorageBlocked


R,C=X.R,X.C
OUT=X.OUT/'source-causal'
TEMP=X.TEMP/'source-causal'
ARMS=('sham','reference-address','reference-value','request-value','matched-reference','random-reference',
      'joint-values','restore-reference','restore-all')


def fresh(tok):
    before=M.SIZES
    try:
        # Previous frozen pools occupy the first 256 unordered count pairs.
        M.SIZES=dict(previous=256,source_causal=64)
        return list(M.material(tok,'source_causal'))
    finally:
        M.SIZES=before


def setup():
    R.OUT,R.TEMP=X.OUT,X.TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN+'-source-causal',M.SEED,X.WIDTH
    R.BATCH,R.SLAB=X.BATCH,128
    R.configure(64*2**20)
    checked_dir(OUT,64*2**20,operation='范式源因果检验')


def write(name,value):
    previous=R.OUT
    try:
        R.OUT=OUT
        return R.write(name,value)
    finally:
        R.OUT=previous


def read(name):
    import json

    return json.loads((OUT/name).read_bytes())


def freeze():
    from transformers import AutoTokenizer

    setup()
    if (OUT/'plan.json').exists():raise FileExistsError('Existing source-causal preregistration')
    base=X.verify()
    candidate=R.read('candidate-freeze.json')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    rows=fresh(tok)
    script=Path(__file__).relative_to(R.ROOT).as_posix()
    plan=dict(at=C.stamp(),run=M.RUN+'-source-causal',provenance=base['provenance'],snapshot=base['snapshot'],
              parent_plan=C.A.digest(X.OUT/'plan.json'),candidate=C.A.digest(X.OUT/'candidate-freeze.json'),
              rows=rows,material_sha=M.digest(rows),sources={**base['sources'],script:C.A.digest(R.ROOT/script)},
              head=candidate['head'],control=candidate['control_head'],units=candidate['units'],random_units=candidate['random_units'],
              arms=ARMS,
              question='For the existing discovery-frozen head and original 32 units, test which two source regions change native writes and which selected downstream units receive those changes.',
              scope='Actual native responses at the shared free output query, including incorrect or quoted answers. Functional bare-Python onset is separately failed in the parent study and receives no positive label from this causal test.',
              design='64 new number-pair families after all parent content pools, eight factors each. Same source spans and same shared final-token identity. All views retained. No old confirm family reused. Hold parent lockbox closed.',
              edits='At one query/head only mask reference address, clear reference Value, clear request Value, clear both Values, then independently restore reference or all native sources. Request and reference are disjoint token regions. Other heads share original values. Matched reference/control Value displacements delivered at minimum original projected norm. Recompute subsequent Q/K, attention, MLP query states. Capture G/U/A for all frozen selected and random units at and after this head.',
              endpoints='Family-balanced effect of reference Value vs equal-dose control toward the requested format in print-The contrast. Mean >=.10, 99% interval lower>0, p95 dose relative mismatch<=.03 and at least 60 all-eight-view sham-aligned families. Partial native reference restoration after joint removal separately measured with 99% intervals. All source effects and full-vocab flips reported. Downstream selected/random A changes in full original units are descriptive and retain signs.',
              technical='Sham agreement with clean full-prefix top1 and print/The raw margin <=1 logit. Exact full-native self-restore in all nine-arm packed replays. No selecting by target edit outcome. Strictly source-determined cue edit sites.')
    sha=write('plan.json',plan)
    print('Source-causal frozen',sha,plan['head'],len(rows),flush=True)


def verify():
    p=read('plan.json')
    assert all(C.A.digest(R.ROOT/name)==sha for name,sha in p['sources'].items())
    assert p['parent_plan']==C.A.digest(X.OUT/'plan.json') and p['candidate']==C.A.digest(X.OUT/'candidate-freeze.json')
    assert M.digest(p['rows'])==p['material_sha']
    return p


def save_np(name,arrays):
    target=check_write(OUT/name,sum(x.nbytes for x in arrays.values())+2**18,operation='范式来源原生效应')
    with staging_file(target) as staging,staging.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    return C.A.digest(target)


def run():
    import torch

    p=verify()
    setup()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    layer_id,head=p['head'];control=p['control']
    anchors=R.read('plan.json')['anchors']
    units=p['units']+p['random_units']
    linked=[i for i,(layer,_) in enumerate(units) if layer>=layer_id]
    rows=p['rows'];results=[]
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('source-causal-load',0,2,start)
        assert C.Prior.identity()==p['snapshot']
        model=C.N.Engine()
        C.N.progress('source-causal-load',2,2,start)
        try:
            for part,batch_rows in enumerate(batched(rows,128)):
                panels=[C.packet(model,b) for b in batched(batch_rows,X.BATCH)]
                for p0 in panels:
                    p0['unit_gua']=np.zeros((len(p0['rows']),len(ARMS),len(units),3),np.float32)
                for index in range(40):
                    with model.block(index) as block:
                        for panel in panels:
                            x,mask=panel['x'].cuda(),panel['mask'].cuda()
                            angles=[a.cuda() for a in panel['angles']]
                            q,k,v=model.qkv(block,x,angles)
                            o=model.attention(block,q,k,v,mask)
                            j=torch.arange(len(x),device='cuda');c=panel['c'].cuda();b=len(j);arms=len(ARMS)
                            if index>=layer_id:
                                live=x[j,c].expand(arms,-1,-1).clone() if index==layer_id else panel['live'].cuda()
                                aa=[value[j,c,None].repeat(arms,1,1) for value in angles]
                                qq,kk,vv=model.qkv(block,live.reshape(arms*b,1,model.width),aa)
                                bk=k.repeat(arms,1,1,1);bv=v.repeat(arms,1,1,1)
                                idx=torch.arange(arms*b,device='cuda');cc=c.repeat(arms)
                                bk[idx,:,cc]=kk[:,:,0];bv[idx,:,cc]=vv[:,:,0]
                                vis=mask[j,:,c].unsqueeze(2)
                                wrote=model.attention(block,qq,bk,bv,vis.repeat(arms,1,1,1)).reshape(arms,b,40,128)
                                if index==layer_id:
                                    before=wrote[0].clone()
                                    source=torch.zeros((b,X.WIDTH),device='cuda',dtype=torch.bool)
                                    request=torch.zeros_like(source)
                                    for jj,row in enumerate(panel['rows']):
                                        source[jj,row['spans']['reference']]=True
                                        request[jj,row['spans']['request']]=True
                                    q0,k0,v0=qq[:b],bk[:b],bv[:b]
                                    configs=[('reference-address',source,'address'),('reference-value',source,'value'),
                                             ('request-value',request,'value'),('joint-values',source|request,'value'),
                                             ('restore-reference',request,'value')]
                                    for name,where,mode in configs:
                                        delta=K.raw_change(model,block,head,q0,k0,v0,vis,where,mode)
                                        wrote[ARMS.index(name),:,head]=(before[:,head].float()+delta).to(wrote.dtype)
                                    wrote[ARMS.index('restore-all')]=before
                                    diffs=[K.raw_change(model,block,h,q0,k0,v0,vis,source,'value') for h in (head,control)]
                                    norms=[model.projections(block,(h,),d[:,None]).norm(dim=-1)[:,0] for h,d in zip((head,control),diffs,strict=True)]
                                    goal=torch.minimum(*norms)
                                    delivered=[]
                                    for arm,h,delta,norm in zip(('matched-reference','random-reference'),(head,control),diffs,norms,strict=True):
                                        cast=(before[:,h].float()+delta*(goal/norm.clamp_min(1e-8))[:,None]).to(wrote.dtype)
                                        wrote[ARMS.index(arm),:,h]=cast
                                        measured=model.projections(block,(h,),(cast.float()-before[:,h].float())[:,None]).norm(dim=-1)[:,0]
                                        delivered.append(measured)
                                    panel['dose']=torch.stack((goal,*delivered),-1).cpu().numpy()
                                after=live+block.self_attn.o_proj(wrote.reshape(arms,b,model.width).contiguous())
                                normalized=block.post_attention_layernorm(after)
                                g=block.mlp.gate_proj(normalized);u=block.mlp.up_proj(normalized)
                                a=torch.nn.functional.silu(g)*u
                                chosen=[(ii,unit) for ii,(layer,unit) in enumerate(units) if layer==index]
                                for unit_index,neuron in chosen:
                                    snapshot=torch.stack((g[:,:,neuron],u[:,:,neuron],a[:,:,neuron]),-1)
                                    panel['unit_gua'][:,:,unit_index]=snapshot.transpose(0,1).float().cpu().numpy()
                                panel['live']=(after+block.mlp.down_proj(a)).cpu()
                            panel['x']=model.residual(block,x,o).cpu()
                    if (index+1)%5==0:
                        C.N.progress('source-causal',part*40+index+1,160,start)
                for panel in panels:
                    live=model.logits(panel['live'].cuda()).float()
                    clean=model.logits(panel['x'].cuda()[torch.arange(len(panel['rows']),device='cuda'),panel['c'].cuda()]).float()
                    assert torch.equal(live[0],live[-1])
                    margin=live[:,:,anchors['print']]-live[:,:,anchors['The']]
                    cleanmargin=clean[:,anchors['print']]-clean[:,anchors['The']]
                    top=live.argmax(-1).T.cpu().numpy()
                    results.append(dict(ids=np.array([r['id'] for r in panel['rows']]),top=top,margin=margin.T.cpu().numpy(),
                                        clean_top=clean.argmax(-1).cpu().numpy(),clean_margin=cleanmargin.cpu().numpy(),
                                        dose=panel['dose'],unit_gua=panel['unit_gua']))
                C.N.progress('source-causal-saved-memory',min(len(rows),(part+1)*128),len(rows),start)
        finally:
            model.close()
    arrays={key:np.concatenate([r[key] for r in results]) for key in results[0]}
    sha=save_np('effects.npz',arrays)
    aligned=(arrays['top'][:,0]==arrays['clean_top']) & (abs(arrays['margin'][:,0]-arrays['clean_margin'])<=1)
    take=aligned.reshape(-1,8).all(1).repeat(8)
    sign=np.asarray([1 if r['regime'][0] else -1 for r in rows])
    effects=(arrays['margin'][:,ARMS.index('random-reference')]-arrays['margin'][:,ARMS.index('matched-reference')])*sign
    eligible=take.reshape(-1,8).all(1)
    loss=V.family_summary(effects.reshape(-1,8).mean(1)[eligible]) if eligible.any() else dict(n=0,mean=None,ci=None)
    gain=(arrays['margin'][:,ARMS.index('restore-reference')]-arrays['margin'][:,ARMS.index('joint-values')])*sign
    restore=V.family_summary(gain.reshape(-1,8).mean(1)[eligible]) if eligible.any() else dict(n=0,mean=None,ci=None)
    rel=abs(arrays['dose'][:,1]-arrays['dose'][:,2])/np.maximum(arrays['dose'][:,0],.02)
    dose=float(np.quantile(rel[take],.95)) if take.any() else None
    report=dict(at=C.stamp(),head=p['head'],control=control,plan=C.A.digest(OUT/'plan.json'),effects=sha,
        sham_aligned=int(take.sum()),families=int(eligible.sum()),excluded_ids=arrays['ids'][~take].tolist(),
        matched_effect=loss,partial_reference_restore=restore,dose_p95=dose,
        passed=bool(eligible.sum()>=60 and loss['mean']>=.1 and loss['ci'][0]>0 and dose<=.03),
        actual_top_changes={name:int((take&(arrays['top'][:,i]!=arrays['top'][:,0])).sum()) for i,name in enumerate(ARMS)},
        neuron_receivers=[dict(layer=layer,unit=unit,group='selected' if i<32 else 'random',
             reference_value_mean_abs_A_change=float(abs(arrays['unit_gua'][take,2,i,2]-arrays['unit_gua'][take,0,i,2]).mean()),
             request_value_mean_abs_A_change=float(abs(arrays['unit_gua'][take,3,i,2]-arrays['unit_gua'][take,0,i,2]).mean()))
             for i,(layer,unit) in enumerate(units) if i in linked],
        native_all_restore_exact=True,scope=p['scope'])
    write('audit.json',report)
    print('Source causal',report['passed'],report['families'],loss,'dose',dose,flush=True)


def paragraph():
    p=verify();a=read('audit.json')
    assert a['plan']==C.A.digest(OUT/'plan.json') and a['effects']==C.A.digest(OUT/'effects.npz')
    from record_update_report import show

    text=f'''### 新家族的范式来源 → 注意力头 → 原生单元因果复核

单独预冻结来源协议，在父实验全部内容池之后取 64 个新家族共 512 视图，候选头和单元完全沿用发现冻结结果。此检验考察实发前缀的原生来源作用，包含模型实际的引号和错误格式响应。父实验 Python 首词校准失败单独保留。先于新家族响应写入计划 `{a['plan']}`，头 `{a['head']}`，对照 `{a['control']}`。仅在同一个查询位置，分别移除示例来源的寻址、示例 Value、请求 Value、两组 Value，并作独立示例恢复及全原生恢复。等幅的示例 Value 与控制头用实际 W_O 投影核对。

64 个家族中 {a['families']} 个通过全八视图 sham 技术对齐。相对等幅控制的 requested-format 分差效应为 {show(a['matched_effect'])}，剂量误差 P95 为 {a['dose_p95']}，冻结主检验为 {a['passed']}。联合移除后独立恢复示例 Value 的分差变化为 {show(a['partial_reference_restore'])}。各分支真实全词表首选变化为 {a['actual_top_changes']}。全原生自恢复逐位相同。下游所选与同层随机单元 G/U/A 的原生响应保留在 source-causal/effects.npz，读者清单与绝对变化在 source-causal/audit.json。它们将输入来源编辑与后续 MLP 单元响应连接，作用范围限定为这个头、这两个源区及原生生成入口。
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','all','close'))
    args=parser.parse_args()
    R.OUT=X.OUT
    if args.action=='freeze':freeze()
    elif args.action=='all':run()
    else:
        text=paragraph()
        original=V.render
        V.render=lambda plan,record:original(plan,record)+'\n\n'+text
        V.close(X.verify())
        files={p.name:dict(bytes=p.stat().st_size,sha256=C.A.digest(p)) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
        write('manifest.json',dict(files=files,sources=read('plan.json')['sources']))


if __name__=='__main__':
    try:main()
    except StorageBlocked as error:raise SystemExit(str(error)) from None
