"""Traceable summary, full-coordinate figures and native artifact hashes."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C
import branch_style_analysis as A
from shared.dist_protocol.mechanism_trace import unbits,file_sha

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    effect=np.load(C.OUT/'discovery_full_coordinate_effects.npz')
    raw=effect['mean_style_difference'].mean(1);score=effect['joint_score']
    fig,axes=plt.subplots(2,1,figsize=(16,7),constrained_layout=True)
    for ax,value,title,cm in [(axes[0],raw,'Raw mean debate-minus-chat A; discovery b0/b1, both tasks','RdBu_r'),(axes[1],score,'Joint standardized response score; native order retained','magma')]:
        limit=float(np.max(abs(value)))
        im=ax.imshow(value,aspect='auto',interpolation='nearest',cmap=cm,vmin=-limit if cm=='RdBu_r' else 0,vmax=limit,extent=[-.5,9727.5,35.5,-.5])
        ax.set_title(title);ax.set_ylabel('Block (zero-based)');ax.set_xlabel('Native MLP unit index');fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(C.OUT/'full_coordinate_fields.png',dpi=180);plt.close(fig)
    pred=C.read(C.OUT/'prediction_report.json');causal=C.read(C.OUT/'causal_report.json')
    fig,axes=plt.subplots(1,2,figsize=(14,4.7),constrained_layout=True)
    names=['embedding','additive','shared','interaction','shuffle']
    x=np.arange(len(names));axes[0].bar(x,[pred['methods'][n]['nmse'] for n in names],color=['#8b929d','#8b929d','#3279ac','#da8540','#8b929d'])
    axes[0].set_xticks(x,names,rotation=25);axes[0].set_ylabel('Selected-unit style-difference NMSE');axes[0].set_title('128 unseen families; all three constructions')
    names=['target','random0','random1','wrong_donor']
    axes[1].bar(np.arange(4),[causal['conditions'][n]['mean_kl_improvement_toward_donor'] for n in names],color=['#da8540','#8b929d','#8b929d','#3279ac'])
    axes[1].set_xticks(np.arange(4),names,rotation=25);axes[1].set_ylabel('Mean KL improvement toward opposite style (nat)');axes[1].set_title('Native A patches: selected vs matched controls')
    fig.savefig(C.OUT/'prediction_and_causal.png',dpi=170);plt.close(fig)

def cards():
    f=C.read(C.OUT/'frozen.json');out=[]
    disc=C.selected_rows('discovery');conf=C.selected_rows('confirm');pd=A.pairs(disc);pc=A.pairs(conf)
    ad=np.load(C.OUT/'discovery'/'A.npy',mmap_mode='r');ac=np.load(C.OUT/'confirm'/'A.npy',mmap_mode='r')
    gc=np.load(C.OUT/'confirm'/'G.npy',mmap_mode='r');uc=np.load(C.OUT/'confirm'/'U.npy',mmap_mode='r')
    for l,u in f['group']:
        card={'id':f"{f['model_snapshot']}:B{l}:MLP{u}:output_prefix",'block':l,'unit':u,'native_parameters':{'gate':f'model.layers.{l}.mlp.gate_proj.weight[{u},:]','up':f'model.layers.{l}.mlp.up_proj.weight[{u},:]','down':f'model.layers.{l}.mlp.down_proj.weight[:,{u}]'},'cells':{},'evidence':'Discovery-selected native unit; confirmation paired-response replication; only group patch tested. No individual semantic label, necessity, training genealogy or parameter causality established.'}
        for t in [0,1]:
            m=np.mean([float(unbits(ad[i1,l,1,u]))-float(unbits(ad[i0,l,1,u])) for (fam,b,task),i0,i1 in pd if b<2 and task==t])
            for b in range(3):
                d=np.array([float(unbits(ac[i1,l,1,u]))-float(unbits(ac[i0,l,1,u])) for (fam,bb,task),i0,i1 in pc if bb==b and task==t])
                card['cells'][f'b{b}-t{t}']={'mean_debate_minus_chat':float(d.mean()),'same_sign_as_discovery':float(np.mean(d*m>0))}
        g=unbits(gc[:,l,1,u]);v=unbits(uc[:,l,1,u])
        card['gate_up_sign_regimes']={'++':float(np.mean((g>=0)&(v>=0))),'--':float(np.mean((g<0)&(v<0))),'+-':float(np.mean((g>=0)&(v<0))),'-+':float(np.mean((g<0)&(v>=0)))}
        out.append(card)
    C.save(C.OUT/'feature_cards.json',{'cards':out,'all_six_cells_same_sign_fraction_at_least_0p65':sum(all(c['same_sign_as_discovery']>=.65 for c in r['cells'].values()) for r in out),'scope':'Direction replication only; strong style offsets make it insufficient for content-dependent generalization.'})

def parameters():
    import torch
    from safetensors import safe_open
    frozen=C.read(C.OUT/'frozen.json');group=frozen['group']
    index=C.read(C.MODEL/'model.safetensors.index.json')['weight_map']
    values={};weights={}
    for l in sorted({l for l,u in group}):
        units=[u for ll,u in group if ll==l]
        for name in ['gate_proj','up_proj','down_proj']:
            key=f'model.layers.{l}.mlp.{name}.weight'
            with safe_open(C.MODEL/index[key],framework='pt',device='cpu') as f:
                w=f.get_tensor(key)
                selected=w[:,units].T.contiguous() if name=='down_proj' else w[units].contiguous()
                values[f'B{l}_{name}']=selected.view(torch.uint16).numpy().copy()
                if name=='down_proj':weights[l]=selected.float().numpy()
        key=f'model.layers.{l}.post_attention_layernorm.weight'
        with safe_open(C.MODEL/index[key],framework='pt',device='cpu') as f:values[f'B{l}_norm_gamma']=f.get_tensor(key).view(torch.uint16).numpy().copy()
    np.savez(C.OUT/'native_selected_parameters.npz',**values)
    rows=C.selected_rows('confirm');pairs=A.pairs(rows);ar=np.load(C.OUT/'confirm'/'A.npy',mmap_mode='r')
    write={};cosines={}
    for l,w in weights.items():
        units=[u for ll,u in group if ll==l]
        for b in range(3):
            for t in range(2):
                delta=np.stack([unbits(ar[i1,l,1,units])-unbits(ar[i0,l,1,units]) for (fam,bb,tt),i0,i1 in pairs if bb==b and tt==t]).mean(0)
                write[f'B{l}_b{b}_t{t}']=delta@w
        for t in range(2):
            a,b=write[f'B{l}_b0_t{t}'],write[f'B{l}_b2_t{t}']
            cosines[f'B{l}_t{t}_direct_vs_alias']=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    np.savez(C.OUT/'selected_local_write_fields.npz',**write)
    C.save(C.OUT/'native_parameter_summary.json',{'model_snapshot':frozen['model_snapshot'],'group':group,'parameter_dtype':'Native BF16 uint16 bits; selected gate/up rows and transposed down columns, plus norm gamma.','source_sha256':file_sha(C.OUT/'native_selected_parameters.npz'),'local_write_cosines':cosines,'scope':'FP32 recomputation of selected columns times mean native A differences. Local algebra, not a causal edge or full module decomposition. Post-attention normalized inputs are not stored, so gate/up linear reads cannot be independently reconstructed from saved hs alone.'})

def manifest():
    start=time.time();files=[]
    custody=C.read(C.OUT/'custody.json');custody.update(confirm_output_state='opened_completed',lockbox_output_state='sealed_not_run',fresh_2x2='new-family follow-up completed; no original lockbox pairs');C.save(C.OUT/'custody.json',custody)
    current=C.snapshot(C.MODEL);assert current==C.read(C.OUT/'frozen.json')['model_snapshot']
    C.save(C.OUT/'model_snapshot_verification.json',{'model_snapshot':current,'unchanged':True,'scope':'All weight/tokenizer/config files covered by native snapshot function rehashed after GPU work.'})
    targets=sorted(p for p in C.OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','final_report.json','progress.jsonl'] and '.tmp' not in p.name and 'provenance_receipts' not in p.parts)
    for i,p in enumerate(targets):
        files.append({'path':str(p.relative_to(C.OUT)),'bytes':p.stat().st_size,'sha256':file_sha(p)})
        if p.stat().st_size>10**8 or i==len(targets)-1:C.log('artifact_hashing',i+1,len(targets),start,file=str(p.relative_to(C.OUT)))
    C.save(C.OUT/'manifest.json',{'kind':'real','model_snapshot':C.read(C.OUT/'frozen.json')['model_snapshot'],'files':files,'total_bytes':sum(r['bytes'] for r in files),'native_retention':'Retained for future full-coordinate/reanalysis and audit. Main has all units at two readouts; fresh 2x2 only pre-frozen 96 units. Other token positions/attention internals absent.'})

def final():
    pred=C.read(C.OUT/'prediction_report.json');causal=C.read(C.OUT/'causal_report.json');fusion=C.read(C.OUT/'fresh_fusion_probe'/'report.json')
    cc=C.read(C.OUT/'feature_cards.json');audit=C.read(C.OUT/'free_audit.json') if (C.OUT/'free_audit.json').exists() else None
    summaries={p:C.read(C.OUT/p/'summary.json') for p in ['pilot','discovery','confirm']}
    full=C.read(C.OUT/'full_score_audit.json')
    free_all=C.read(C.OUT/'free_all_report.json')
    report={'kind':'real','status':'local_real_pending_independent_audit','created_chicago':datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M'),'model_snapshot':C.read(C.OUT/'frozen.json')['model_snapshot'],'main_counts':{p:{'prompts':s['rows'],'families':s['families'],'content_forced_choice':s['behavior']['content_accuracy']} for p,s in summaries.items()},'fresh_probe':{'prompts':3072,'families':128,'report':fusion},'prediction':pred,'causal':causal,'feature_cards_summary':cc['all_six_cells_same_sign_fraction_at_least_0p65'],'free_audit':audit,'scientific_limits':['Constructed direct/paraphrase/reference patterns, not proven training-recursive branches.','Same vocabulary is reused; heldout roots are novel object pairs, not wholly unseen vocabulary.','Predictor observes real chat intermediate states; not independent embedding-to-output mechanism.','Interaction model passes specified additive comparison but loses to shared linear model.','Pooled vector correlations and sign accuracy are dominated by fixed style offsets; content-centered correlations and NMSE matter.','Wrong-family donor performs similarly, limiting content-specific mechanism claim.','Finite donor A patches do not prove unique/necessary pathways or individual unit causality.','Translation is one target-object color statement, not unrestricted sentence translation.','PDH shared GPU memory unavailable; never recorded as zero.','No learning checkpoints, natural-corpus branch discovery, cross-model or independent-node replication.'],'lockbox':'Original 128 roots / 1536 prompts never run; fresh 2x2 uses different pairs with extra nouns.','manifest_sha256':file_sha(C.OUT/'manifest.json')}
    report['full_score_audit']=full
    report['free_all_behavior']=free_all
    report['sensitivity_audit']=C.read(C.OUT/'sensitivity_report.json')
    report['native_parameter_summary']=C.read(C.OUT/'native_parameter_summary.json')
    report['engineering_checks']=C.read(C.OUT/'engineering_checks.json')
    for part in report['main_counts']:
        report['main_counts'][part]['content_forced_choice_cached']=report['main_counts'][part]['content_forced_choice']
        report['main_counts'][part]['content_forced_choice']=full['sets'][part]['behavior']['content_accuracy']
    report['fresh_probe']['authoritative_full_score_behavior']=full['sets']['fresh_2x2']['behavior']
    report['fresh_probe']['style_fusion_evidence_eligible']=False
    report['scientific_verdict']={'conditional_native_response_prediction':'supported within specified paired-real-state scope','interaction_superiority_over_shared_linear':'not supported; shared linear is better','correct_object_donor_advantage':'not demonstrated; nominal interval spans zero','recursive_training_genealogy':'not tested','joint_style_fusion':'not identifiable with failed mode manipulation','translation_mechanism':'not established; free generation exposes language/task compliance failures'}
    report['scientific_limits'].extend(['Cached/full candidate scores differed; final behavior uses complete independent cache-free rescoring without feature refitting.','Causal candidate-margin accuracy remains a cached-execution diagnostic; primary causal full-vocabulary KL does not depend on it.','Fresh 2x2 mode manipulation failed: output/nonadditivity numbers are descriptive, not evidence for or against a general fusion law.','Full free generation audit was added after pilot mismatch, with transparent but imperfect predicate screening; it is secondary, not a retroactive preregistered endpoint.','Cross-record donor is not guaranteed to change entity/color; 17.19% share the target name and 16.41% the target color. Strict mismatch subset remains post-hoc.','Random groups match block/count, not activation size or weight norm.','Record pairs share vocabulary/entities; family-bootstrap intervals are nominal for that clustering assumption. Entity-deletion checks are descriptive sensitivity only.'])
    C.save(C.OUT/'final_report.json',report)
    C.PV.RESULTS=C.OUT/'provenance_receipts';C.PV._git=lambda *a: __import__('subprocess').run(['git','-c',f'safe.directory={C.ROOT.as_posix()}',*a],cwd=C.ROOT,capture_output=True,text=True).stdout.strip()
    C.PV.write_result('branch_style_joint_final',report,C.MODEL,C.SEED,'branch_style_report')
    print({'status':report['status'],'main_prompts':sum(v['prompts'] for v in report['main_counts'].values()),'additional_fusion_prompts':3072},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['figures','cards','parameters','manifest','final']);a=p.parse_args()
    {'figures':figures,'cards':cards,'parameters':parameters,'manifest':manifest,'final':final}[a.command]()
