"""Build and validate an immutable center bundle for the three-way binding benchmark."""
import argparse
from itertools import combinations
import io
import os
from pathlib import Path
import tarfile

import numpy as np

import attribute_binding_benchmark as B
import mlp_factor_steer_common as prior
from deploy.research_library import Library, canonical
from mlp_fidelity_v2_tail import capability_interval

METHODS=('native','attribution','transcoder')
LABELS={'native':'原生 MLP','attribution':'行为归因','transcoder':'同快照 transcoder'}
SPLITS={'new_frame':'新句式','new_answer':'新答案词','new_relation':'新关系'}


def plot_dose(out, summary):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True,layout='constrained')
    for ax,(name,info) in zip(axes,summary.items()):
        for method,label in zip(METHODS,('Native MLP','Gradient attribution','Transcoder')):
            y=[info['mean_clean_gap']-info['methods'][method]['dose_curve'][str(a)]['gap'] for a in B.DOSES]
            ax.plot([0,*B.DOSES],[0,*y],marker='o',linewidth=1.6,label=label)
        ax.set_title(name.replace('_',' ').title());ax.set_xlabel('Dose');ax.grid(alpha=.25)
    axes[0].set_ylabel('Clean minus edited logit margin')
    axes[-1].legend(fontsize=7)
    fig.savefig(out/'dose-curves.png',dpi=150)
    plt.close(fig)


def capability_coverage(cap, frozen, out):
    from transformers import AutoTokenizer

    tok=AutoTokenizer.from_pretrained(prior.MODEL,local_files_only=True)
    data=prior.read(prior.CAP)
    assert len(data)==len(cap['rows']) and prior.file_sha(prior.CAP)==cap['source_sha256']
    offsets=frozen['capability_offsets']
    for question, row in zip(data,cap['rows']):
        assert question.get('id')==row['id'] and question['answer']==row['answer']
        body=question['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',question['choices']))+'\n请只回答正确选项字母。'
        text=tok.apply_chat_template([{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        length=len(tok.encode(text,add_special_tokens=False))
        row['available_offsets']=[d for d in offsets if d<length]
    cap['eligible_counts_by_available_sites']={str(k):sum(r['eligible'] and len(r['available_offsets'])==k for r in cap['rows'])
                                               for k in range(len(offsets)+1)}
    with np.load(out/'capability-vectors.npz') as vectors:
        cap['frozen_site_vector_norms']={m:np.linalg.norm(vectors[m],axis=-1).tolist() for m in METHODS}


def package(out):
    lib=Library(out/'validation-library');shas=[]
    prereg=prior.read(out/'prereg.json');frozen=prior.read(out/'frozen-candidates.json')
    assert frozen['prereg_sha256']==prior.file_sha(out/'prereg.json')
    inputs={r['id']:r for r in prior.read_gz(out/'inputs.json.gz')['rows']}
    snap=prereg['model_snapshot']

    def archive(name,schema,body):
        record={'schema':schema,'kind':'real','run_id':B.RUN,'model_snapshot':snap,
                'provenance':{**prereg['provenance'],'seed':B.SEED},**body}
        sha=lib.ingest(record,name)['sha256'];shas.append(sha)
        return sha

    source=archive('source','research_source_bundle.v1',{'files':[
        {'path':f'tests/glm5/{name}','sha256':prior.file_sha(Path(__file__).with_name(name)),
         'text':Path(__file__).with_name(name).read_text(encoding='utf-8')}
        for name in ('attribute_binding_benchmark.py','attribute_binding_circuit.py',
                     'attribute_binding_archive.py','attribute_binding_path.py')]})
    design=archive('prereg','attribute_binding_mlp_prereg.v1',{'design':prereg,'candidates':frozen,'source_sha256':source,
        'pilots':{p:prior.read(out/p) for p in ('pilot-v1-failed.json','pilot-v2-failed.json','pilot.json')}})
    cap=prior.read_gz(out/'capability.json.gz');eligible=np.asarray([r['eligible'] for r in cap['rows']]);n=int(eligible.sum())
    assert n==2200
    correct=lambda method:[r['choice'][method]==r['answer'] for r in cap['rows']]
    def intervals(mask):
        return {m:{**capability_interval(correct('clean'),correct(m),mask,alpha=.05/3),
                'method':'paired exact Clopper-Pearson bounds with .05/3 across three methods'} for m in METHODS}

    cap['summary']=intervals(eligible)
    capability_coverage(cap,frozen,out)
    applied=eligible & np.asarray([bool(r['available_offsets']) for r in cap['rows']])
    cap['diagnostic_at_least_one_site']=intervals(applied)
    capability=archive('capability','attribute_binding_capability.v1',cap)
    summary={};cases={};examples={};counterexamples=[]
    for split in SPLITS:
        data=prior.read_gz(out/f'{split}.json.gz');rows=data['rows']
        assert data['frozen_candidates_sha256']==prior.file_sha(out/'frozen-candidates.json')
        assert len(rows)==B.COUNTS[split] and all(r['id'].startswith(split+':') for r in rows)
        effects={m:np.array([r['methods'][m]['matched']['gap']-r['methods'][m]['doses']['1.0']['gap'] for r in rows])
                 for m in METHODS}
        draws=np.random.default_rng(B.SEED).integers(len(rows),size=(5000,len(rows)))
        contrasts={f'{a}_minus_{b}':{'mean':float((effects[a]-effects[b]).mean()),
                    'descriptive_ci95':np.quantile((effects[a]-effects[b])[draws].mean(-1),(.025,.975)).tolist()}
                   for a,b in combinations(METHODS,2)}
        summary[split]={**data['summary'],'canary':data['canary'],'seconds':data['seconds'],
                        'mean_clean_gap':float(np.mean([r['clean']['gap'] for r in rows])),
                        'descriptive_paired_method_contrasts':contrasts}
        for method in METHODS:
            norms=np.array([r['methods'][method]['applied_norm'] for r in rows])
            summary[split]['methods'][method]['mean_realized_l2']=float(np.linalg.norm(norms,axis=-1).mean())
            summary[split]['methods'][method]['nonzero_sites']=int((norms>0).sum())
        enriched=[{**row,'input':inputs[row['id']]} for row in rows]
        refs=[archive(f'{split}-{i}','functional_cases.v1',{'start':i,'cases':enriched[i:i+16]}) for i in range(0,len(rows),16)]
        cases[split]={'n':len(rows),'chunk_size':16,'chunks':refs}
        negatives=[(i,row,[m for m,test in row['methods'].items() if test['doses']['1.0']['word']!=row['contrast']])
                   for i,row in enumerate(rows)]
        for i,row,methods in negatives:
            if methods:
                counterexamples.append({'id':row['id'],'methods':methods,'artifact_sha256':refs[i//16]})
        picked=next((i for i,row,ms in negatives if ms),0)
        examples[split]={'id':rows[picked]['id'],'artifact_sha256':refs[picked//16]}
    plot_dose(out,summary)
    dose=archive('dose','attribute_binding_dose.v1',{'x':'activation-delta multiplier','y':['signed margin','KL(clean||edited)','KL(donor||edited)'],
         'clean_gap':{s:v['mean_clean_gap'] for s,v in summary.items()},
         'splits':{s:{m:v['dose_curve'] for m,v in result['methods'].items()} for s,result in summary.items()}})
    npath=out/'native-path.json'
    path_doc=prior.read(npath)
    native=archive('native-path','attribute_binding_native_path.v1',path_doc)
    anchor={s:[m for m,v in r['methods'].items() if v['anchor'] and cap['summary'][m]['status']=='noninferior'] for s,r in summary.items()}
    report=archive('report','attribute_binding_circuit_report.v1',{
        'title':'属性绑定 · 三方法与三类独立留出','summary':summary,'case_index':cases,'counterexamples':counterexamples,
        'prereg_sha256':design,'dose_curve_sha256':dose,'native_path_sha256':native,'qualified_groups':anchor,
        'capability':{'artifact_sha256':capability,'summary':cap['summary'],'scope':cap['scope'],
                      'eligible_counts_by_available_sites':cap['eligible_counts_by_available_sites'],
                      'frozen_site_vector_norms':cap['frozen_site_vector_norms'],
                      'diagnostic_at_least_one_site':cap['diagnostic_at_least_one_site']},
        'limitations':['固定 B30、三个位置；方法使用真实反事实供体激活，不能作为图谱独立预测。',
         '新关系是独立留出家族上的两跳团队关联，关系变化包含相应提示变化；新答案词组同时改变了表面词和材料/颜色类别。',
         '三种方法都限定每位置最多 64 项，等范数针对各自的对照，方法之间的实际写入 L2 可以不同。',
         '干预为激活写入，不修改权重；归因是 B30 局部梯度候选筛选。能力门仅针对所列固定偏移。']})
    cards=[];edges=[]

    def add(name,label,index,level,artifact,scope,errors=(),block=B.BLOCK,boundary='mlp.output'):
        key=f'{B.RUN}:{name}'
        cards.append({'id':key,'layer':block,'label':label,'boundary':boundary,'index':index,
            'method':name,'run_id':B.RUN,'evidence':{'level':level,'artifact_sha256':artifact},
            'scope':scope,'counterexamples':list(errors)})
        return key

    def connect(source,target,name,artifact,level='observation'):
        edges.append({'id':prior.digest([source,target,name]),'source':source,'target':target,'relation':name,
                      'run_id':B.RUN,'evidence':{'level':level,'artifact_sha256':artifact}})

    for m in METHODS:
        unit=frozen['indices'][m][0][0];space='dictionary' if m=='transcoder' else 'mlp'
        key=add(m,f'B30 · {LABELS[m]}（64 项／位置）',{'space':space,'unit':unit},
                'intervention' if any(m in group for group in anchor.values()) else 'observation',dose,
                {'dose_curve_sha256':dose,'report_sha256':report,'site_indices':frozen['indices'][m],
                 'group_only':True,'qualified_splits':[s for s in anchor if m in anchor[s]]})
        connect('language:attribute_binding',key,'tested_at_frozen_budget',report)
        for s,result in summary.items():
            v=result['methods'][m];qualified=m in anchor[s]
            node=add(f'{m}-{s}',f'{LABELS[m]} · {SPLITS[s]} · 首词 {v["alternate_word"]}/{result["n"]}',
                {'space':space,'unit':unit},'intervention' if qualified else 'negative',
                examples[s]['artifact_sha256'],{'split':s,'result':v,'dose_curve_sha256':dose,'case_index':cases[s]},[examples[s]])
            connect(key,node,'heldout_result',report,'intervention' if qualified else 'negative')
    cache=add('kv','原生 KV · 回放与恢复',{'space':'attention','field':'kv_cache'},'parameter_source',native,
              {'native_path_sha256':native,'qualification':path_doc.get('upstream_status')},block=31,boundary='prefix.kv')
    if path_doc.get('upstream'):
        for item in path_doc['upstream']:
            head=add('head-'+str(item['head']),f'B30/H{item["head"]} · 上游探查',{'space':'attention','head':item['head']},
                     'observation',native,{'note':'局部真前向干预，范围见原始报告。'},boundary='attention.o_proj.input')
            connect(head,cache,'upstream_diagnostic',native)
    graph=archive('graph','research_graph.v1',{'feature_cards':cards,'graph_edges':edges,
             'note':'单元坐标为冻结组入口。组证据不逐个归因；KV 回放不等同于首词成功。'})
    dimensions={'n_layers':36,'n_heads':32,'n_kv_heads':8,'d_model':2560,'d_ff':9728,'head_dim':128}
    binding=archive('binding','research_space_binding.v1',{'report_sha256':report,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'attribute_binding_circuit','model_dimensions':dimensions,'native_bank':True,'case_index':cases}})
    work=archive('workpackage','research_workpackage.v1',{'title':'属性绑定 · 三方法 × 三类留出','stages':{
        'design':{'status':'done','prereg_sha256':design},'heldouts':{'status':'done','summary':summary},
        'capability':{'status':'done','artifact_sha256':capability},'native_path':{'status':path_doc['upstream_status'],'artifact_sha256':native},
        'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graph}}})
    registration={'run_id':B.RUN,'model_snapshot':snap,'report_sha256':report,'graph_sha256':graph,
                  'binding_sha256':binding,'workpackage_sha256':work}
    bundle=out/'ai2050-research-attribute-binding-phase2809.tar.gz'
    manifest=canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in shas],'studies':[registration]})
    with tarfile.open(bundle,'w:gz',compresslevel=5) as tar:
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(manifest);tar.addfile(info,io.BytesIO(manifest))
        for sha in shas:
            file=lib.root/'objects'/sha[:2]/(sha+'.json')
            assert prior.file_sha(file)==sha
            tar.add(file,arcname=f'objects/{sha}.json')
    result={**registration,'objects':len(shas),'bundle_sha256':prior.file_sha(bundle),
            'dose_curve_sha256':dose,'native_path_sha256':native,'capability_sha256':capability,'qualified_groups':anchor,
            'counterexamples':len(counterexamples),'summary':summary,'capability':cap['summary'],
            'capability_offsets_coverage':cap['eligible_counts_by_available_sites'],
            'capability_applied_subset':cap['diagnostic_at_least_one_site']}
    prior.save(out/'summary.json',result)
    print('Bundle',bundle,registration,'objects',len(shas),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);package(parser.parse_args().out)
