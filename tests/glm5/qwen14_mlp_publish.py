"""Audit and persist every native unit, negative result and deployment identity."""
from collections import defaultdict
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import qwen14_mlp_atlas as C
from qwen14_mlp_provenance import save_result


def effects_summary(stage):
    p=C.OUT/f'{stage}-effects.json'
    if not p.exists():return {'status':'not_completed','stage':stage}
    from scipy.stats import beta,binomtest
    d=C.read(p);groups=defaultdict(list)
    for r in d['rows']:groups[r['branch']].append(r)
    valid={(x['sample'],x['branch']):x['valid'] for x in d['controls']}
    admissible={r['sample']:r['restore_bitwise'] and r['full_vs_kv_kl']<=.001 for r in d['replay']}
    rows=[];counterexamples=[]
    for c in d['branches']:
        if c.get('condition') not in ['native','norm_random']:continue
        all_rows=groups[c['id']]
        control_id=c['id'].rsplit(':',1)[0]+':norm_random'
        keep=[r for r in all_rows if admissible[r['sample']] and valid.get((r['sample'],control_id),False)]
        row={**c,'n_total':len(all_rows),'n_matched':len(keep),'mean_kl_all':float(np.mean([r['kl'] for r in all_rows])),
             'argmax_changes_all':sum(r['argmax']!=r['baseline_argmax'] for r in all_rows)}
        if stage.startswith('natural'):
            row.update(mean_nll_change_all=float(np.mean([r['nll']-r['baseline_nll'] for r in all_rows])),
                mean_nll_change_matched=float(np.mean([r['nll']-r['baseline_nll'] for r in keep])) if keep else None,
                correct_to_wrong=sum(r['baseline_argmax']==r['target'] and r['argmax']!=r['target'] for r in all_rows),
                wrong_to_correct=sum(r['baseline_argmax']!=r['target'] and r['argmax']==r['target'] for r in all_rows))
            discordant=row['correct_to_wrong']+row['wrong_to_correct']
            row.update(correctness_delta=(row['wrong_to_correct']-row['correct_to_wrong'])/len(all_rows),
                paired_exact_p=float(binomtest(row['wrong_to_correct'],discordant,.5).pvalue) if discordant else 1.,
                statistical_scope='Descriptive exact paired test; dose zero is the primary comparison, 16 unit comparisons; no semantic promotion.')
            worst=sorted(all_rows,key=lambda r:r['nll']-r['baseline_nll'],reverse=True)[:2]
        else:
            lost=sum(int(np.argmax(r['baseline_scores']))==r['gold'] and not r['correct'] for r in all_rows)
            gained=sum(int(np.argmax(r['baseline_scores']))!=r['gold'] and r['correct'] for r in all_rows)
            upper=float(beta.ppf(.95,lost+1,len(all_rows)-lost)) if lost<len(all_rows) else 1.
            row.update(lost=lost,gained=gained,lost_fraction_upper95=upper,capability_gate=upper<=.02 and all(admissible.values()),
                baseline_correct=sum(int(np.argmax(r['baseline_scores']))==r['gold'] for r in all_rows),edited_correct=sum(r['correct'] for r in all_rows))
            worst=[r for r in all_rows if int(np.argmax(r['baseline_scores']))==r['gold'] and not r['correct']][:2]
        rows.append(row);counterexamples+=worst
    return {'status':'complete','samples':d['samples'],'rows':rows,'counterexamples':counterexamples,
        'replay_passed':sum(admissible.values()),'restore_all':all(r['restore_bitwise'] for r in d['replay']),
        'max_replay_kl':max(r['full_vs_kv_kl'] for r in d['replay']),
        'norm_controls_valid':sum(valid.values()),'norm_controls_total':len(valid),'formal_promotion':False,
        'scope':'Finite final-token native-column writes; scoped cached-prefix replay. Complete and matched denominators both retained.'}


def finalize():
    start=time.monotonic();plan=C.read(C.OUT/'plan.json');collection=C.read(C.OUT/'collection.json')
    assert collection['documents']==1024 and collection['completed_chunks']==[0,256,512,768]
    assert all(r['product_bitwise_equal'] and r['output_relative_error']<=.02 for r in collection['replay'])
    selected=C.read(C.OUT/'intervention-selection.json')['units']
    per_layer=[]
    for b in C.BLOCKS:
        for field,w in [('G',C.WIDTH),('V',C.WIDTH),('A',C.WIDTH),('input',C.DIM),('output',C.DIM)]:
            a=np.load(C.OUT/f'B{b}-{field}.npy',mmap_mode='r');assert a.shape==(1024,8,w) and a.dtype==np.uint16
            for lo in range(0,1024,64):assert np.isfinite(C.unbits(a[lo:lo+64])).all()
        p=C.read(C.OUT/f'B{b}-profile.json');assert len(p['fields']['A']['mean'])==C.WIDTH
        cues=C.read(C.OUT/f'B{b}-cues.json')['units'];assert len(cues)==C.WIDTH
        prediction=C.read(C.OUT/f'B{b}-prediction.json')['fields']
        per_layer.append({'block':b,'units':C.WIDTH,'conditional_prediction_candidates':{f:v['count'] for f,v in prediction.items()},
            'zero_holdout_keyword_hits':sum(x['repeat']['holdout']['hits']==0 for x in cues)})
        C.log('audit_dimensions',b-27,8,start)
    natural=effects_summary('natural_exact');cap=effects_summary('capability_exact')
    if natural['status']!='complete' or cap['status']!='complete':raise RuntimeError('Do not publish unfinished intervention/capability as complete')
    save_result(C.OUT/'intervention-summary.json',{'natural':natural,'capability':cap,'selected_units':selected})
    save_result(C.OUT/'audit.json',{'passed':True,'scope':'Same-node identity/dimension/arithmetic audit, not independent scientific replication',
        'collection_sha256':C.file_sha(C.OUT/'collection.json'),'native_replay_checks':collection['replay'],'layers':per_layer,
        'provenance_boundary':C.read(C.OUT/'admission-note.json')['scope']})
    neighbors=C.read(C.OUT/'neighbor-summary.json')
    compact_neighbors={**neighbors,'rollout':[{k:v for k,v in row.items() if k!='unit_ratios'} for row in neighbors['rollout']]}
    report={'schema':'qwen14_mlp_atlas_report.v1','title':'Qwen3-14B · B28–B35全部MLP原生特性与关联',
        'model_name':'Qwen3-14B','precision':'BF16','blocks':C.BLOCKS,'width':C.WIDTH,'hidden_size':C.DIM,'model_layers':40,'config':C.read(C.MODEL/'config.json'),
        'documents':1024,'context_tokens':1024,'n_units':139264,'positions':C.POSITIONS,
        'coverage':{'native_profiles':139264,'native_parameter_vectors':139264*3,'semantic_confirmed':0,
                    'intervened_native_units':16,'adjacent_boundaries':7},
        'layers':per_layer,'neighbor_summary':compact_neighbors,
        'prediction_summary':C.read(C.OUT/'prediction-summary.json'),'intervention':{'natural':natural,'capability':cap,'selected_units':selected},
        'scope':'Native-neuron observational census, heldout response prediction and finite write interventions. Not a complete semantic decoding of all units.',
        'formal_evidence_promotion':False,'collection_seconds':collection['seconds'],
        'limitations':['B28–B35 are eight selected blocks of the 40-block model, not its last eight blocks.',
            'The licensed natural corpus was used before in 4B research; document holdouts are estimator-specific, not globally new.',
            'Full G/V/A raw arrays at eight positions; extrema over all positions 512–1023. Other all-token activations can be recomputed, not all retained.',
            'Keywords are discovery-only context retrieval cues, not verified concept names. No complete language/grammar/reasoning atlas is claimed.',
            'Correlations and static write/read cosines are not causal edges; adjacent predictors start from observed upstream A.',
            'Native-column interventions change the final residual write; they are not model weight edits or proof of a unique mechanism.',
            'KV caches are used and numerically replayed during execution; per-token K/V tensors are not retained in this archive. Inputs, exact runner and normalized final states support recomputation.',
            'Capability uses 512 previously exposed keyed questions and finite final-token edits; no universal free-generation guarantee.',
            'Initial census/pilot lacked the required five-field provenance envelope; later auditing does not retroactively establish prior compliance.',
            'One physical node and one model; WDDM shared-memory allocation was not measured. No VPD training or SAE/transcoder comparative claim.'],
        'references':plan['references'],'read_api':'/api/research/qwen14-mlp'}
    save_result(C.OUT/'report.json',report)
    shutil.copyfile(C.MODEL/'tokenizer.json',C.OUT/'tokenizer.json')
    shutil.copyfile(C.ROOT/'tests/glm5/result/mlp_tail_goodfire_v1/QWEN-LICENSE.txt',C.OUT/'QWEN-LICENSE.txt')
    (C.OUT/'WEIGHT-NOTICE.txt').write_text('Qwen3-14B by Qwen/Alibaba, Apache-2.0. Source: https://huggingface.co/Qwen/Qwen3-14B\nLicense verified at https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE\nThis dataset exports the original B28-B35 gate/up/down parameters as lossless BF16 uint16 arrays; no model weights are trained or edited. Model file checksums: model-identity.json. Natural-document attribution and licenses are retained per document in documents.json.\n',encoding='utf-8')
    files=[]
    names=['plan.json','analysis-plan.json','collector-amendment.json','model-identity.json','tokenizer-audit.json','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt',
        'documents.json','tokenizer.json','tokens.npy','prefix-extrema.npz','report.json','audit.json',
        'collection.json','prediction-summary.json','neighbor-summary.json','intervention-plan.json','intervention-selection.json',
        'intervention-summary.json','natural_exact-effects.json','capability_exact-effects.json',
        'natural_exact-final-states.npy','capability_exact-final-states.npy','capability-inputs.json',
        'natural_exact-full-forward-final.npy','capability_exact-full-forward-final.npy',
        'admission-note.json','implementation-amendment.json','intervention-execution-binding.json',
        'exact-calibration-admission.json','kv-calibration-recovery-plan.json',
        'calibrate-effects.json','calibrate_precise-effects.json','calibrate_exact-effects.json',
        'context-terms.json','text-features.npz']
    names += [f'B{b}-{f}.npy' for b in C.BLOCKS for f in ['G','V','A','input','output','W-gate','W-up','W-down']]
    names += [f'B{b}-{f}.json' for b in C.BLOCKS for f in ['profile','cues','prediction','readout']]
    names += [f'B{b}-B{b+1}-prediction.json' for b in C.BLOCKS[:-1]]
    names += [f'B{b}-B{b+1}-predictor.npz' for b in C.BLOCKS[:-1]]
    names += [f'B{b}-{f}-context-coeff.npy' for b in C.BLOCKS for f in ['G','V','A']]
    for name in ['qwen14_mlp_atlas.py','qwen14_mlp_analysis.py','qwen14_mlp_intervene.py','qwen14_mlp_provenance.py','qwen14_mlp_publish.py']:
        dest='source-'+name;shutil.copyfile(C.ROOT/'tests/glm5'/name,C.OUT/dest);names.append(dest)
    for i,name in enumerate(names):
        path=C.OUT/name;files.append({'path':name,'bytes':path.stat().st_size,'sha256':C.file_sha(path)})
        if i%16==0:C.log('audit_hashes',i+1,len(names),start)
    save_result(C.OUT/'manifest.json',{'schema':'qwen14_mlp_manifest.v1','state':'complete','blocks':C.BLOCKS,'width':C.WIDTH,
        'files':files,'retention':'Client microscope, all native unit queries, numerical reanalysis and reproducibility. Not disposable cache.'})


def publish():
    from deploy.research_library import Library,canonical
    lib=Library(C.ROOT/'distributed_data/research_library');objects={};plan=C.read(C.OUT/'plan.json')
    def put(name,doc):
        body={**doc,'kind':'real','run_id':C.RUN,'model_snapshot':plan['model_snapshot'],
              'provenance':doc.get('provenance',C.read(C.OUT/'report.json')['provenance'])}
        sha=lib.ingest(body,f'{C.RUN}/{name}',kind='local_report')['sha256'];objects[sha]=canonical(body)
        return sha
    report=put('report',C.read(C.OUT/'report.json'));manifest=put('native-manifest',C.read(C.OUT/'manifest.json'))
    graphs=[];layer_refs={}
    for b in C.BLOCKS:
        p=C.read(C.OUT/f'B{b}-profile.json');cues=C.read(C.OUT/f'B{b}-cues.json')['units']
        ref=put(f'layer-{b}',{'schema':'qwen14_layer_index.v1','block':b,'width':C.WIDTH,'native_manifest_sha256':manifest,
            'arrays':{f'{f}_{k}':p['fields'][f][k] for f in ['G','V','A'] for k in ['mean','std','min','max']},
            'scope':'Every native unit; full profiles and parameters queried from the separately verified native atlas.'})
        layer_refs[str(b)]=ref
        for lo in range(0,C.WIDTH,512):
            cards=[{'id':f'{plan["model_snapshot"]}:{C.RUN}:B{b}:U{u}','layer':b,'boundary':'mlp.down_proj.input',
                'index':{'space':'mlp','unit':u},'run_id':C.RUN,'label':f'B{b}/U{u} · '+cues[u]['description'],
                'method':'Native BF16 SwiGLU census; both poles and 512-token-or-longer prefixes; discovery-derived conditional-response screen',
                'evidence':{'level':'observation','artifact_sha256':ref},
                'counterexamples':[{'artifact_sha256':report,'reason':'Holdout lexical/prediction failures, limited causal scope and provenance boundary retained.'}],
                'scope':{'semantic_confirmed':False,'unit_api':f'/api/research/qwen14-mlp/units/{b}/{u}',
                         'relation_api':f'/api/research/qwen14-mlp/relations/{b}/{u}'}} for u in range(lo,min(lo+512,C.WIDTH))]
            graphs.append(put(f'graph-{b}-{lo}',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':[],
                'scope':'No causal graph edges inferred from correlations. All neighboring values remain queryable in the native atlas.'}))
        print(f'register B{b} {C.WIDTH} native unit cards',flush=True)
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':report,'graph_sha256':graphs[0],'traces':[],
        'context':{'study_type':'qwen14_native_atlas','native_bank':True,'model_dimensions':{'n_layers':40,'d_ff':17408,'d_model':5120,'n_heads':40},
            'all_graph_sha256':graphs,'layer_artifacts':layer_refs,'native_manifest_sha256':manifest,'read_api':'/api/research/qwen14-mlp'}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-14B · B28–B35全部139,264个单元',
        'stages':{'collection':{'status':'done','units':139264,'documents':1024},
                  'semantic_mechanisms':{'status':'not_complete','confirmed':0},
                  'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graphs[0]}}})
    reg={'run_id':C.RUN,'model_snapshot':plan['model_snapshot'],'report_sha256':report,'graph_sha256':graphs[0],
         'binding_sha256':binding,'workpackage_sha256':work,'all_graph_sha256':graphs}
    bundle=C.OUT/'ai2050-qwen14-research-upload.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objects],'studies':[reg]}))
        for sha,raw in objects.items():add(f'objects/{sha}.json',raw)
    save_result(C.OUT/'registration.json',{**reg,'objects':len(objects),'unique_native_units':139264,'bundle':str(bundle),
        'bundle_sha256':C.file_sha(bundle),'remote_receipt':None,'native_data_separate':True})


if __name__=='__main__':
    import sys
    {'finalize':finalize,'publish':publish}[sys.argv[1]]()
