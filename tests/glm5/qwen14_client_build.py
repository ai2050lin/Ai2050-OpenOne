"""Compose immutable B28--B39 client evidence and an uploadable catalog. No model run."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.qwen14_mlp_pack import RUN, RUN_V2, SNAPSHOT, sha, validate
from deploy.storage_policy import check_write
from deploy.research_library import Library, canonical

OLD=ROOT/'tests/glm5/result/qwen14_mlp_atlas_v1'
NEW=ROOT/'tests/glm5/result/qwen14_mlp_b28_b39_v2'
OUT=ROOT/'tests/glm5/result/qwen14_client_v2'
NATIVE=OUT/'native'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):check_write(p).write_bytes(canonical(d))


def native():
    check_write(OUT,600*2**20);NATIVE.mkdir(parents=True,exist_ok=True)
    files=[];start=time.monotonic()
    def link(source,name=None,base=False):
        dest=NATIVE/(name or source.name)
        if not source.is_file():raise FileNotFoundError(source)
        if dest.exists():
            if not os.path.samefile(source,dest):raise ValueError('Refuse overwrite: '+str(dest))
        else:os.link(source,dest)
        r={'path':dest.name,'bytes':source.stat().st_size,'sha256':sha(source)}
        if base:r.update(source_run=RUN,source_path=source.name)
        files.append(r)
    for b in range(28,40):
        source=OLD if b<36 else NEW
        for s in ['G','V','A','input','output','W-gate','W-up','W-down']:link(source/f'B{b}-{s}.npy',base=b<36)
        for s in ['profile','cues','prediction','readout']:link(source/f'B{b}-{s}.json',base=b<36)
        for s in ['regimes','scaled-prediction']:link(NEW/f'B{b}-{s}.npz')
        if b<39:link((OLD if b<35 else NEW)/f'B{b}-B{b+1}-prediction.json',base=b<35)
        elapsed=time.monotonic()-start
        print(f'native {b-27}/12 elapsed={elapsed:.1f}s eta={elapsed/(b-27)*(39-b):.1f}s',flush=True)
    for n in ['tokenizer.json','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt']:link(OLD/n,base=True)
    link(NEW/'documents.json')
    for n in ['prefix-extrema.npz','natural_exact-effects.json','capability_exact-effects.json','capability-inputs.json']:
        link(OLD/n,'legacy-'+n,base=True);link(NEW/n)
    for n in ['report.json','manifest.json','admission-note.json']:link(OLD/n,'historical-'+n,base=True)
    for n in ['report.json','audit.json','final-verification.json','manifest.json','regime-summary.json','scaled-prediction-summary.json','plot-specification.json','all-neuron-overview.png','A-std-native-pixels.png','double-negative-native-pixels.png','heatmap-native-arrays.npz']:
        link(NEW/n,'extension-'+n)
    old=read(OLD/'report.json');new=read(NEW/'report.json')
    intervention={'selected_units':old['intervention']['selected_units']+new['new_interventions']['selected_units']}
    for stage in ['natural','capability']:
        intervention[stage]={'rows':[{**row,'source_run_id':r} for obj,r in [(old['intervention'],RUN),(new['new_interventions'],RUN_V2)] for row in obj[stage]['rows']],
            'scope':'Separate frozen experiments; each row retains its own baseline and source run. No pooled significance.'}
    report={**old,**new,'schema':'qwen14_mlp_atlas_report.v2','n_units':208896,
        'coverage':{'units':208896,'semantic_confirmed':0},'intervention':intervention,
        'neighbor_summary':new['neighbors'],'run_id':RUN_V2,'read_api':'/api/research/qwen14-mlp?run='+RUN_V2,
        'scope':'Full observational archive; historical and new interventions retain separate baselines. No semantic promotion.',
        'limitations':new['limitations'][:-1]+['Client integration is engineering verification, not new scientific evidence.'],
        'source_reports':{'historical':sha(OLD/'report.json'),'extension':sha(NEW/'report.json')}}
    save(NATIVE/'report.json',report)
    files.append({'path':'report.json','bytes':(NATIVE/'report.json').stat().st_size,'sha256':sha(NATIVE/'report.json')})
    manifest={'schema':'qwen14_mlp_manifest.v2','state':'complete','kind':'real','run_id':RUN_V2,
        'model_snapshot':SNAPSHOT,'blocks':list(range(28,40)),'width':17408,'files':files,
        'source_manifests':{RUN:sha(OLD/'manifest.json'),RUN_V2:sha(NEW/'manifest.json')},
        'retention':'Immutable client query data. Same-volume hard links share bytes with preserved research arrays; never modify linked files.',
        'delta_requires':RUN,'provenance':new['provenance']}
    validate(manifest);save(NATIVE/'manifest.json',manifest)
    save(OUT/'native-build.json',{'files':len(files),'manifest_sha256':sha(NATIVE/'manifest.json'),
        'logical_bytes':sum(x['bytes'] for x in files),'delta_raw_bytes':sum(x['bytes'] for x in files if 'source_run' not in x),
        'all_linked_sources_unchanged':True,'source_manifests':manifest['source_manifests']})
    print(read(OUT/'native-build.json'),flush=True)


def catalog():
    check_write(OUT,600*2**20)
    lib=Library(check_write(ROOT/'distributed_data/research_library',600*2**20));objects={};start=time.monotonic()
    prov=read(NEW/'report.json')['provenance']
    def put(name,doc):
        body={**doc,'kind':'real','run_id':RUN_V2,'model_snapshot':SNAPSHOT,'provenance':doc.get('provenance',prov)}
        digest=lib.ingest(body,RUN_V2+'/'+name,kind='local_report')['sha256'];objects[digest]=canonical(body);return digest
    report=put('report',read(NATIVE/'report.json'));manifest=put('native-manifest',read(NATIVE/'manifest.json'))
    graphs=[];layers={}
    for b in range(28,40):
        p=read(NATIVE/f'B{b}-profile.json');cues=read(NATIVE/f'B{b}-cues.json')['units']
        ref=put(f'layer-{b}',{'schema':'qwen14_layer_index.v1','block':b,'width':17408,'native_manifest_sha256':manifest,
            'arrays':{f'{f}_{k}':p['fields'][f][k] for f in ['G','V','A'] for k in ['mean','std','min','max']}})
        layers[str(b)]=ref
        for lo in range(0,17408,512):
            cards=[{'id':f'{SNAPSHOT}:{RUN_V2}:B{b}:U{u}','layer':b,'boundary':'mlp.down_proj.input',
                'index':{'space':'mlp','unit':u},'run_id':RUN_V2,'label':f'B{b}/U{u} · '+cues[u]['description'],
                'method':'Native BF16 observation, signed regimes and descriptive response prediction',
                'evidence':{'level':'observation','artifact_sha256':ref},
                'counterexamples':[{'artifact_sha256':report,'reason':'Prediction failures and limited intervention scope retained.'}],
                'scope':{'semantic_confirmed':False,'unit_api':f'/api/research/qwen14-mlp/units/{b}/{u}?run={RUN_V2}'}}
                for u in range(lo,min(lo+512,17408))]
            graphs.append(put(f'graph-{b}-{lo}',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':[]}))
        elapsed=time.monotonic()-start
        print(f'catalog {b-27}/12 units={(b-27)*17408}/208896 elapsed={elapsed:.1f}s eta={elapsed/(b-27)*(39-b):.1f}s',flush=True)
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':report,'graph_sha256':graphs[0],'traces':[],
        'context':{'study_type':'qwen14_native_atlas','native_bank':True,'model_dimensions':{'n_layers':40,'d_ff':17408,'d_model':5120,'n_heads':40},
            'all_graph_sha256':graphs,'layer_artifacts':layers,'native_manifest_sha256':manifest,'read_api':'/api/research/qwen14-mlp?run='+RUN_V2}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-14B · B28–B39全部208,896个单元',
        'stages':{'collection':{'status':'done','units':208896,'documents':1024},
            'semantic_mechanisms':{'status':'not_complete','confirmed':0},
            'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graphs[0]}}})
    reg={'run_id':RUN_V2,'model_snapshot':SNAPSHOT,'report_sha256':report,'graph_sha256':graphs[0],
        'binding_sha256':binding,'workpackage_sha256':work,'all_graph_sha256':graphs}
    bundle=OUT/'ai2050-qwen14-b28-b39-research-upload.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objects],'studies':[reg]}))
        for digest,raw in objects.items():add(f'objects/{digest}.json',raw)
    save(OUT/'registration.json',{**reg,'objects':len(objects),'unique_native_units':208896,'bundle':str(bundle),
        'bundle_sha256':sha(bundle),'remote_receipt':None,'native_data_separate':True})
    print('catalog complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['native','catalog']);a=p.parse_args();globals()[a.stage]()
