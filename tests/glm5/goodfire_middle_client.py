"""Register audited middle-layer atlases in the existing local client library."""
import argparse
import io
from pathlib import Path
import sys
import tarfile
import time

import goodfire_middle as M
C=M.C
from deploy.storage_policy import check_write
from deploy.research_library import Library,canonical


def catalog():
    report=C.read(M.OUT/'report.json');manifest=C.read(M.OUT/'manifest.json')
    assert C.read(M.OUT/'audit.json')['passed'] and manifest['state']=='complete'
    check_write(M.BASE,650*2**20)
    lib=Library(check_write(M.ROOT/'distributed_data/research_library',650*2**20))
    objects={};start=time.monotonic()
    def put(name,doc):
        body={**doc,'kind':'real','run_id':M.RUN,'model_snapshot':M.SPEC['snapshot'],
            'provenance':doc.get('provenance',report['provenance'])}
        h=lib.ingest(body,M.RUN+'/'+name,kind='local_report')['sha256'];objects[h]=canonical(body);return h
    report_sha=put('report',report);manifest_sha=put('native-manifest',manifest)
    graphs=[];layers={}
    for b in M.BLOCKS:
        p=C.read(M.OUT/f'B{b}-profile.json');cues=C.read(M.OUT/f'B{b}-cues.json')['units']
        layer_sha=put(f'layer-{b}',dict(schema='native_mlp_layer_index.v1',block=b,width=C.WIDTH,
            native_manifest_sha256=manifest_sha,
            arrays={f'{f}_{k}':p['fields'][f][k] for f in ['G','V','A'] for k in ['mean','std','min','max']}))
        layers[str(b)]=layer_sha
        for lo in range(0,C.WIDTH,512):
            cards=[dict(id=f'{M.SPEC["snapshot"]}:{M.RUN}:B{b}:U{u}',layer=b,boundary='mlp.down_proj.input',
                index=dict(space='mlp',unit=u),run_id=M.RUN,label=f'B{b}/U{u} · '+cues[u]['description'],
                method='Native BF16 all-coordinate observation and frozen response prediction',
                evidence=dict(level='observation',artifact_sha256=layer_sha),
                counterexamples=[dict(artifact_sha256=report_sha,reason='Prediction residuals and finite intervention limits retained.')],
                scope=dict(semantic_confirmed=False,unit_api=f'/api/research/native-mlp/units/{b}/{u}?run={M.RUN}'))
                for u in range(lo,min(lo+512,C.WIDTH))]
            graphs.append(put(f'graph-{b}-{lo}',dict(schema='research_graph.v1',feature_cards=cards,graph_edges=[])))
        M.log('client_registration',(b-17)*C.WIDTH,11*C.WIDTH,start)
    cfg=report['config']
    binding=put('binding',dict(schema='research_space_binding.v1',report_sha256=report_sha,graph_sha256=graphs[0],traces=[],
        context=dict(study_type='native_mlp_atlas',native_bank=True,
            model_dimensions=dict(n_layers=cfg['num_hidden_layers'],d_ff=C.WIDTH,d_model=C.DIM,n_heads=cfg['num_attention_heads']),
            all_graph_sha256=graphs,layer_artifacts=layers,native_manifest_sha256=manifest_sha,
            read_api='/api/research/native-mlp?run='+M.RUN)))
    work=put('workpackage',dict(schema='research_workpackage.v1',title=f'{M.SPEC["name"]} · B18–B28全部{11*C.WIDTH:,}个单元',
        stages=dict(collection=dict(status='done',units=11*C.WIDTH,documents=1024),
            semantic_mechanisms=dict(status='not_complete',confirmed=0),
            registration_validated=dict(status='done',report_sha256=report_sha,graph_sha256=graphs[0]))))
    registration=dict(run_id=M.RUN,model_snapshot=M.SPEC['snapshot'],report_sha256=report_sha,
        graph_sha256=graphs[0],binding_sha256=binding,workpackage_sha256=work,all_graph_sha256=graphs)
    package=M.BASE/f'{M.MODEL_KEY}-b18-b28-research-upload.tar.gz'
    with tarfile.open(package,'w:gz',compresslevel=1) as tar:
        def add(name,body):
            info=tarfile.TarInfo(name);info.size=len(body);tar.addfile(info,io.BytesIO(body))
        add('MANIFEST.json',canonical(dict(schema='ai2050_research_upload.v1',objects=[dict(sha256=s) for s in objects],studies=[registration])))
        for digest,raw in objects.items():add(f'objects/{digest}.json',raw)
    M.write(M.OUT/'client-registration.json',dict(**registration,objects=len(objects),native_units=11*C.WIDTH,
        bundle=str(package),bundle_sha256=C.file_sha(package),remote_receipt=None,native_data_separate=True))
    print('Client catalog registered',M.MODEL_KEY,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS);a=p.parse_args()
    M.configure(a.model);M.require_admission();catalog()


if __name__=='__main__':main()
