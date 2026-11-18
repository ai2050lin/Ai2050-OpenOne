"""Content-addressed core + lossless native parts; authenticated client protocol."""
import base64
import hashlib
import io
import json
import tarfile
import subprocess
import numpy as np
from transformers import AutoTokenizer
from deploy.research_library import Library,canonical
from shared.dist_protocol.mechanism_trace import unbits
from tools import provenance
import mlp_trigger_design as D

def main():
    p=D.checked();report=D.READ(D.OUT/'report.json');sel=D.READ(D.OUT/'selection.json')
    assert D.READ(D.OUT/'frozen.json')['heldout_unopened']
    # Read existing CAS bytes without touching the nearly-full D-drive catalog.
    priorlib=object.__new__(Library);priorlib.root=D.ROOT/'distributed_data/research_library'
    lib=Library(D.OUT/'publication-library');core={};native={}
    def put(name,body,binary=False):
        d={**body,'kind':'real','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'provenance':p['provenance']};raw=canonical(d);h=hashlib.sha256(raw).hexdigest()
        assert len(raw)<16*2**20
        if binary:native[h]=raw
        else:
            assert lib.ingest(d,D.RUN+'/'+name,kind='local_report')['sha256']==h
            core[h]=raw
        return h
    artifacts={}
    for name,file in [('冻结预注册','prereg.json'),('冻结预测器','frozen.json'),('发现集选择','selection.json'),
          ('原生实现核验','implementation-audit.json'),('批处理敏感性','repeatability.json'),('权重逐张量加载审计','loader-audit.json'),('技术审计','verification.json'),('能力保持精确功效','capability-power.json'),('行为准入与证据否决','semantic-admission.json')]:
        source=D.OUT/file
        if not source.exists() and p.get('discovery_reused_from'):source=D.ROOT/p['discovery_reused_from']/file
        body=D.READ(source)
        if source.parent!=D.OUT:body={**body,'reused_source_sha256':D.HASH(source),'reused_source_provenance':body.get('provenance'),'reuse_scope':'Prior discovery implementation check, not a new independent replication.'}
        artifacts[name]=put(file,body)
    raw_parts=[];manifest=[]
    for split,n in [('discovery',D.N_DISC),('holdout',D.N_HOLD)]:
        if split=='discovery' and p.get('discovery_reused_from'):continue
        for start in range(0,n,16):
            files=[]
            for f in range(start,min(n,start+16)):
                path=D.OUT/f'{split}-{f:04d}.npz';raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
                files.append({'path':path.name,'sha256':sha,'bytes':len(raw),'encoding':'base64(npz; lossless uint16 BF16 bits)',
                    'data':base64.b64encode(raw).decode(),'metadata':D.READ(path.with_suffix('.json'))})
                manifest.append({'path':path.name,'sha256':sha,'bytes':len(raw)})
            raw_parts.append(put(f'{split}-native-{start:04d}',{'schema':'mlp_trigger_native.v1','files':files,
                  'layout':'prefix: 8 conditions × [B29/U17,B30/U2,B33/U1521,B34/U6097] × [G,V,A] × token; R4:8×4×token; residual:8×[R29 final,R33 peak]×2560. Padding excluded using material token lengths.'},True))
    artifacts['全部原生前缀与残差索引']=put('native-index',{'schema':'mlp_trigger_native_index.v1','parts':raw_parts,'files':manifest,
                  'scope':f'all {len(manifest)} native families in this release; reused discovery is referenced by provenance; complete bundle set required'})
    material_parts=[]
    for split in ['discovery','holdout']:
        rows=D.materials(p,split)
        for start in range(0,len(rows),128):material_parts.append(put(f'{split}-texts-{start}',{'schema':'mlp_trigger_materials.v1','split':split,'rows':rows[start:start+128]}))
    artifacts['全部文本、词元与切分索引']=put('text-index',{'schema':'mlp_trigger_material_index.v1','parts':material_parts,'source':'author-created controlled long English documents'})
    causal=[]
    for start in range(0,D.N_HOLD,32):
        causal.append(put(f'causal-{start}',{'schema':'mlp_trigger_causal.v1','records':[
              {'measurement':D.READ(D.OUT/f'causal-{f:04d}.json'),'commitment':D.READ(D.OUT/f'prediction-{f:04d}.json')} for f in range(start,min(start+32,D.N_HOLD))]}))
    artifacts['全部先预测后干预证据']=put('causal-index',{'schema':'mlp_trigger_causal_index.v1','parts':causal})
    artifacts['能力检查原文与结果']=put('capability',{'schema':'mlp_trigger_capability.v1','inputs':D.READ(D.OUT/'capability-materials.json'),'measurements':D.READ(D.OUT/'capability.json')})
    cal=(D.OUT/'calibration.npz').read_bytes()
    artifacts['冻结全词表预测系数']=put('prediction-arrays',{'schema':'mlp_trigger_predictor_arrays.v1','sha256':hashlib.sha256(cal).hexdigest(),'encoding':'base64(npz)','data':base64.b64encode(cal).decode()})
    previous=D.READ(D.ROOT/'tests/glm5/result/mlp_condition_v2/registration.json');old=priorlib.artifact(previous['report_sha256'])
    for sha in [*old['condition_layers'].values(),old['native_parameter_sha256']]:
        core[sha]=canonical(priorlib.artifact(sha));assert hashlib.sha256(core[sha]).hexdigest()==sha
    report['condition_layers']=old['condition_layers'];report['native_parameter_sha256']=old['native_parameter_sha256'];report['artifacts']=artifacts
    admission=D.READ(D.OUT/'semantic-admission.json');assert not admission['language_mechanism_promotion_allowed']
    report['semantic_admission']=admission
    report['limits'].append('Numerical triggering and common-input observations do not establish a role-binding semantic mechanism. '+('All role frames pass the frozen behavior gate; no answer-switch mechanism was established.' if report['behavior']['passed'] else 'Failed role-answer frames veto language-mechanism promotion.'))
    tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True);materials=D.materials(p,'holdout');examples=[]
    for f,k in [(0,0),(0,4),(1,2),(2,6),(3,1),(3,7)]:
        z=np.load(D.OUT/f'holdout-{f:04d}.npz');m=D.READ(D.OUT/f'holdout-{f:04d}.json')['rows'][k];row=materials[f*8+k];pos=m['positions']['rare'];length=len(row['tokens'])
        examples.append({'id':row['id'],'prompt':row['prompt'],'position':pos,'token':tok.decode([row['tokens'][pos]]),
            'R4':unbits(z['R4'])[k,2,:length].tolist(),'source_A':unbits(z['prefix'])[k,2,2,:length].tolist(),
            'target_A':unbits(z['prefix'])[k,3,2,:length].tolist(),'residual':unbits(z['residual'])[k,1].tolist(),
            'native_sha256':D.HASH(D.OUT/f'holdout-{f:04d}.npz')})
    report['examples']=examples
    if (D.OUT/'native-math.json').exists():
        math=D.READ(D.OUT/'native-math.json');artifacts['原生权重代数分解']=put('native-math',math)
        report['native_math']={k:v for k,v in math.items() if k not in ['records','gamma','gate_row','up_row']}
    if (D.OUT/'native-math-two-coordinate.json').exists():
        math=D.READ(D.OUT/'native-math-two-coordinate.json');artifacts['归一化与读入坐标分解']=put('relative-read',math)
        report['relative_read']={k:v for k,v in math.items() if k!='records'}
    report['conclusions']=D.READ(D.OUT/'interpretation.json')['conclusions']
    ref=put('report',report);cards=[];edges=[]
    def feature(id,b,space,index,label,scope):
        cards.append({'id':id,'run_id':D.RUN,'layer':b,'boundary':'post_attention_layernorm.input' if space=='residual' else 'mlp.down_proj.input',
          'label':label,'index':{'space':space,**index},'method':'frozen native-coordinate input intervention and heldout full-prefix observation',
          'evidence':{'level':'observation','artifact_sha256':ref},'scope':scope,
          'counterexamples':[{'artifact_sha256':ref,'field':'counterexamples'}]})
    for key,(b,u,t,v) in D.PAIRS.items():
        for block,unit in [(b,u),(t,v)]:feature(f'{D.RUN}:B{block}:U{unit}',block,'mlp',{'unit':unit},f'B{block}/U{unit} · {"共同输入" if key=="broad" else "强响应"}检验',{'pair':key,'formal_promotion':False})
        for j in sel['coordinates'][key]:
            cid=f'{D.RUN}:R{b}:C{j}';feature(cid,b,'residual',{'coordinate':j},f'R{b} 坐标{j} · MLP前输入',{'pair':key,'group_coordinates':sel['coordinates'][key],'group_effect_not_each_coordinate':key=='broad'})
            for block,unit in [(b,u),(t,v)]:edges.append({'id':f'{cid}:B{block}:U{unit}','source':cid,'target':f'{D.RUN}:B{block}:U{unit}',
                'run_id':D.RUN,'relation':'候选共同输入；原生缩减、源单元阻断与等范数检验分开回查',
                'evidence':{'level':'observation','artifact_sha256':ref},'scope':{'pair':key,'group_effect':key=='broad','not_unique_cause':True}})
    for key,(b,u,t,v) in D.PAIRS.items():
        measured=next(x for x in report['dose_curve'] if x['pair']==key and x['mode']=='unit')
        edges.append({'id':f'{D.RUN}:{key}:source-unit-zero','source':f'{D.RUN}:B{b}:U{u}',
          'target':f'{D.RUN}:B{t}:U{v}','run_id':D.RUN,
          'relation':'源单元置零后的下游响应：'+('增强（范围内抑制作用候选）' if measured['target_delta_mean']>0 else '减弱'),
          'evidence':{'level':'observation','artifact_sha256':ref},
          'scope':{'method':'native source A=0; full suffix; norm-matched control',
            'measured_target_delta_mean':measured['target_delta_mean'],'control_delta_mean':measured['control_delta_mean'],
            'not_weight_sign':True,'not_unique_path':True,'not_semantic_promotion':True}})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
         'context':{'study_type':'mlp_trigger','native_bank':True,'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32}}})
    title='Qwen3-4B · 坐标100读入的新前缀确认' if p.get('discovery_reused_from') else 'Qwen3-4B · 稀有触发、共同上游与原生阻断'
    work=put('workpackage',{'schema':'research_workpackage.v1','title':title,
         'stages':{'preregistration':{'status':'done','power_passed':True},'heldout':{'status':'done','families':D.N_HOLD,'samples':8*D.N_HOLD},
                   'interventions':{'status':'done','families':D.N_HOLD,'formal_evidence_added':0},
                   'language_behavior_gate':{'status':'passed' if report['behavior']['passed'] else 'failed','failed_frames':admission['failed_frames'],'mechanism_promoted':False},
                   'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    registry={'run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    parts=[];current={};size=0
    for sha,raw in native.items():
        if current and size+len(raw)>48*2**20:parts.append(current);current={};size=0
        current[sha]=raw;size+=len(raw)
    if current:parts.append(current)
    archives=[]
    for part,extra in enumerate(parts,1):
        objs={**extra,**core};folder=D.OUT/'packages';folder.mkdir(exist_ok=True)
        path=folder/f'ai2050-research-{D.RUN.lower()}-part{part}.tar.gz'
        assert sum(len(v)+1024 for v in objs.values())<127*2**20
        with tarfile.open(path,'w:gz',compresslevel=6) as tar:
            def add(name,raw):
                info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
            add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objs],'studies':[registry]}))
            for sha,raw in objs.items():add(f'objects/{sha}.json',raw)
        assert path.stat().st_size<64*2**20
        archives.append({'path':str(path.relative_to(D.ROOT)),'sha256':D.HASH(path),'bytes':path.stat().st_size,'objects':len(objs)})
    D.save('registration.json',{**registry,'bundles':archives,'all_objects':list(dict.fromkeys([*native,*core])),
           'native_parts':raw_parts,'native_family_count':len(manifest),'cards':len(cards),'edges':len(edges),'core_objects':len(core),'native_objects':len(native)},p)
    provenance._git=lambda *args:subprocess.check_output(['git','--no-optional-locks','-c',f'safe.directory={D.ROOT.as_posix()}',*args],cwd=D.ROOT,text=True).strip()
    provenance.model_revision=lambda _:D.SNAPSHOT;provenance.RESULTS=D.OUT/'provenance'
    audit=provenance.write_result('mlp_trigger_upstream',{'registry':registry,'bundles':archives,'scope':'real single-node results; remote upload receipt separate'},model_path=D.MODEL,seed=D.SEED,tool='mlp_trigger_publish')
    print({'registry':registry,'bundles':archives,'native_parts':len(raw_parts),'cards':len(cards),'edges':len(edges),'audit':str(audit)},flush=True)
if __name__=='__main__':main()
