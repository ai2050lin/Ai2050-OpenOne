"""Recompute native file hashes and expose real generation-step fields, no inference."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P
from deploy.research_library import Library


def floats(bits):return (bits.astype(np.uint32)<<16).view(np.float32)


def main():
    W=P.W;c=W.read(P.OUT/'corpus.json');execution=W.read(P.OUT/'execution.json');lib=Library(W.ROOT/'distributed_data/research_library')
    assert W.sha(Path(P.N.__file__))==W.read(P.OUT/'freeze.json')['runner_sha256']
    byid={r['id']:r for r in c['items']};old=np.load(W.ROOT/'tests/glm5/result/phase2773_w6_v3/selected-edit.npz')
    for s in execution['summaries']:
        assert s['weights_restored'] and s['model_snapshot']==c['model_snapshot']
        for t in s['edit_metadata']['target']:
            assert P.N.hashlib.sha256(old[f'l{t["block"]}_edited_bits'].tobytes()).hexdigest()==t['edited_bits_sha256']
    seen=set();files=[];bytes_=steps=0;preview=[];traces=[]
    chosen={r['id'] for r in c['items'] if r['family_id']=='nat4-5000' and r['frame']==0 and r['length']==1024}
    units={l:c['edit']['columns'][str(l)][0] for l in range(18,28)}
    for task in sorted((P.OUT/'node-tasks').glob('*')):
        if not (task/'outputs/summary.json').exists():continue
        summary=W.read(task/'outputs/summary.json');manifest=W.read(task/'outputs/native-manifest.json')
        assert P.N.digest(manifest)==summary['native_manifest_sha256']
        native=P.OUT/'native'/execution['template']['design_sha']/summary['shard'].replace('/','-')
        outputs={(r['sample_id'],r['condition']):r for r in W.read(task/'outputs/samples.json')['outputs']}
        for meta in manifest['files']:
            p=native/meta['path'];assert W.sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes']
            with np.load(p) as z:
                b=len(z['ids']);t=meta['generation_steps']
                for name,shape in [('prefix_hidden_bits',(b,37,5,2560)),('generation_hidden_bits',(b,t,37,2560)),
                    ('prefix_mlp_bits',(b,10,5,9728)),('generation_mlp_bits',(b,t,10,9728))]:
                    assert z[name].dtype==np.uint16 and z[name].shape==shape,(p.name,name,z[name].shape)
                for j,sid in enumerate(z['ids'].tolist()):
                    key=(sid,meta['condition']);assert key not in seen;seen.add(key)
                    valid=int(z['valid_generation_steps'][j]);assert valid==len(outputs[key]['token_ids'])<=t;steps+=valid
                    assert z['prefix_positions'].tolist()==[0,byid[sid]['tokens']//4,byid[sid]['tokens']//2,byid[sid]['tokens']-2,byid[sid]['tokens']-1]
                    if sid not in chosen:continue
                    src=byid[sid];output=outputs[key];record={'id':sid+':'+meta['condition'],'entity':src['register']+' · '+meta['condition'],
                        'split':src['split'],'tokens':src['tokens'],'text':src['text'],'condition':meta['condition'],'trace_responses':{}}
                    hs=floats(z['generation_hidden_bits'][j,:valid]);ms=floats(z['generation_mlp_bits'][j,:valid])
                    assert np.isfinite(hs).all() and np.isfinite(ms).all()
                    traces.append({'sample_id':sid,'condition':meta['condition'],'generated_token_ids':output['token_ids'],
                        'hidden_rms':np.sqrt(np.mean(hs**2,axis=-1)).tolist(),'mlp_rms':np.sqrt(np.mean(ms**2,axis=-1)).tolist(),
                        'native_sha256':meta['sha256']})
                    for li,l in enumerate(range(18,28)):
                        record['trace_responses'][str(l)]=[]
                        for step in sorted({0,min(5,valid-1),min(10,valid-1),valid-1}):
                            values=ms[step,li];doc={'schema':'research_mlp_response.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
                                'block':l,'boundary':'mlp.down_proj.input','readout':'generation','selected_unit':units[l],
                                'sample_id':sid,'text':src['text'],'tokens':src['tokens']+step,'values':values.tolist(),
                                'response_state':'unedited baseline' if meta['condition']=='base' else 'edited alpha6',
                                'generation_step':step,'prefix_token_ids':src['token_ids']+output['token_ids'][:step],
                                'predicted_token_id':output['token_ids'][step],'native_file_sha256':meta['sha256'],
                                'source_edit_sha256':c['source_edit_sha256'],'provenance':c['provenance']}
                            path=P.PUBLIC/'responses'/f'{sid}-{meta["condition"]}-B{l}-G{step}.research.json';W.save(path,doc,True)
                            ref=lib.ingest(doc,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
                            record['trace_responses'][str(l)].append({'step':step,'sha256':ref,'token_id':output['token_ids'][step]})
                    preview.append(record)
            files.append({**meta,'shard':summary['shard']});bytes_+=meta['bytes']
    assert len(files)==96 and len(seen)==192 and {sid for sid,_ in seen}==set(byid)
    audit={'schema':'research_generation_native_audit.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'files':files,'verified_files':len(files),'unique_sample_conditions':len(seen),'valid_generation_steps':steps,'bytes':bytes_,
        'full_hidden_layers':37,'full_mlp_units':9728,'mlp_blocks':list(range(18,28)),
        'prefix_positions_per_input':5,'all_prefix_tokens_captured':False,'all_actual_generation_steps_captured':True,
        'edit_hashes_match_previous_capability_run':True,'matched_blocks_across_shards':80,'physical_nodes':1,
        'alignment':'step0 predicts first generated token; step t uses actual generated tokens[:t]. Base/target prefixes may diverge after step0, so cross-condition state differences are descriptive.',
        'sample_preview':preview,'traces':traces,'provenance':c['provenance']}
    W.save(P.OUT/'native-audit.json',audit,True)
    P.status('native_audit',files=len(files),sample_conditions=len(seen),valid_steps=steps,bytes=bytes_,edit_hashes_match=True)
    print({'files':len(files),'bytes':bytes_,'valid_steps':steps,'preview':len(preview)})


if __name__=='__main__':main()
