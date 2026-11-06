"""Append native manifest annotation revisions, preserving all original tensors."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest,save_json
def main():
    store=TraceStore();roots=[P.OUT/'native',P.OUT/'extra-native',P.OUT/'polarity-native',P.OUT/'place-polarity/polarity-native',P.OUT/'operation-native']
    roots += [P.ROOT/f'tests/glm5/result/native_mlp_function_v{v}/audit-native' for v in [1,2,3]]
    changes=[]
    for root in roots:
        for path in sorted(root.glob('*/manifest.json')):
            old=P.read(path);d=dict(old);positions=d['readout_positions'];n=len(positions)
            roles={4:['before_selector','selector_partial_question','question_end','assistant_boundary'],3:['before_selector','selector_key','observation_end_before_question'],2:['question_end','assistant_boundary'],1:['assistant_boundary']}[n]
            d['readout_annotations']=[{'position':p,'visible_prefix_tokens':p+1,'role':role} for p,role in zip(positions,roles)]
            d['causal_visibility']='Each readout depends only on its actual prefix. The explicit readout_annotations supersede inherited template-specific prose. No generated answer is in the captured prefix.'
            if n in [2,4]:d['response_profile_stage_indices']=[2,3] if n==2 else [0,1,2,3]
            d['annotation_revision']={'version':2,'supersedes_trace_id':old['trace_id'],'original_manifest_sha256':file_sha(path),
                'reason':'Normalize readout roles; where inherited v1 prose was reused afterquestion moved in v3/v4, correct that prose. Numeric arrays/indices unchanged. Two-readout polarity traces are bothafterquestion; one-readout operation traces are atthe assistant boundary.',
                'numeric_data_unchanged':True,'source_sha256':file_sha(__file__)}
            d.pop('trace_id');d['trace_id']=digest(d);newpath=path.parent/'manifest-annotated.v2.json'
            if newpath.exists():assert P.read(newpath)==d
            else:save_json(newpath,d)
            store.register(newpath);changes.append({'old_trace_id':old['trace_id'],'trace_id':d['trace_id'],'run_id':d['run_id'],'sample_id':d['sample_id'],'path':str(newpath.relative_to(P.ROOT)),'deep':d['deep']})
    P.save(P.OUT/'visibility-revisions.json',{'kind':'real','run_id':P.RUN,'corrections':changes,'all_original_manifests_retained':True,'no_new_forward_passes':True,'provenance':P.read(P.OUT/'observations.json')['provenance']})
    print(json.dumps({'annotation_revisions':len(changes),'original_tensors_unchanged':True}))
if __name__=='__main__':main()
