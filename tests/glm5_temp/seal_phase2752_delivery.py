"""Delivery provenance; records after-the-fact hashes honestly, never backdates."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2752_context_interaction import OUT,sha,write,now,snapshot


def main():
    import numpy
    import torch
    import transformers
    import transformers.models.qwen3.modeling_qwen3 as implementation
    def read(p):
        return json.loads(p.read_text(encoding='utf-8'))
    write(OUT/'resource_events.json',dict(recorded_utc=now(),events=[
        dict(kind='pre_main_material_revision',preserved_path='pre_main_design_v1',successful_pilot_prompts=16,
             reason='Before formal capture/analysis, changed training depth from constant2 to1/2 so depth3 is a meaningful extrapolation; original calibration data retained.'),
        dict(kind='failed14B_load',path='14B_pilot_failed_load',log='14B_pilot_failed_load_stdout.txt',
             observed='Process ended with exit1 while loading weights; no traceback or capture chunks. Cause not established.',
             action='Retried standalone after4B analysis; pilot then completed16 rows.'),
        dict(kind='semantic_scoring_control',path='assertion_control',
             reason='Primary4B negative-question first-token results exposed pragmatic ambiguity. Replaced questions by explicit assertion truth classification, same worlds/splits,4B only.',
             evidence_status='Controlled follow-up after primary4B results, not independent corpus replication')]))
    write(OUT/'runtime_identity.json',dict(recorded_utc=now(),scope='Environment identity collected at delivery, not a backdated pre-run seal',
        python=sys.version,numpy=numpy.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        qwen_implementation=dict(path=str(Path(implementation.__file__)),sha256=sha(Path(implementation.__file__)))))
    checkpoints={}
    for side,model in [('4B','qwen3-4b'),('14B','Qwen3-14B')]:
        mdir=ROOT/'models/hf'/model
        checkpoints[side]={p.name:sha(p) for p in sorted(mdir.glob('*.safetensors'))}
    write(OUT/'checkpoint_identity.json',dict(recorded_utc=now(),scope='Checksums at delivery. Model paths/config IDs were recorded before each run; repeated pilot identity separately verified.',models=checkpoints))
    # Original4B diagnostic predates a metadata-only source change; its old code
    # content is already preserved by the original diagnostic design seal.
    old=read(OUT/'source_ablation_design.json')['source']
    if not (OUT/'4B/source_ablation_execution.json').exists():
        write(OUT/'4B/source_ablation_execution.json',dict(recorded_utc=now(),scope='Retrospective source identity from the existing4B diagnostic seal, not a fabricated pre-run timestamp',source=old))
    sources=[snapshot(p) for p in sorted((ROOT/'tests/glm5').glob('phase2752_*.py'))]
    sources+=[snapshot(ROOT/'tests/glm5/test_context_interaction_service.py'),snapshot(ROOT/'server/rdc_context_interaction_service.py')]
    files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write(OUT/'artifact_manifest.json',dict(recorded_utc=now(),sources_at_delivery=sources,artifacts=files,
        proof_scope='Integrity inventory, not a substitute for scientific validity; execution-time code snapshots retained separately.'))
    print(json.dumps(dict(artifacts=len(files),source_files=len(sources),total_bytes=sum(v['bytes'] for v in files.values()))))


if __name__=='__main__':
    main()
