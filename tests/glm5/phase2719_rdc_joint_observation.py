"""One bounded capture-to-analysis transaction; release replayable fields only after checks."""
from rdc_joint_common import *
from rdc_joint_capture import capture, ledger
from rdc_joint_layer_atlas import layer_atlas
from rdc_joint_prior_confirmation import prior_confirmation
from rdc_joint_relation_atlas import relation_atlas


def main():
    assert read(BASE/'preflight_checks.json')['passed']
    out = BASE/'observation'
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'material_sha':sha(material_path()),'fresh_sha':sha(material_path(True)),
        'fresh_responses_not_used':True,'new_training_occurs_only_after_frozen_prior_rule_evaluation':True,
        'preregistration':'Source scripts snapshotted before full main field analysis; thresholds and routes in these files.',
        'memory_policy':'Capture full requested arrays in RAM; discard nonfixture RAM only after results, immutable array commits, exact replay recipe and checks.'})
    for name in ('rdc_joint_layer_atlas.py','rdc_joint_prior_confirmation.py','rdc_joint_relation_atlas.py'):
        snapshot(ROOT/'tests/glm5'/name)
    cache = capture()
    start = time.monotonic()
    try:
        prior_confirmation(cache)
        layer_atlas(cache)
        relation_atlas(cache)
        save(out/'result.json',{'timestamp':stamp(),'status':'completed','sources':len(cache.material),
            'bytes_full_fields_in_RAM':cache.bytes,'fresh_capture_performed':False,
            'analyses':{k:sha(BASE/k/'result.json') for k in ('prior_confirmation','layer_atlas','relation_atlas')},
            'nonfixture_raw_retention':'Released from temporary RAM after analysis; all arrays hash-committed, material/model/config/code and replay entrypoint retained; four full main fixtures remain for client.'})
    finally:
        cache.clear()
    ledger('phase2719_full_coordinate_analysis',time.monotonic()-start,sources=512)
    guard()
    print('JOINT_OBSERVATION_COMPLETE',usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
