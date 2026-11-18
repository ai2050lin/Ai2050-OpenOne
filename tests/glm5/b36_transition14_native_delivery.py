"""Run the unchanged frozen analysis with one metadata-writer compatibility fix.

The feature card supplies model_snapshot, which S.save also supplies. Remove only
that identical redundant key before writing; no numerical computation, threshold,
source hash check, or evidence decision is replaced.
"""
import b36_transition14_native_analyze as A
S=A.S


def run():
    S.configure()
    p=S.read(S.OUT/'native-plan.json')
    original=S.ROOT/'tests/glm5/b36_transition14_native_analyze.py'
    assert S.file_sha(original)==p['source_hashes'][original.name]
    original_save=S.save
    original_save('native-analysis-execution-amendment.json',dict(
        frozen_analysis_sha256=S.file_sha(original),wrapper_sha256=S.file_sha(__file__),
        reason='Static review found duplicate model_snapshot keyword in feature-card metadata: the card and shared S.save both supply it.',
        change='Remove only the identical model_snapshot key from native-feature-card.json payload before invoking the unchanged S.save; S.save writes the same value.',
        preserved='Frozen numerical algorithm, source hashes, raw arrays, all metrics, thresholds, gates and statistical comparisons unchanged.',
        discovery='Found before executing the native analysis or inspecting confirmation intervention effects; only numerical pilot calibration status had been inspected.'))
    def compatible_save(name,data):
        if name=='native-feature-card.json':
            assert data['model_snapshot']==S.SNAPSHOT
            data=dict(data);del data['model_snapshot']
        return original_save(name,data)
    S.save=compatible_save
    try:A.run()
    finally:S.save=original_save


if __name__=='__main__':run()
