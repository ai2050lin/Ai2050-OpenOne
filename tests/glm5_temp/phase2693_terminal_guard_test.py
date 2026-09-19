"""Actual safety tests while crossmodel completion is still pending.

These are workflow tests, not new model experiments or delivery completion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'glm5'))
import phase2693_campaign_terminal as terminal
import phase2692_finalize_with_rounding as linked_finalizer
from phase2620_native_coordinate_contract import ROOT, RESULT, MEMO, read, save, sha


def main():
    assert not (terminal.CROSS / 'analysis/final.json').exists(), 'This pending-phase test is not for completed science'
    protected = [MEMO, terminal.OUT / 'material/staged_client_panel_catalog.json']
    before = {str(p): sha(p) for p in protected}
    future = [terminal.OUT / f'analysis/{name}.json' for name in (
        'final', 'scientific_checks', 'terminal_audit', 'storage_inventory')]
    future += [terminal.LINK / 'analysis/final.json', terminal.LINK / 'analysis/numerical_review_append.json']
    assert not any(p.exists() for p in future)
    observed = []
    for name, function in [('science_requires_actual2691', terminal.scientific_review),
                           ('2692_wrapper_waits_before_loading_or_writing', linked_finalizer.main),
                           ('preview_report_cannot_be_actual_delivery', lambda: terminal.checked_report('staged_panel_direct_audit')),
                           ('final_requires_contiguous2692_before_any_write', terminal.finalize),
                           ('actual_storage_requires_real_science_browser', lambda: terminal.storage_inventory(False))]:
        try:
            function()
        except (AssertionError, FileNotFoundError) as exc:
            observed.append({'test': name, 'actually_rejected': True, 'exception': type(exc).__name__, 'detail': str(exc)})
        else:
            raise AssertionError(name + ' accepted incomplete work')
        assert not any(p.exists() for p in future)
    for invalid in (ROOT, ROOT / 'models/hf/qwen3-4b/config.json', RESULT / '../outside_result.json'):
        try:
            terminal.checked_path(invalid)
        except AssertionError:
            observed.append({'test': 'path_stays_in_existing_result_file', 'path': str(invalid), 'actually_rejected': True})
        else:
            raise AssertionError('Unsafe target accepted')
    assert before == {str(p): sha(p) for p in protected}
    assert not any(p.exists() for p in future)
    save(terminal.OUT / 'analysis/staged_terminal_guard_test.json', {
        'all_checks_passed': True, 'preview_only': True, 'phase_completed': False,
        'actual_safety_tests': observed, 'memo_and_catalog_unchanged': before,
        'future_phase_files_created': 0, 'new_model_forwards': 0,
        'source_sha256': sha(Path(terminal.__file__)), 'test_sha256': sha(Path(__file__)),
        'boundary': 'Testing rejection of incomplete work; does not claim completed science, HTTP or real browser QA.'})
    print('2693 actual safety guards passed:', len(observed), 'rejections; future completion files0', flush=True)


if __name__ == '__main__': main()
