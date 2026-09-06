"""Actual post-cleanup terminal audit; missing evidence fails, never fabricates QA."""
import argparse, ast, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT, MEMO, read, save, sha, re

OUT = RESULT/'phase2684_source_campaign_delivery'
CROSS = RESULT/'phase2683_crossmodel_function_atlas'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-final', action='store_true')
    args = parser.parse_args()
    checks = {}
    reports = {name: read(OUT/f'analysis/{name}.json') for name in
               ('scientific_checks', 'delivery_checks', 'live_api_checks',
                'browser_checks', 'post_cleanup_checks')}
    for name, report in reports.items():
        checks[name] = report['all_checks_passed'] is True and not report.get('preview_only', False)
    checks['actual_HTTP_before_and_after_cleanup'] = all(
        reports[name].get('real_HTTP') is True for name in ('live_api_checks', 'post_cleanup_checks'))
    checks['post_cleanup_HTTP_is_post_cleanup'] = reports['post_cleanup_checks'].get('post_cleanup') is True
    # This flag must be written by the task only after actual CUA inspection.
    checks['actual_browser_explicitly_recorded'] = reports['browser_checks'].get('real_browser') is True
    checks['production_build_passed'] = read(OUT/'analysis/build_output.json')['returncode'] == 0

    last = 2684 if args.post_final else 2683
    for number in range(2677, last+1):
        paths = list(RESULT.glob(f'phase{number}_*/analysis/final.json'))
        assert len(paths) == 1, (number, paths)
        result = read(paths[0])
        checks[f'phase{number}_actual_complete_not_semantic_closure'] = (
            result['phase'] == number and result['all_checks_passed'] is True
            and result['language_mechanism_closed'] is False)
    text = MEMO.read_text(encoding='utf-8-sig')
    headings = re.findall(r'^## Phase (\d+):[^\n]*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]$', text, re.M)
    selected = [(int(number), stamp) for number, stamp in headings if 2677 <= int(number) <= last]
    checks['campaign_MEMO_once_contiguous_timestamped'] = [n for n, _ in selected] == list(range(2677, last+1))
    checks['actual_MEMO_frontier'] = int(headings[-1][0]) == last

    publication = read(OUT/'analysis/publication.json')
    catalog = OUT/'material/client_panel_catalog.json'
    checks['final_published_catalog_hash'] = (publication['preview_only'] is False
        and sha(catalog) == publication['publication_manifest_sha256'])
    checks['128_published_source_cases'] = len(read(OUT/'material/published_source_cases.json')) == publication['source_cases'] == 128
    clean = read(OUT/'analysis/cleanup_completed.json')
    targets = clean['targets']; kept = clean['retained_referenced_files']
    referenced = {Path(r['path']).resolve() for r in publication['referenced_files']}
    allowed_parents = {(RESULT/name/'field').resolve() for name in
                       ('phase2678_padded_source_field', 'phase2679_native_source_ledger')}
    checks['8832_exact_unpublished_targets_absent'] = len(targets) == 8832 and all(
        Path(r['path']).resolve().parent in allowed_parents
        and re.fullmatch(r'case_\d{4}\.npz', Path(r['path']).name)
        and not r['published'] and not Path(r['path']).exists()
        and Path(r['path']).resolve() not in referenced for r in targets)
    checks['deleted_byte_accounting'] = clean['deleted_bytes'] == sum(r['bytes'] for r in targets)
    checks['every_referenced_array_retained_hashed'] = (
        {Path(r['path']).resolve() for r in kept} == referenced
        and all(Path(r['path']).is_file() and Path(r['path']).stat().st_size == r['bytes']
                and sha(r['path']) == r['sha256'] for r in kept))
    checks['cleanup_report_matches_actual_state'] = clean['all_deleted'] is True and clean['all_retained'] is True
    asset = RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
    checks['legacy_asset_unchanged'] = sha(asset) == '4e15b56f30a89f5f523ddea4b35ab46394f2a9a9015ac565b1672b25142207cb'

    frozen = read(CROSS/'protocol/frozen.json')
    checks['crossmodel_frozen_material_and_calibration'] = all(
        sha(CROSS/key/'material/cases.json') == spec['material_sha256']
        and sha(CROSS/key/'material/calibration.json') == spec['calibration_sha256']
        for key, spec in frozen['models'].items())
    nxt = read(OUT/'analysis/next_campaign.json')
    checks['same_goal_whole_next_campaign'] = (
        nxt['same_goal'] is True and nxt['created_after_complete_results'] is True
        and sorted(map(int, nxt['phases'])) == list(range(2685, 2694)))

    paths = sorted({p for folder in (ROOT/'tests/glm5', ROOT/'tests/glm5_temp')
                    for number in range(2677, 2685) for p in folder.glob(f'phase{number}_*.py')})
    paths += [ROOT/'server'/name for name in
              ('native_source_parameter_query.py', 'native_atlas_heatmap_query.py', 'research_asset_service.py')]
    hashes = {}
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'))
        hashes[str(path.relative_to(ROOT))] = sha(path)
    checks['scoped_source_AST_valid'] = bool(hashes)
    changed = list(hashes) + ['research/glm5/docs/AGI_GLM5_MEMO.md'] + [
        f'frontend/src/components/app/{name}' for name in
        ('NativeSourceParameterInspector.jsx', 'NativeParameterInspector.jsx', 'NativeAtlasHeatmap.jsx')]
    diff = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                           'diff', '--check', '--', *changed], cwd=ROOT, capture_output=True, text=True)
    checks['scoped_diff_whitespace'] = diff.returncode == 0
    report = {'all_checks_passed': all(checks.values()), 'checks': checks,
              'post_final': args.post_final, 'memo_headings': selected,
              'deleted_bytes': clean['deleted_bytes'], 'source_hashes_at_audit': hashes,
              'scoped_diff_output': diff.stdout+diff.stderr, 'mechanism_closed': False,
              'scope': 'Actual evidence and retained bytes, not new model forwards or semantic closure. Source hashes describe audit versions, not historical source immutability.'}
    save(OUT/f'analysis/terminal_{"post_final" if args.post_final else "audit"}.json', report)
    print(checks, flush=True)
    assert report['all_checks_passed'], [key for key, value in checks.items() if not value]


if __name__ == '__main__':
    main()
