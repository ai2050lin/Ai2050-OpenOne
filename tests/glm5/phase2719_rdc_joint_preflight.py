"""A bounded streamed fallback keeps the existing D-volume reserve without assuming a storage answer."""
from rdc_joint_common import *


def main():
    import psutil
    guard(1024**2, preflight=True)
    path = BASE / 'resource_allocation.json'
    if path.exists():
        print('EXISTING_JOINT_ALLOCATION', read(path), flush=True)
        return
    free = shutil.disk_usage(ROOT).free
    ceiling = 64 * 1024**2
    assert free - ceiling > D_FLOOR, ('Not enough for streamed result envelope', free, ceiling)
    config = {
        'timestamp': stamp(), 'status': 'streamed preflight allocation; full execution requires measured capture pilot',
        'authority': 'Current user research request; no permission inferred for C-drive writes or a lowered D-drive reserve.',
        'result_ceiling_bytes': ceiling, 'result_volume_floor_bytes': D_FLOOR,
        'workspace_volume_floor_bytes': D_FLOOR, 'initial_result_volume_free_bytes': free,
        'initial_available_host_bytes': psutil.virtual_memory().available,
        'engineer_selected_not_user_numeric_budget': True,
        'storage_mode': 'Lossless compressed materials; BF16 complete requested fields resident temporarily in RAM; retain full-coordinate summaries, predeclared representative fields, per-array hashes, frozen sources and exact recomputation entrypoints. No PCA/Top-K.',
        'pending_larger_storage_choice': 'C:/AI2050ResearchResults with project junction, or explicit D-reserve revision. Neither is assumed; answer can enable full raw archive without changing scientific splits.',
        'host_runtime_floor_bytes': 2 * 1024**3, 'maximum_cuda_models': 1,
        'capture_pilot_sources': 4, 'full_stream_memory_estimate_ceiling_bytes': 3 * 1024**3,
        'maximum_model_and_analysis_compute_seconds': 14400,
        'per_process_max_seconds': 5400,
        'retention': 'Do not delete old fields used by existing clients. New uncached raw fields may be discarded only after in-process analysis and hash/coverage checks, with full recipe. Selected displayed raw fields remain.',
    }
    save(path, config)
    save(BASE / 'preflight.json', {'timestamp': stamp(), 'source': snapshot(Path(__file__)),
        'material_urls_head_checked': {
            'en_gum_train': 16533561, 'en_gum_dev': 2410178,
            'en_gum_test': 2401409, 'zh_pud_test': 2169911},
        'source_storage': 'Download into memory, preserve gzip losslessly; no original source files deleted or overwritten.',
        'material_design_revision_before_any_new_model_output': 'English GUM train/dev/test with explicit document IDs and rich Entity/Discourse/Cxn labels; Chinese unused GSD training data for main and PUD only for external confirmation. PUD is not trained on, respecting its test-only role.',
        'allocation': config})
    print('STREAMED_PREFLIGHT_ALLOCATED', usage(), ceiling, free, flush=True)


if __name__ == '__main__':
    main()
