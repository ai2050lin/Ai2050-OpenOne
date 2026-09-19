"""Preserve the first population-head preflight failure and its exact source."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_operator_common import *

out=BASE/'metric_followup/population_geometry'
save(out/'initial_head_load_failure.json',{'timestamp':stamp(),
    'failed_source':snapshot(ROOT/'tests/glm5/phase2727_rdc_operator_population_geometry.py'),
    'observed_exception':"KeyError: 'lm_head.weight'",
    'confirmed_checkpoint':'qwen3-4b/config.json tie_word_embeddings=true; index contains model.embed_tokens.weight but no independent lm_head.weight.',
    'readout_weights_loaded':False,'population_result_arrays_written':len(list(out.glob('*.npz'))),
    'tool_elapsed_seconds':3.066143,'timing_scope':'Actual tool-reported process wall time including startup, not GPU-only time.',
    'impact':'The full-model metric used the actual model.lm_head and is unaffected. Only the independent-head reader failed before producing population estimates.'})
ledger('population_geometry_head_preflight_failure_tool_elapsed',3.066143)
