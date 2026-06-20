@echo off
REM Phase 559 Smoke Test: minimal params, 3 models sequential
REM 3 conditions x 1 route x 1 seed x 4 tokens x test_n=4
REM Verifies script correctness + generation + object audit

set PY=C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe
set OUT=results/glm5_phase559_smoke
set COND=baseline,resid_remove_perp,resid_donor_vehicle_mean_cache_add,resid_donor_vehicle_random_cache_add
set ROUTES=forbidden_sentence_completion:temperature<-forbidden_definition

echo ================================================================
echo Phase 559 SMOKE TEST - Round 1
echo ================================================================

echo.
echo === Qwen3 Smoke ===
%PY% tests/glm5/phase559_prototype_generation_closure.py qwen3 ^
  --windows 10,12,14 ^
  --pair vehicle_tool ^
  --train-n 8 --test-n 4 ^
  --sample-seeds 101 ^
  --routes %ROUTES% ^
  --conditions %COND% ^
  --layer-sets all ^
  --max-new-tokens 4 ^
  --batch-size 4 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo === GLM4 Smoke ===
%PY% tests/glm5/phase559_prototype_generation_closure.py glm4 ^
  --windows 24,26,28 ^
  --pair vehicle_tool ^
  --train-n 8 --test-n 4 ^
  --sample-seeds 101 ^
  --routes %ROUTES% ^
  --conditions %COND% ^
  --layer-sets all ^
  --max-new-tokens 4 ^
  --batch-size 4 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo === DS7B Smoke ===
%PY% tests/glm5/phase559_prototype_generation_closure.py deepseek7b ^
  --windows 16,18,20 ^
  --pair vehicle_tool ^
  --train-n 8 --test-n 4 ^
  --sample-seeds 101 ^
  --routes %ROUTES% ^
  --conditions %COND% ^
  --layer-sets all ^
  --max-new-tokens 4 ^
  --batch-size 4 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo ================================================================
echo Phase 559 Smoke Test COMPLETE
echo ================================================================
