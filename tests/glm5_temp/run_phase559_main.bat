@echo off
REM Phase 559 Main Test: 11 conditions x 2 routes x 6 seeds x 12 tokens x test_n=12
REM Three models sequential: qwen3 -> glm4 -> deepseek7b
REM With test_n=12: repeat2=helicopter, repeat4=rocket

set PY=C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe
set OUT=results/glm5_phase559_prototype_generation_closure
set ROUTES=forbidden_sentence_completion:temperature<-forbidden_definition,forbidden_definition:top_p<-forbidden_definition

echo ================================================================
echo Phase 559 MAIN TEST - Round 2
echo 11 conditions x 2 routes x 6 seeds x 12 tokens x test_n=12
echo ================================================================

echo.
echo === Qwen3 Main ===
%PY% tests/glm5/phase559_prototype_generation_closure.py qwen3 ^
  --windows 10,12,14 ^
  --pair vehicle_tool ^
  --train-n 12 --test-n 12 ^
  --sample-seeds 101,103,107,109,113,127 ^
  --routes %ROUTES% ^
  --layer-sets all ^
  --max-new-tokens 12 ^
  --batch-size 12 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo === GLM4 Main ===
%PY% tests/glm5/phase559_prototype_generation_closure.py glm4 ^
  --windows 24,26,28 ^
  --pair vehicle_tool ^
  --train-n 12 --test-n 12 ^
  --sample-seeds 101,103,107,109,113,127 ^
  --routes %ROUTES% ^
  --layer-sets all ^
  --max-new-tokens 12 ^
  --batch-size 12 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo === DS7B Main ===
%PY% tests/glm5/phase559_prototype_generation_closure.py deepseek7b ^
  --windows 16,18,20 ^
  --pair vehicle_tool ^
  --train-n 12 --test-n 12 ^
  --sample-seeds 101,103,107,109,113,127 ^
  --routes %ROUTES% ^
  --layer-sets all ^
  --max-new-tokens 12 ^
  --batch-size 12 ^
  --output-dir %OUT% ^
  --hard-exit-after-model

echo.
echo ================================================================
echo Phase 559 Main Test COMPLETE
echo ================================================================
