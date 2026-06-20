@echo off
cd /d d:\AI2050\Ai2050-OpenOne

echo [%date% %time%] Starting qwen3 test...
"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe" tests\glm5\phase548_paraphrase_robustness_audit.py qwen3 --train-n 8 --test-n 6 --alpha 8 --random-seeds 11,23,37 --max-new-tokens 10 --full-pairs --output-dir results/glm5_phase548_paraphrase_robustness_audit --hard-exit-after-model > tests\glm5_temp\phase548_qwen3.log 2>&1
echo [%date% %time%] qwen3 done. Exit code: %errorlevel%

echo [%date% %time%] Starting GLM4 test...
"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe" tests\glm5\phase548_paraphrase_robustness_audit.py glm4 --train-n 8 --test-n 6 --alpha 8 --random-seeds 11,23,37 --max-new-tokens 10 --output-dir results/glm5_phase548_paraphrase_robustness_audit --hard-exit-after-model > tests\glm5_temp\phase548_glm4.log 2>&1
echo [%date% %time%] GLM4 done. Exit code: %errorlevel%

echo [%date% %time%] Starting DS7B test...
"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe" tests\glm5\phase548_paraphrase_robustness_audit.py deepseek7b --train-n 8 --test-n 6 --alpha 8 --random-seeds 11,23,37 --max-new-tokens 10 --output-dir results/glm5_phase548_paraphrase_robustness_audit --hard-exit-after-model > tests\glm5_temp\phase548_ds7b.log 2>&1
echo [%date% %time%] DS7B done. Exit code: %errorlevel%

echo All tests complete!
