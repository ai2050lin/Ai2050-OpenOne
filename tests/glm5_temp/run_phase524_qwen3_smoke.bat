@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%time%] Starting qwen3 smoke test...
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase524_semantic_selectivity.py qwen3 --smoke > tests\glm5_temp\phase524_qwen3_smoke_log.txt 2>&1
echo [%time%] Done with exit code %errorlevel%
