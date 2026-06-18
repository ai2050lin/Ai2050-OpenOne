@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%time%] Starting qwen3 main test...
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase527_dc_decomposition.py qwen3 --n-fruit-objects 8 --n-test 15 > tests\glm5_temp\phase527_qwen3_log.txt 2>&1
echo [%time%] Done with exit code %errorlevel%
