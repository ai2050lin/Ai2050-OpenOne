@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%time%] Starting qwen3 smoke test...
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase526_activation_verification.py qwen3 --smoke > tests\glm5_temp\phase526_qwen3_smoke_log.txt 2>&1
echo [%time%] Done with exit code %errorlevel%
