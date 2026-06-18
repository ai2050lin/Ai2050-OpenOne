@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%time%] Starting DS7B main test...
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase525_mid_layer_causality.py deepseek7b --n-fruit-objects 8 --n-test 10 > tests\glm5_temp\phase525_ds7b_log.txt 2>&1
echo [%time%] Done with exit code %errorlevel%
