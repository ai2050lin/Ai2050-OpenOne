@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%date% %time%] Starting DS7B Phase523... > tests\glm5_temp\phase523_ds7b_log.txt
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase523_subspace_mapping.py deepseek7b >> tests\glm5_temp\phase523_ds7b_log.txt 2>&1
echo [%date% %time%] Done with exit code %errorlevel% >> tests\glm5_temp\phase523_ds7b_log.txt
