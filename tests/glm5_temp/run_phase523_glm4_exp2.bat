@echo off
cd /d d:\AI2050\Ai2050-OpenOne
echo [%date% %time%] Starting GLM4 Exp2... > tests\glm5_temp\phase523_glm4_exp2_log.txt
C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe -u tests\glm5\phase523_subspace_mapping.py glm4 --skip-exp1 >> tests\glm5_temp\phase523_glm4_exp2_log.txt 2>&1
echo [%date% %time%] Done with exit code %errorlevel% >> tests\glm5_temp\phase523_glm4_exp2_log.txt
