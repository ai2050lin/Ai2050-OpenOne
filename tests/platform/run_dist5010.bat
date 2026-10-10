@echo off
rem AI2050 :5010 distributed service launcher (run by Task Scheduler, outside session tree)
cd /d D:\AI2050\Ai2050-OpenOne\deploy
set AI2050_DIST_DIR=D:\AI2050\Ai2050-OpenOne\tests\dist_test_data
"D:\AI2050\Ai2050-OpenOne\.venv\Scripts\python.exe" -m distributed_service >> "D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010.log" 2>&1
