@echo off
rem AI2050 client (visualization frontend) one-click launcher.
rem Double-click this file, or run it from any terminal. No VSCode required.
rem The window stays open while the client runs; close it to stop the client.
title AI2050 Client (http://localhost:5173)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_visualization.ps1" %*
echo.
echo [AI2050] Client stopped. Press any key to close this window.
pause >nul
