@echo off
REM ========================================
REM TransformerLens AGI Lab - 环境部署启动脚本
REM ========================================

setlocal enabledelayedexpansion

REM --- 路径配置 ---
set PYTHON_DIR=C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9
set PYTHON_PATH=%PYTHON_DIR%\python.exe
set NODE_DIR=C:\Users\Admin\.workbuddy\binaries\node\versions\22.21.0
set PROJECT_ROOT=%~dp0

REM 设置 PATH
set PATH=%PYTHON_DIR%\Scripts;%PYTHON_DIR%;%NODE_DIR%;%PATH%

REM 设置环境变量
set HF_HOME=D:\develop\model
set HF_ENDPOINT=https://hf-mirror.com
set TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

echo ========================================
echo TransformerLens AGI Lab - 启动中...
echo ========================================
echo.
echo Python: %PYTHON_PATH%
echo Node.js: %NODE_DIR%\node.exe
echo.

REM 检查 Python
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python not found at %PYTHON_PATH%
    exit /b 1
)

REM 检查 Node.js
if not exist "%NODE_DIR%\node.exe" (
    echo [ERROR] Node.js not found at %NODE_DIR%\node.exe
    exit /b 1
)

echo [1/2] Starting Backend Server (port 5001)...
start "AGI-Backend" cmd /c "cd /d %PROJECT_ROOT% && %PYTHON_PATH% -m uvicorn server.server:app --host 0.0.0.0 --port 5001 --log-level warning"

REM 等待后端启动
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Frontend Dev Server (port 5173)...
start "AGI-Frontend" cmd /c "cd /d %PROJECT_ROOT%frontend && %NODE_DIR%\node.exe %NODE_DIR%\node_modules\npm\bin\npm-cli.js run dev"

echo.
echo ========================================
echo 启动完成！
echo   后端: http://localhost:5001
echo   前端: http://localhost:5173
echo   API文档: http://localhost:5001/docs
echo ========================================
echo.
pause
