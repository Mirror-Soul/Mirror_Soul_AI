@echo off
setlocal
set "SCRIPT=%~dp0ai_pipeline_monitor.py"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%SCRIPT%" %*
    exit /b %errorlevel%
)

where python >nul 2>nul
if not errorlevel 1 (
    python "%SCRIPT%" %*
    exit /b %errorlevel%
)

set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BUNDLED_PYTHON%" (
    "%BUNDLED_PYTHON%" "%SCRIPT%" %*
    exit /b %errorlevel%
)

echo ERROR: Python 3 was not found.
exit /b 2
