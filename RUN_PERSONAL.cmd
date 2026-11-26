@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_shared_demo.ps1" -Mode Personal %*
if errorlevel 1 pause
