@echo off
title Sensor Fusion CSV Graph Viewer
echo Launching Interactive Sensor Fusion CSV Viewer...
rem Resolve an optional CSV argument before changing directory
set "CSV_ARG="
if not "%~1"=="" set "CSV_ARG=%~f1"
pushd "%~dp0"
if defined CSV_ARG (
    python -m antenna_fusion.log_viewer "%CSV_ARG%"
) else (
    python -m antenna_fusion.log_viewer
)
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% NEQ 0 (
    echo.
    echo Application exited with error code %EXITCODE%.
    pause
)
