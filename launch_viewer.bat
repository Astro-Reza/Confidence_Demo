@echo off
title Sensor Fusion CSV Graph Viewer
echo Launching Interactive Sensor Fusion CSV Viewer...
python "%~dp0csv_viewer.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
