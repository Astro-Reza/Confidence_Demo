@echo off
title Antenna Pointing 3D Simulation ^& Confidence Sensor Fusion
echo =====================================================================
echo  Antenna Pointing System: 3D Kinematics ^& Confidence Sensor Fusion
echo =====================================================================
echo Launching Interactive 2-Page Simulation Application...
python "%~dp0antenna_fusion_app.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
