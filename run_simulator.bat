@echo off
title Antenna Pointing 3D Simulation ^& Confidence Sensor Fusion
echo =====================================================================
echo  Antenna Pointing System: 3D Kinematics ^& Confidence Sensor Fusion
echo =====================================================================
echo Launching Interactive 2-Page Simulation Application...
pushd "%~dp0"
python -m antenna_fusion.simulator_app %*
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% NEQ 0 (
    echo.
    echo Application exited with error code %EXITCODE%.
    pause
)
