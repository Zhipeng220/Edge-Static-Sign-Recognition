@echo off
setlocal
cd /d "%~dp0\.."

set "SCALER=%~1"
if "%SCALER%"=="" set "SCALER=models\scaler_full_model_seed42.pkl"

set "OUTPUT_DIR=%~2"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=deployment\jetson"

python src\export_scaler_params.py --scaler "%SCALER%" --output-dir "%OUTPUT_DIR%"
