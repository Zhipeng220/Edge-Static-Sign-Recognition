@echo off
setlocal
cd /d "%~dp0\.."

set "PREDICTIONS=%~1"
if "%PREDICTIONS%"=="" set "PREDICTIONS=multi_dataset_results_fixed_labels\asl_large_dataset\predictions_full_model_seed42.csv"

set "OUTPUT_DIR=%~2"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=results\evaluation\asl_large_dataset_full_model_seed42"

python src\evaluate.py --predictions "%PREDICTIONS%" --output-dir "%OUTPUT_DIR%" --overwrite
