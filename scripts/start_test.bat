@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Fine-tune Qwen3-0.6B - num_train_epochs=2 (test)

set "ROOT=D:\ScriptPython\Training"
set "SCRIPTS=%ROOT%\scripts"
set "PY=%ROOT%\.unsloth-env\Scripts\python.exe"
set "MODEL=%ROOT%\models\Qwen3-0.6B"
set "DATASET=%ROOT%\dataset_vn\merged_vn_uncensored_*.jsonl"
set "OUT=%ROOT%\output_lora"
set "LOG=%ROOT%\train_log.txt"

if not exist "%PY%" (
    echo [LOI] Khong tim thay python: %PY%
    pause
    exit /b 1
)

echo ============================================================
echo   FINETUNE TEST - num_train_epochs=2
echo   Model  : %MODEL%
echo   Dataset: %DATASET%
echo   Output : %OUT%
echo   Log    : %LOG%
echo ============================================================
echo.

"%PY%" "%SCRIPTS%\finetune_qwen3.py" --model "%MODEL%" --epochs 2 --dataset "%DATASET%" --output "%OUT%" > "%LOG%" 2>&1

echo.
if errorlevel 1 (
    echo [LOI] Finetune that bai (errorlevel %errorlevel%)
    pause
    exit /b 1
)
echo ==== HOAN TAT. Model merged tai: %ROOT%\output_lora_merged ====
pause
exit /b 0