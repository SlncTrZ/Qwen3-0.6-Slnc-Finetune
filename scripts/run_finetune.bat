@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Fine-tune QLoRA - Interactive

set "ROOT=D:\ScriptPython\Training"
set "SCRIPTS=%ROOT%\scripts"
set "PY=%ROOT%\.unsloth-env\Scripts\python.exe"
set "DATASET=%ROOT%\dataset_vn\merged_vn_uncensored_*.jsonl"

if not exist "%PY%" (
    echo [LOI] Khong tim thay python trong .unsloth-env: %PY%
    pause
    exit /b 1
)

echo ============================================================
echo   FINE-TUNE QLORA - TUONG TAC
echo   Dataset mac dinh: %DATASET%
echo ============================================================
echo.

REM ============ 1. QUET DANH SACH MODEL ============
echo ==== CHON MODEL CAN FINETUNE ====
set /a idx=0
for /d %%d in ("%ROOT%\models\*") do (
    set /a idx+=1
    set "model_!idx!=%%~nxd"
    echo   [!idx!] %%~nxd
)
if %idx%==0 (
    echo [LOI] Khong co model nao trong %ROOT%\models
    pause
    exit /b 1
)
echo.
set /p sel="Nhap so model (1-%idx%): "
set "MODEL_NAME="
set /a n=0
for /d %%d in ("%ROOT%\models\*") do (
    set /a n+=1
    if !n! equ %sel% set "MODEL_NAME=%%~nxd"
)
if "!MODEL_NAME!"=="" (
    echo [LOI] Lua chon khong hop le
    pause
    exit /b 1
)
set "MODEL_PATH=%ROOT%\models\!MODEL_NAME!"
echo   - Da chon: !MODEL_NAME!
echo.

REM ============ 2. CHON SO BUOC ============
echo ==== SO BUOC TRAIN (max_steps) ====
echo   Dataset: %DATASET%
set /p steps="Nhap so buoc (vd 200 / 500 / 5000, Enter = 500): "
if "%steps%"=="" set "steps=500"
set check=0
set /a check=steps 2>nul
if !check! LEQ 0 (
    echo [LOI] Phai nhap so nguyen
    pause
    exit /b 1
)
echo   - So buoc: %steps%
echo.

REM ============ 3. XAC NHAN BAT DAU ============
echo ==== XAC NHAN ====
echo   Model : !MODEL_NAME!
echo   Steps : %steps%
echo   Output: %ROOT%\output_lora
set /p go="Bat dau finetune? (Y/N): "
if /i not "%go%"=="Y" (
    echo Huy bo.
    pause
    exit /b 0
)
echo.

REM ============ 4. CHAY FINETUNE ============
echo ==== BAT DAU FINETUNE ... ====
"%PY%" "%SCRIPTS%\finetune_qwen3.py" --model "%MODEL_PATH%" --steps %steps% --dataset "%DATASET%"
if errorlevel 1 (
    echo [LOI] Finetune that bai (errorlevel %errorlevel%)
    pause
    exit /b 1
)
echo.
echo ==== FINETUNE HOAN TAT ====
echo.

REM ============ 5. XUAT GGUF + OLLAMA ============
echo ==== XUAT MODEL RA OLLAMA DE TEST ? ====
set /p exp="Xuat GGUF Q4_K_M va tao model ollama? (Y/N): "
if /i not "%exp%"=="Y" (
    echo Xong. Model merged luu tai: %ROOT%\output_lora_merged
    pause
    exit /b 0
)
echo.
echo ==== XUAT GGUF ... ====
"%PY%" "%SCRIPTS%\export_gguf.py" --model "%ROOT%\output_lora_merged" --out "%ROOT%\gguf_out_gguf"
if errorlevel 1 (
    echo [LOI] Export GGUF that bai
    pause
    exit /b 1
)
echo.

REM ============ 6. TAO MODELFILE + OLLAMA CREATE ============
set "GGUF=%ROOT%\gguf_out_gguf\output_lora_merged.Q4_K_M.gguf"
if not exist "%GGUF%" (
    echo [LOI] Khong tim thay GGUF: %GGUF%
    pause
    exit /b 1
)

set "MODELFILE=%ROOT%\Modelfile"
echo   Dang sinh Modelfile chuan (tu detect chat template)...
"%PY%" "%SCRIPTS%\make_modelfile.py" --gguf "%GGUF%" --model "%MODEL_PATH%" --name "!MODEL_NAME!" --out "%MODELFILE%"
if errorlevel 1 (
    echo [LOI] Khong sinh duoc Modelfile
    pause
    exit /b 1
)
echo   Da cap nhat Modelfile: %MODELFILE%

set "OLLAMA_NAME=!MODEL_NAME!-finetuned"
echo.
echo ==== TAO MODEL OLLAMA: %OLLAMA_NAME% ====
ollama create "%OLLAMA_NAME%" -f "%MODELFILE%"
if errorlevel 1 (
    echo [LOI] ollama create that bai
    pause
    exit /b 1
)
echo.
echo ==== XONG! Chay thu: ollama run %OLLAMA_NAME% ====
pause
exit /b 0
