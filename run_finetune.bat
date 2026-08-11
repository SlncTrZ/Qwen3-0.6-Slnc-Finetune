@echo off
chcp 65001 >nul
cd /d D:\ScriptPython\Training
echo ============================================
echo  Fine-tune Qwen3-0.6B - 550 steps (EN+VI dataset)
echo  Output: output_lora_merged
echo  Estimated time: ~15 minutes
echo ============================================
echo.
"D:\ScriptPython\Training\.unsloth-env\Scripts\python.exe" "D:\ScriptPython\Training\finetune_qwen3.py"
echo.
if %errorlevel%==0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code %errorlevel%
)
pause
