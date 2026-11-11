@echo off
title He Thong Nhan Dien Do Bao Ho Lao Dong (PPE AI)
chcp 65001 > nul
cd /d "%~dp0"

echo ==============================================================
echo   HE THONG GIAM SAT DO BAO HO LAO DONG (PPE DETECTION AI)
echo   Camera Imou + YOLOv8 Deep Learning
echo ==============================================================
echo.
echo Dang khoi dong ung dung, vui long cho giay lat...

python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Khong the chay ung dung.
    pause
)
