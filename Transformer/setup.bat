@echo off
REM Windows setup script for Protein-to-Drug generation pipeline

echo ==========================================
echo Protein-to-Drug Generation Setup
echo ==========================================

REM Check Python
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch
echo.
echo Installing PyTorch...
REM For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
REM Or for CPU only, uncomment:
REM pip install torch torchvision torchaudio

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Optional: Install protein models
echo.
echo Installing optional protein language models...
pip install fair-esm

REM Create directories
echo.
echo Creating output directories...
if not exist "results" mkdir results
if not exist "checkpoints" mkdir checkpoints

REM Check data
echo.
echo Checking for data files...
if exist "..\CPI\CPI\smiles.smi" (
    echo Data directory found: ..\CPI\CPI
    for /f %%A in ('find /c /v "" ^< ..\CPI\CPI\smiles.smi') do set smiles_count=%%A
    for /f %%A in ('find /c /v "" ^< ..\CPI\CPI\uniprot_ID.smi') do set protein_count=%%A
    echo - !smiles_count! SMILES compounds
    echo - !protein_count! protein IDs
) else (
    echo Warning: CPI data directory not found at ..\CPI\CPI
    echo Please ensure your data is in the correct location
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To run the pipeline:
echo   python main.py --stage all
echo.
echo Or by stages:
echo   python main.py --stage preprocess
echo   python main.py --stage train --epochs 50
echo   python main.py --stage generate
echo.
pause
