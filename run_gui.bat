@echo off
:: ============================================================
:: run_gui.bat  —  Launch the Cellpose Segmentation GUI on Windows
::
:: REQUIREMENTS:
::   - Python 3.9+ installed on Windows  (for the GUI itself)
::   - WSL 2 with Ubuntu installed        (for ML workloads)
::       → Install: wsl --install          (PowerShell as Admin)
::
:: USAGE:
::   1. Double-click this file, OR
::   2. From cmd.exe:
::        set WSL_DISTRO_NAME_OVERRIDE=Ubuntu   (optional — uses default distro)
::        set WSL_PYTHON=python3                 (optional — default: python3)
::        run_gui.bat
::
:: The GUI window opens on Windows.
:: Training/inference are delegated to WSL (Linux), which has
:: GPU access via the NVIDIA WSL 2 driver.
:: No Docker required.
:: ============================================================

:: ---- Verify WSL is available -----------------------------------------------
wsl --status >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: WSL does not appear to be installed or enabled.
    echo.
    echo  Fix: Open PowerShell as Administrator and run:
    echo      wsl --install
    echo  Then restart your computer and re-run this script.
    echo.
    pause
    exit /b 1
)

:: ---- Install minimal Windows-side dependencies (GUI only) ------------------
pip show pyyaml >nul 2>&1 || pip install pyyaml

:: ---- Launch the GUI --------------------------------------------------------
python "%~dp0src\gui.py"

pause
