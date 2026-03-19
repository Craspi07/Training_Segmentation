@echo off
:: ============================================================
:: run_gui.bat  —  Launch the Cellpose Segmentation GUI on Windows
::
:: USAGE:
::   1. Double-click this file,  OR
::   2. From cmd.exe:
::        set DOCKER_CONTAINER=<your_container_name_or_id>
::        run_gui.bat
::
:: The GUI will open on Windows (no X server needed).
:: Training and inference are delegated to the Docker container
:: that has Detectron2 / Cellpose installed.
::
:: If DOCKER_CONTAINER is not set you can also type the container
:: name directly inside the GUI (Training tab → Docker field).
:: ============================================================

:: Install minimal Windows requirements (only pyyaml is needed for the GUI itself)
pip show pyyaml >nul 2>&1 || pip install pyyaml

:: Launch the GUI
python "%~dp0src\gui.py"

pause
