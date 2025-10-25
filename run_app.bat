@echo off
setlocal

:: Resolve project directory without trailing slash
for %%i in ("%~dp0.") do set "PROJECT_DIR=%%~fi"
cd /d "%PROJECT_DIR%"

call :detect_python
if not defined PYTHON_EXE goto :no_python

for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if not defined PY_MINOR set "PY_MINOR=0"

if "%PY_MAJOR%" NEQ "3" goto :unsupported_version
if %PY_MINOR% LSS 9 goto :unsupported_low
if %PY_MINOR% GTR 12 goto :unsupported_high

set "VENV_DIR=%PROJECT_DIR%\.venv_win_py%PY_MAJOR%%PY_MINOR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment for Python %PYTHON_VERSION% ...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Failed to create a virtual environment in "%VENV_DIR%".
        echo Make sure you can run "%PYTHON_EXE% -m venv" manually, then re-run this script.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Unable to activate the virtual environment at "%VENV_DIR%".
    echo Confirm that you have permission to access the folder, then try again.
    exit /b 1
)

echo.
echo Installing/validating dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo Failed to upgrade pip inside the virtual environment.
    echo Try running "%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip" manually.
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    echo Review the error details above. You may need to upgrade build tools or install wheel files manually.
    exit /b 1
)

echo.
echo Launching Concept Map Builder with Streamlit...
python -m streamlit run app.py
exit /b %errorlevel%

:detect_python
set "PYTHON_EXE="
set "PYTHON_VERSION="

call :probe "py" "-3"
if defined PYTHON_EXE exit /b 0

call :probe "py" ""
if defined PYTHON_EXE exit /b 0

for %%C in (python python3) do (
    call :probe "%%C" ""
    if defined PYTHON_EXE exit /b 0
)
exit /b 1

:probe
setlocal
set "CMD=%~1"
set "ARG=%~2"
for /f "usebackq tokens=1,2 delims=;" %%A in (`"%CMD%" %ARG% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor};{sys.executable}')" 2^>NUL`) do (
    endlocal & set "PYTHON_VERSION=%%A" & set "PYTHON_EXE=%%B" & exit /b 0
)
endlocal & exit /b 1

:no_python
echo.
echo ERROR: No suitable Python installation was found.
echo This project requires Python 3.9 - 3.12.
echo Install Python from https://www.python.org/downloads/windows/ or via the Microsoft Store,
echo ensure "Add Python to PATH" is selected, then re-run run_app.bat.
exit /b 1

:unsupported_version
echo.
echo ERROR: Detected Python %PYTHON_VERSION%, but this project needs Python 3.9 - 3.12.
echo Download a compatible Python release from https://www.python.org/downloads/windows/
echo and re-run this script after installation.
exit /b 1

:unsupported_low
echo.
echo ERROR: Detected Python %PYTHON_VERSION%, which is older than the minimum supported version (3.9).
echo Please install Python 3.9 through 3.12 from https://www.python.org/downloads/windows/
echo and make sure it is available on your PATH before running run_app.bat again.
exit /b 1

:unsupported_high
echo.
echo ERROR: Detected Python %PYTHON_VERSION%, which is newer than the tested range (max 3.12).
echo Upstream dependencies such as matplotlib and lxml do not yet publish wheels for 3.13.
echo Install Python 3.9-3.12 from https://www.python.org/downloads/windows/ and re-run run_app.bat.
exit /b 1
