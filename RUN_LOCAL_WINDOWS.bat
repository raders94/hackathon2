@echo off
chcp 65001 > nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Python Launcher is missing. Install Python 3.12 first.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 py -3.12 -m venv .venv
 if errorlevel 1 goto :failure
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failure
".venv\Scripts\python.exe" check_release.py
if errorlevel 1 goto :failure
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0
:failure
echo Setup or validation failed. See messages above.
pause
exit /b 1
