@echo off
REM ================================================================
REM  ONE-TIME SETUP  -  double-click this once.
REM  It installs the Python packages and saves your Groq API key.
REM  Requires Python 3.10+  (https://www.python.org/downloads/ ,
REM  tick "Add Python to PATH" during install).
REM ================================================================
cd /d "%~dp0"
setlocal EnableDelayedExpansion

echo.
echo === Installing Python packages ===
python -m pip install --user -r requirements.txt
if errorlevel 1 (
  echo.
  echo Python not found. Install it from https://www.python.org/downloads/
  echo and be sure to tick "Add Python to PATH" during install, then run setup again.
  pause
  exit /b 1
)

echo.
if exist ".env" (
  echo A .env file already exists - keeping your current settings.
  echo (To change the API key, edit .env in Notepad.)
  goto done
)

echo ================================================================
echo  GROQ API KEY
echo  Get a FREE key here:  https://console.groq.com/keys
echo    1. Sign in ^(Google/GitHub/email^)
echo    2. Click "Create API Key", give it any name
echo    3. Copy the key ^(it starts with  gsk_ ^)
echo ================================================================
echo.
set "GROQ_KEY="
set /p "GROQ_KEY=Paste your Groq API key here, then press Enter: "

if "!GROQ_KEY!"=="" (
  echo.
  echo No key entered. Re-run setup.bat when you have your key.
  pause
  exit /b 1
)

> ".env" echo MODEL_BACKEND=groq
>> ".env" echo GROQ_API_KEY=!GROQ_KEY!
>> ".env" echo GROQ_MODEL=openai/gpt-oss-120b
>> ".env" echo DB_PATH=hvac.db
>> ".env" echo RESTRICT_TO_MECHANICAL=true
echo.
echo Saved your key to .env

:done
echo.
echo ================================================================
echo  Setup complete!  Now use run.bat:
echo    - DRAG a folder of PDFs onto run.bat, or
echo    - double-click run.bat and paste a folder path.
echo ================================================================
pause
