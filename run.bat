@echo off
REM ================================================================
REM  RUN THE EXTRACTOR
REM  Easiest: DRAG a folder of PDFs onto this file.
REM  Or just double-click and paste a folder path.
REM ================================================================
cd /d "%~dp0"
setlocal

if not exist ".env" (
  echo.
  echo No .env found. Run setup.bat first to install packages and save your Groq key.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo.
  echo Tip: you can also just DRAG a folder of PDFs onto run.bat
  echo.
  set /p "FOLDER=Paste the folder that has your PDFs, then press Enter: "
) else (
  set "FOLDER=%~1"
)

echo.
echo Reading PDFs in: %FOLDER%
echo.
python run.py "%FOLDER%" --export hvac.csv
if errorlevel 1 (
  echo.
  echo Something went wrong. Make sure setup.bat was run first
  echo and that your Groq API key in .env is valid.
  pause
  exit /b 1
)
echo.
echo ================================================================
echo  DONE. Your results are in this folder:
echo     hvac.csv        = all equipment (schedule / tag / model ...)
echo     hvac_team.csv   = the project team directory
echo ================================================================
pause
