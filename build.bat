@echo off
cd /d %~dp0

REM Clean previous build
REM Make sure no running instance keeps dist\renamer.exe locked
taskkill /f /im renamer.exe 2>nul
rmdir /s /q dist 2>nul
rmdir /s /q build 2>nul
del /q main.spec 2>nul

pyinstaller ^
  --onefile ^
  --noconsole ^
  --name Renamer ^
  --icon "assets/logo.ico" ^
  --add-data "poppler;poppler" ^
  --add-data "tesseract;tesseract" ^
  main.py

echo.
echo Build complete. EXE is in "dist\renamer.exe"
pause
