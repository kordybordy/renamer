@echo off
cd /d %~dp0

REM Clean previous build
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del /q main.spec 2>nul

pyinstaller ^
  --onefile ^
  --noconsole ^
  --name renamer ^
  --add-data "poppler;poppler" ^
  --add-data "tesseract;tesseract" ^
  main.py

echo.
echo Build complete. EXE is in "dist\renamer.exe"
pause
