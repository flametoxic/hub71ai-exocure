@echo off
setlocal
pushd "%~dp0"
call node --version >nul 2>&1
if errorlevel 1 (
  echo Install Node.js 22.12 or newer, then run INSTALL.cmd again.
  goto failed
)
node -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a<22 || (a===22 && b<12) ? 1 : 0)"
if errorlevel 1 (
  echo Node.js 22.12 or newer is required.
  goto failed
)
py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if not errorlevel 1 (
  py -3.11 -m venv backend\.venv
) else (
  python -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
  if errorlevel 1 (
    echo Install Python 3.11 with the Python launcher, then run INSTALL.cmd again.
    goto failed
  )
  python -m venv backend\.venv
)
if errorlevel 1 goto failed
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 goto failed
if not exist backend\.env copy /y backend\.env.example backend\.env >nul
pushd front
call npm.cmd ci
if errorlevel 1 (
  popd
  goto failed
)
call npm.cmd run build
if errorlevel 1 (
  popd
  goto failed
)
popd
echo Installation complete. Launch CURE.cmd.
popd
pause
exit /b 0
:failed
echo Installation failed. See the message above and README.md.
popd
pause
exit /b 1
