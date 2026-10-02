@echo off
setlocal
pushd "%~dp0front"
call npm.cmd start
set "CURE_EXIT=%errorlevel%"
popd
exit /b %CURE_EXIT%
