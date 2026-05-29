@echo off
REM Activate the MSVC x64 toolchain (cl.exe) needed by nvcc as the host
REM compiler on Windows, then run whatever command was passed in.
REM
REM Usage (from Git Bash):
REM   MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\with_msvc.bat python scripts\phase0_saxpy.py"
REM
REM Or from a cmd prompt:
REM   scripts\with_msvc.bat python scripts\phase0_saxpy.py
REM
REM NOTE: avoid parenthesized if-blocks here -- the "(x86)" in the Visual
REM Studio path confuses cmd's block parser. Use goto labels instead.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :novcvars
call "%VCVARS%" >nul
if errorlevel 1 goto :failenv
call %*
exit /b %errorlevel%

:novcvars
echo [with_msvc] ERROR: vcvars64.bat not found at "%VCVARS%"
echo Edit scripts\with_msvc.bat to point at your Visual Studio install.
exit /b 1

:failenv
echo [with_msvc] ERROR: failed to initialize MSVC environment.
exit /b 1
