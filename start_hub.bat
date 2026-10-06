@echo off
chcp 65001 > nul
title Imputation Hub Launcher

echo ===================================================
echo           Running Imputation Hub          
echo ===================================================


cd /d "%~dp0"


docker info > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Docker Desktop is not running. Running Docker...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo Awaiting Docker engine...
    :wait_docker
    timeout /t 3 /nobreak > nul
    docker info > nul 2>&1
    if %ERRORLEVEL% NEQ 0 goto wait_docker
    echo [+] Docker is ready.
)

echo [*] Starting containers through docker compose...
docker compose up -d

timeout /t 4 /nobreak > nul

echo [*] Opening browser interface...
start http://localhost:8000

echo ===================================================
echo       Imputation Hub is running!             
echo ===================================================
timeout /t 3 /nobreak > nul
exit