@echo off
cd /d "%~dp0"
echo ====================================================
echo 🚀 KHOI DONG RAG ENTERPRISE & TAO PUBLIC WEB HTTPS 🚀
echo ====================================================

REM 1. Start Backend FastAPI
start "FastAPI Backend" cmd /c "cd /d "%~dp0" && ..\.venv\Scripts\uvicorn app.api.main:app --port 8000"
echo [Backend] Khoi dong thanh cong tren cong 8000.

REM Wait 2s
timeout /t 2 /nobreak >nul

REM 2. Start Frontend Vite
start "React Frontend" cmd /c "cd /d "%~dp0\frontend" && npm run dev"
echo [Frontend] Khoi dong thanh cong tren cong 5173.

REM Wait 3s
timeout /t 3 /nobreak >nul

REM 3. Launch Cloudflare Public Tunnel
echo.
echo [Cloudflare Tunnel] Dang tao link HTTPS cong khai toan cau...
..\.venv\Scripts\python scripts/launch_public_tunnel.py

pause
