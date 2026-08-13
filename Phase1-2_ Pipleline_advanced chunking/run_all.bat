@echo off
cd /d "%~dp0"
echo ====================================================
echo 🚀 KHOI DONG HE THONG RAG ENTERPRISE 🚀
echo ====================================================

REM Start Backend
start "FastAPI Backend" cmd /c "cd /d "%~dp0" && ..\.venv\Scripts\uvicorn app.api.main:app --reload --port 8000"
echo [Backend] Da khoi dong tai http://localhost:8000

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM Start Frontend
start "React Frontend" cmd /c "cd /d "%~dp0\frontend" && npm run dev"
echo [Frontend] Da khoi dong. Vui long kiem tra cua so moi hien ra.
echo.
echo He thong dang chay. An phim bat ky de thoat va dong toan bo cac cua so.
pause
