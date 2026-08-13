@echo off
cd /d "%~dp0"
echo ====================================================
echo 🚀 KHOI DONG SERVER FASTAPI (AGENTIC RAG API) 🚀
echo ====================================================
echo.
echo Server se chay tai: http://localhost:8000
echo Tai lieu API (Swagger UI): http://localhost:8000/docs
echo.
..\.venv\Scripts\uvicorn app.api.main:app --reload
pause
