#!/bin/bash

# ==============================================================================
# Script khởi chạy đồng thời Backend (FastAPI) & Frontend (React + Vite)
# ==============================================================================

echo "========================================================"
echo "🚀 Đang khởi động Enterprise Contract Intelligence Platform..."
echo "========================================================"

# Kiểm tra file .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "⚠️ Không tìm thấy file .env, tự động tạo từ .env.example..."
        cp .env.example .env
    fi
fi

# Hàm dọn dẹp khi tắt script (Ctrl + C)
cleanup() {
    echo ""
    echo "🛑 Đang tắt toàn bộ dịch vụ (Backend & Frontend)..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Khởi động Backend FastAPI
echo "▶️ [1/2] Đang khởi động Backend FastAPI (Port 8000)..."
if [ -d ".venv" ]; then
    if [ -f ".venv/Scripts/python.exe" ]; then
        .venv/Scripts/python.exe scripts/run_server.py &
    else
        .venv/bin/python scripts/run_server.py &
    fi
else
    python scripts/run_server.py &
fi
BACKEND_PID=$!

# Đợi 2 giây cho backend khởi động
sleep 2

# 2. Khởi động Frontend Vite
echo "▶️ [2/2] Đang khởi động Frontend Vite (Port 5173 / 3000)..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Đang cài đặt thư viện frontend (npm install)..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================================"
echo "✅ Hệ thống đã sẵn sàng!"
echo "👉 Backend API Docs : http://localhost:8000/docs"
echo "👉 Frontend Web App : http://localhost:5173 (hoặc port hiển thị của Vite)"
echo "💡 Nhấn Ctrl + C để dừng đồng thời cả 2 dịch vụ."
echo "========================================================"

# Giữ script chạy để lắng nghe Ctrl + C
wait
