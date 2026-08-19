"""
Cross-Platform Launcher: Starts Backend (FastAPI) and Frontend (Vite) concurrently.
Works on Windows, Linux, and macOS.
"""
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
os.chdir(ROOT_DIR)


def check_env_file():
    env_file = ROOT_DIR / ".env"
    example_file = ROOT_DIR / ".env.example"
    if not env_file.exists() and example_file.exists():
        print("⚠️ Chưa có file .env, đang tự động tạo từ .env.example...")
        shutil.copy(example_file, env_file)
        print("✅ Đã tạo file .env thành công.")


def get_python_executable():
    venv_win = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    venv_unix = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_win.exists():
        return str(venv_win)
    elif venv_unix.exists():
        return str(venv_unix)
    return sys.executable


def main():
    print("=" * 65)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ENTERPRISE CONTRACT INTELLIGENCE PLATFORM")
    print("=" * 65)

    check_env_file()
    py_exec = get_python_executable()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    # 1. Khởi động Backend
    print("\n▶️ [1/2] Đang khởi động Backend FastAPI (Port 8000)...")
    backend_proc = subprocess.Popen(
        [py_exec, "scripts/run_server.py"],
        cwd=str(ROOT_DIR),
        env=env,
    )

    time.sleep(2)

    # 2. Khởi động Frontend
    frontend_dir = ROOT_DIR / "frontend"
    print("\n▶️ [2/2] Đang khởi động Frontend Vite (Port 5173)...")
    
    frontend_proc = subprocess.Popen(
        "npm run dev",
        cwd=str(frontend_dir),
        shell=True,
    )

    print("\n" + "=" * 65)
    print("🎉 HỆ THỐNG ĐÃ KHỞI CHẠY THÀNH CÔNG!")
    print("👉 Backend API Docs : http://localhost:8000/docs")
    print("👉 Frontend Web App : http://localhost:5173")
    print("💡 Nhấn Ctrl + C để dừng đồng thời cả Backend và Frontend.")
    print("=" * 65 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Đang tắt toàn bộ dịch vụ...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("✅ Đã dừng hệ thống an toàn.")


if __name__ == "__main__":
    main()
