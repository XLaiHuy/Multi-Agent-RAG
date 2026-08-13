"""
Auto Public Web Tunnel Launcher:
- Tự động tải cloudflared.exe (nếu chưa có) từ Cloudflare chính thức
- Mở đường hầm HTTPS công khai ra internet kết nối vào React Frontend (cổng 5173)
- In ra link HTTPS công khai để truy cập từ điện thoại hoặc chia sẻ demo ngay lập tức!
"""
import os
import sys
import subprocess
import urllib.request
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
TOOLS_DIR = Path("data/tools")
TOOLS_DIR.mkdir(parents=True, exist_ok=True)
CLOUDFLARED_EXE = TOOLS_DIR / "cloudflared.exe"


def ensure_cloudflared() -> Path:
    if not CLOUDFLARED_EXE.exists():
        print("[Cloudflare Tunnel] Downloading lightweight cloudflared.exe...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_EXE)
        print("[Cloudflare Tunnel] Download completed!", flush=True)
    return CLOUDFLARED_EXE


def start_tunnel(port: int = 5173):
    exe_path = ensure_cloudflared()
    print(f"\n========================================================", flush=True)
    print(f"🌐 KHỞI ĐỘNG PUBLIC WEB TUNNEL (Port {port})", flush=True)
    print(f"========================================================", flush=True)
    print("Đang khởi tạo đường hầm HTTPS bảo mật qua Cloudflare...", flush=True)

    cmd = [str(exe_path), "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    url_found = None
    for line in iter(process.stdout.readline, ""):
        line_clean = line.strip()
        # Tìm link trycloudflare.com
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line_clean)
        if match and not url_found:
            url_found = match.group(0)
            print("\n" + "🎉 " * 15, flush=True)
            print(f"🚀 WEB ĐÃ ONLINE TOÀN CẦU! TRUY CẬP TẠI LINK DƯỚI ĐÂY:", flush=True)
            print(f"👉 \033[1;32m{url_found}\033[0m", flush=True)
            print("🎉 " * 15 + "\n", flush=True)
            print(f"Tài khoản đăng nhập thử nghiệm:", flush=True)
            print(f"  • Admin:   admin / admin", flush=True)
            print(f"  • HR:      hr01 / hr", flush=True)
            print(f"  • Finance: ketoan01 / ketoan", flush=True)
            print(f"\nNhấn CTRL+C để đóng đường hầm Web.\n", flush=True)
        elif not url_found and line_clean:
            # Print status log
            pass

    process.wait()


if __name__ == "__main__":
    start_tunnel(5173)
