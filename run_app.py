import subprocess
import webbrowser
import time
import os
import sys
from threading import Thread

# Percorsi
BACKEND_APP = "backend.main:app"
FRONTEND_DIR = "frontend"

# Porte
BACKEND_PORT = 8000
FRONTEND_PORT = 5500


def start_backend():
    print("[SERVER] Avvio backend FastAPI...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        BACKEND_APP,
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
        "--reload"
    ]
    subprocess.run(cmd)


def start_frontend():
    print("[CLIENT] Avvio server statico frontend...")
    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(FRONTEND_PORT)
    ]
    subprocess.run(cmd, cwd=FRONTEND_DIR)



def open_browser():
    time.sleep(2)
    url = f"http://localhost:{FRONTEND_PORT}"
    print(f"[CLIENT] Apro il browser su {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    print("=== Local Chatbot Launcher ===")

    # Thread per backend e frontend
    t1 = Thread(target=start_backend)
    t2 = Thread(target=start_frontend)
    t1.start()
    t2.start()

    # Apri browser
    Thread(target=open_browser).start()

    # Rimani attivo
    t1.join()
    t2.join()
