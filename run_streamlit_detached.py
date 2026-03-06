import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        print(f"[ERROR] Python not found: {py}")
        return 1

    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        str(py),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        "8502",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--logger.level",
        "warning",
    ]

    # Fully detach from current terminal/session on Windows.
    creationflags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )

    out_log = open(root / "streamlit_detached.out.log", "ab")
    err_log = open(root / "streamlit_detached.err.log", "ab")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=out_log,
        stderr=err_log,
        creationflags=creationflags,
        close_fds=True,
    )
    print(f"STARTED_PID={proc.pid}")
    print("URL=http://localhost:8502")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
