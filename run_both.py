"""Runs the SMC bot on BTCUSD and XAUUSD in parallel with separate configs."""

import subprocess
import sys

PYTHON = sys.executable

procs = []
for symbol in ["BTCUSD", "XAUUSD"]:
    p = subprocess.Popen(
        [PYTHON, "main.py", "--symbol", symbol],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"Started {symbol} (PID {p.pid})")
    procs.append(p)

try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\nStopping both bots...")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    print("Both bots stopped.")
