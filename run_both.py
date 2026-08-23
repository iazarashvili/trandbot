"""Runs the SMC bot on all configured symbols in parallel with separate configs."""

import subprocess
import sys

PYTHON = sys.executable

SYMBOLS = ["BTCUSD", "XAUUSD", "GBPUSD", "EURUSD"]

procs = []
for symbol in SYMBOLS:
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
    print("\nStopping all bots...")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    print("All bots stopped.")
