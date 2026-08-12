import subprocess
import sys
import os

os.chdir(r"D:\Projects\Retinasense Ai\backend")

# Use pythonw.exe for truly detached background process on Windows
pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
if not os.path.exists(pythonw):
    pythonw = sys.executable

log = open("train_v2.log", "a", encoding="utf-8", buffering=1)
errlog = open("train_v2.err.log", "a", encoding="utf-8", buffering=1)
cmd = [pythonw, "-u", "train_v2.py"]
if "--pause-per-epoch" in sys.argv:
    cmd.append("--pause-per-epoch")
p = subprocess.Popen(
    cmd,
    stdout=log,
    stderr=errlog,
    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
    start_new_session=True,
)
print(f"Training launched, PID={p.pid}", flush=True)
