"""Sidecar system monitor for training benchmarks (Windows).

Samples GPU (nvidia-smi), CPU, RAM and disk I/O every `--interval` seconds
while the target PID is alive, then writes a CSV. No code in train_v2.py is
touched by this monitor.
"""
import argparse
import csv
import subprocess
import time

import psutil

GPU_FIELDS = ["utilization.gpu", "memory.used", "power.draw", "clocks.sm",
              "clocks.mem", "temperature.gpu"]


def sample_gpu():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip()
    parts = [p.strip() for p in out.split(",")]
    return dict(zip(["util", "mem", "power", "sm", "memclk", "temp"], parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    try:
        proc = psutil.Process(args.pid)
    except psutil.NoSuchProcess:
        print(f"monitor: target PID {args.pid} not running")
        return

    disk0 = psutil.disk_io_counters()
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gpu_util_%", "gpu_mem_mb", "power_w", "sm_mhz",
                    "mem_mhz", "temp_c", "cpu_total_%", "cpu_mainproc_%",
                    "disk_read_mbs", "disk_write_mbs", "ram_used_gb"])
        f.flush()
        while proc.is_running():
            time.sleep(args.interval)
            t = time.time() - t0
            try:
                cpu_proc = proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_proc = -1.0
            d = psutil.disk_io_counters()
            rd = (d.read_bytes - disk0.read_bytes) / 1e6 / args.interval
            wr = (d.write_bytes - disk0.write_bytes) / 1e6 / args.interval
            disk0 = d
            g = sample_gpu()
            w.writerow([
                round(t, 1), g["util"], g["mem"], g["power"], g["sm"],
                g["memclk"], g["temp"],
                round(psutil.cpu_percent(interval=None), 1), round(cpu_proc, 1),
                round(rd, 2), round(wr, 2),
                round(psutil.virtual_memory().used / 1e9, 2),
            ])
            f.flush()


if __name__ == "__main__":
    main()
