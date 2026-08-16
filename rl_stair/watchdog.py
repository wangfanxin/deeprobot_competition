"""Lightweight training watchdog: detects stalls / GPU pegs / disk/mem pressure.

Read-only monitor. Appends WARN lines to rl_stair/logs/watchdog.log every 2 min.
No env/training changes.
"""
import os, sys, time, subprocess, glob

LOGDIR = os.environ.get("S10_WATCH_LOGDIR", "/home/wfx/DR_competition/0810new/deeprobot_competition/rl_stair/logs")
TRAIN_LOG = os.path.join(LOGDIR, "train.log")
WATCH = os.path.join(LOGDIR, "watchdog.log")
INTERVAL = 120          # seconds
MAX_TRAIN_AGE = 420      # 7 min without a new iter line = stall
GPU_PEG = 97.0           # % util above this repeatedly = peg
MIN_C_FREE_GB = 20.0
MIN_MEM_FREE_GB = 2.0


def log(msg):
    with open(WATCH, "a") as f:
        f.write("[%s] %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))


def warn(msg):
    log("WARN " + msg)
    print("WARN " + msg, flush=True)


def main():
    log("watchdog started (interval=%ds)" % INTERVAL)
    peg_count = 0
    while True:
        try:
            # 1) train.log freshness
            if os.path.exists(TRAIN_LOG):
                age = time.time() - os.path.getmtime(TRAIN_LOG)
                if age > MAX_TRAIN_AGE:
                    warn("train.log stale %.0fs (possible stall)" % age)
            else:
                warn("train.log missing")

            # 2) process alive
            r = subprocess.run(["pgrep", "-f", "rl_stair/train.py"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                warn("train.py process NOT found")

            # 3) GPU util
            g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip()
            if g:
                util = float(g.split(",")[0])
                if util >= GPU_PEG:
                    peg_count += 1
                    if peg_count >= 5:
                        warn("GPU pegged at %.0f%% for %d checks" % (util, peg_count))
                else:
                    peg_count = 0

            # 4) disk: /mnt/c free
            d = subprocess.run(["df", "-BG", "/mnt/c"], capture_output=True, text=True).stdout
            try:
                parts = d.splitlines()[1].split()
                free_gb = float(parts[3].rstrip("G"))
                if free_gb < MIN_C_FREE_GB:
                    warn("C drive free only %.1fGB" % free_gb)
            except Exception:
                pass

            # 5) memory free
            m = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
            try:
                free_gb = float(m.splitlines()[1].split()[3]) + float(m.splitlines()[1].split()[5])
                if free_gb < MIN_MEM_FREE_GB:
                    warn("free mem only %.1fGB" % free_gb)
            except Exception:
                pass
        except Exception as e:
            warn("watchdog error: %s" % e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
