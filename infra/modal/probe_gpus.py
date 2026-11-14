"""Probe which Modal GPU types schedule right now and how long a container takes to start.

    python infra/modal/probe_gpus.py [H100 H200 B200 A100-80GB H100:2]

Runs nvidia-smi once per GPU type, in parallel, and prints type, seconds until the result, and the GPU line.
"""
import concurrent.futures as cf
import sys
import time

import modal

app = modal.App("fh-gpu-probe")
image = modal.Image.debian_slim()
GPUS = sys.argv[1:] or ["H100", "H200", "B200", "A100-80GB", "H100:2"]


def _smi() -> str:
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().replace("\n", " | ")


fns = {g: app.function(gpu=g, image=image, timeout=600, serialized=True,
                       name="probe_" + g.replace(":", "x").replace("-", "_"))(_smi) for g in GPUS}


def timed(g):
    t = time.time()
    try:
        out = fns[g].remote()
    except Exception as e:  # noqa: BLE001
        out = "ERR " + repr(e)[:200]
    return g, round(time.time() - t, 1), out


if __name__ == "__main__":
    with app.run():
        with cf.ThreadPoolExecutor(len(GPUS)) as ex:
            for g, secs, out in ex.map(timed, GPUS):
                print(f"{g}\t{secs}s\t{out}", flush=True)
