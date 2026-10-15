#!/usr/bin/env python3
"""Run several inference.py instances side by side over one image tree.

Used by run_department.sh when TSA_INSTANCES > 1. On A100/H100 a single
instance leaves the GPU idle while the host prepares each decoding step, so a
few instances sharing the GPU(s) process more pages per GPU-hour.

Each instance gets:
  * a shard: a tree of symlinks to every k-th image (round-robin over the
    sorted list, so the remaining work stays balanced when a run resumes),
    mirroring the source layout so commune/period folder names are unchanged;
  * its own config from make_config.py (only the input/temp dirs differ);
  * its own working directory with symlinks to models/ and
    layout_classification/ and a private logs/inference.log;
  * a GPU from CUDA_VISIBLE_DEVICES (instance i -> device i mod n).

All instances write to the same output directory. The pipeline's per-page
resume skips pages that any instance has already exported, so a killed run can
simply be started again.

    python scripts/hpc/run_parallel_inference.py --department Nord \
        --images /work/scratch/$USER/tsa/Nord/images \
        --output /work/scratch/$USER/tsa/Nord/output \
        --work   /work/scratch/$USER/tsa/Nord/parallel --instances 3
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def list_images(root: Path, page_list: Path = None):
    """Relative paths of all input images, sorted (same filter as inference.py)."""
    if page_list:
        rels = [Path(l.strip()) for l in page_list.read_text().splitlines() if l.strip()]
        return sorted(rels)
    rels = []
    for dirpath, _, files in os.walk(root, followlinks=True):
        for name in files:
            if name.startswith('._'):
                continue
            if name.lower().endswith(IMAGE_EXTENSIONS):
                rels.append(Path(dirpath, name).relative_to(root))
    return sorted(rels)


def build_shard(images_root: Path, rels, shard_dir: Path):
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    for rel in rels:
        link = shard_dir / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.realpath(images_root / rel), link)


def build_rundir(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    for name in ("models", "layout_classification"):
        link = run_dir / name
        if not link.is_symlink():
            link.symlink_to(REPO_ROOT / name)


def count_csvs(output: Path) -> int:
    pages = output / "pages"
    return sum(1 for _ in pages.rglob("*.csv")) if pages.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--department", required=True, help="slug, used in file names")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path,
                    help="directory for shards, configs and per-instance run dirs")
    ap.add_argument("--instances", required=True, type=int)
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP threads per instance (default: cores / instances)")
    ap.add_argument("--page-list", type=Path, default=None,
                    help="only these images (paths relative to --images)")
    ap.add_argument("--progress-every", type=int, default=900,
                    help="seconds between progress lines (0 = off)")
    ap.add_argument("--base-config", type=Path, default=None,
                    help="passed to make_config.py (default: its production base config)")
    bert = ap.add_mutually_exclusive_group()
    bert.add_argument("--with-bert", dest="bert", action="store_const", const="--with-bert")
    bert.add_argument("--no-bert", dest="bert", action="store_const", const="--no-bert")
    args = ap.parse_args()

    images = args.images.resolve()
    output = args.output.resolve()
    work = args.work.resolve()
    k = max(1, args.instances)
    rels = list_images(images, args.page_list)
    if not rels:
        print(f"ERROR: no images found under {images}", file=sys.stderr)
        return 1

    devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d != ""]
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or k
    threads = args.threads or max(1, cores // k)
    print(f"==> {len(rels)} images, {k} instances, GPUs {devices or ['default']}, "
          f"{threads} thread(s) each", flush=True)

    procs = []
    for i in range(k):
        # Name the shard root like the original input dir: the pipeline takes the
        # "commune" from the image's grandparent dir, which for zips without a
        # commune level (Paris: Period/page.jpg) is the input dir itself. Shard
        # outputs then land exactly where a single-instance run puts them.
        shard_parent = work / "shards" / f"s{i}"
        if shard_parent.exists():
            shutil.rmtree(shard_parent)
        shard = shard_parent / images.name
        build_shard(images, rels[i::k], shard)
        run_dir = work / f"run{i}"
        build_rundir(run_dir)
        cfg = work / "configs" / f"{args.department}.s{i}.yaml"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/hpc/make_config.py"),
               "--department", f"{args.department}.s{i}",
               "--input-dir", str(shard), "--output-dir", str(output),
               "--temp-dir", str(output / f"tmp_s{i}"), "--out", str(cfg)]
        if args.bert:
            cmd.append(args.bert)
        if args.base_config:
            cmd += ["--base-config", str(args.base_config.resolve())]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[i % len(devices)]
        log = open(run_dir / "stdout.log", "ab")
        p = subprocess.Popen([sys.executable, str(REPO_ROOT / "inference.py"),
                              "--config", str(cfg)],
                             cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, p, len(rels[i::k])))
        print(f"    instance {i}: pid {p.pid}, {len(rels[i::k])} images, "
              f"GPU {env.get('CUDA_VISIBLE_DEVICES', 'default')}, log {run_dir}/logs/inference.log",
              flush=True)

    stopping = {"sig": None}

    def forward(signum, _frame):
        stopping["sig"] = signum
        for _, p, _ in procs:
            if p.poll() is None:
                p.terminate()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, forward)

    t0 = time.time()
    last = t0
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(5)
        if args.progress_every and time.time() - last >= args.progress_every:
            last = time.time()
            print(f"    progress {time.strftime('%F %T')}: {count_csvs(output)} CSVs in output "
                  f"after {int(last - t0)} s", flush=True)

    rcs = [p.returncode for _, p, _ in procs]
    print(f"==> instances finished after {int(time.time() - t0)} s, exit codes {rcs}", flush=True)
    if stopping["sig"] is not None:
        return 128 + int(stopping["sig"])
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
