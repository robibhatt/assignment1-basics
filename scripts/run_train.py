from cs336_basics.training.train_config import from_yaml
from cs336_basics.training.train import TrainConfig, train
import yaml
import torch
import cProfile, pstats
from pathlib import Path

# Assume we run from repo root; filter to files under these two dirs.
REPO_ROOTS = [
    (Path.cwd() / "cs336_basics").resolve(),
    (Path.cwd() / "scripts").resolve(),
]
PROJECT_ROOT = Path.cwd().resolve()

def is_mine(filename: str) -> bool:
    if not filename or filename.startswith("{") or filename.startswith("<"):
        return False
    try:
        p = Path(filename).resolve()
    except Exception:
        return False
    return any(str(p).startswith(str(root)) for root in REPO_ROOTS)

def relpath(filename: str) -> str:
    """Return path relative to project root, else just the name."""
    try:
        return str(Path(filename).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return Path(filename).name

def main():
    cfg = from_yaml(yamlfile="scripts/config.yaml")

    if getattr(cfg, "profile", False):
        with cProfile.Profile() as pr:
            train_dir = train(cfg=cfg)

        profile_path = Path(train_dir) / "profile.txt"
        rows = []
        stats = pstats.Stats(pr)

        for (filename, lineno, funcname), info in stats.stats.items():
            try:
                ct = info.cumulative
                tt = info.totaltime
                cc = info.callcount
            except AttributeError:
                cc, nc, tt, ct, callers = info
            if is_mine(filename):
                rel = relpath(filename)
                rows.append({
                    "cum_time": ct,
                    "self_time": tt,
                    "calls": cc,
                    "loc": f"{rel}:{lineno}",
                    "func": funcname,
                })

        rows.sort(key=lambda r: r["cum_time"], reverse=True)

        with open(profile_path, "w") as f:
            print(f"{'CUMTIME':>10}  {'SELFTIME':>10}  {'CALLS':>8}  LOCATION  FUNCTION", file=f)
            print("=" * 100, file=f)
            for r in rows:  # print all, not just top 100
                print(
                    f"{r['cum_time']:10.4f}s  {r['self_time']:10.4f}s  {r['calls']:8}  {r['loc']}  {r['func']}",
                    file=f,
                )
    else:
        train_dir = train(cfg=cfg)

if __name__ == "__main__":
    main()
