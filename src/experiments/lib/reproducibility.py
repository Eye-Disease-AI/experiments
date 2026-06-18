import subprocess
import lightning as L

def global_seed_rng(seed):
    L.seed_everything(seed)

def get_git_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    return f"{sha}-dirty" if dirty else sha
