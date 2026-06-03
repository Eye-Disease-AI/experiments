import subprocess
import lightning as L


class RNG:
    def __init__(self):
        pass

    def is_deterministic(self):
        return self.seed is not None

    def set_seed(self, seed):
        self.seed = seed
        L.seed_everything(seed)

    def disable_determinism(self):
        self.seed = None
        L.seed_everything(None)


def get_git_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    return f"{sha}-dirty" if dirty else sha
