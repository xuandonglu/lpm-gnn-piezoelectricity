"""Output and reproducibility helpers shared by the examples."""

import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new output path.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_convergence(info):
    if not info.get("converged", False):
        raise RuntimeError(f"Equilibrium solver did not converge: {info}")
