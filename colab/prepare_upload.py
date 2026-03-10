"""Package project files into a zip for Google Colab training.

Run from the project root:
    python colab/prepare_upload.py

Produces colab/fpl_rl_colab.zip containing:
    src/fpl_rl/          — full source package
    data/                — all collected data (raw, understat, fbref, id_maps)
    models/point_predictor/ — trained LightGBM models
    pyproject.toml       — for pip install -e ".[dev,prediction]"
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Resolve project root (parent of colab/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_ZIP = SCRIPT_DIR / "fpl_rl_colab.zip"

# Directories and files to include
INCLUDE = [
    ("src/fpl_rl", "src/fpl_rl"),
    ("data", "data"),
    ("models/point_predictor", "models/point_predictor"),
    ("pyproject.toml", "pyproject.toml"),
]

# Extensions to skip (large / unnecessary files)
SKIP_EXTENSIONS = {".pyc", ".pyo", ".egg-info"}
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "node_modules"}


def should_skip(path: Path) -> bool:
    """Return True if this path should be excluded from the zip."""
    if path.suffix in SKIP_EXTENSIONS:
        return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def add_directory(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> int:
    """Recursively add a directory to the zip. Returns file count."""
    count = 0
    for root, dirs, files in os.walk(src):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            if should_skip(fpath):
                continue
            arcname = arc_prefix / fpath.relative_to(src)
            zf.write(fpath, str(arcname))
            count += 1
    return count


def main() -> None:
    missing = []
    for local_rel, _ in INCLUDE:
        full = PROJECT_ROOT / local_rel
        if not full.exists():
            missing.append(str(full))

    if missing:
        print("ERROR: The following paths are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nMake sure you run this from the project root and that")
        print("data collection + model training have been completed.")
        sys.exit(1)

    total = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for local_rel, arc_rel in INCLUDE:
            src = PROJECT_ROOT / local_rel
            arc = Path(arc_rel)
            if src.is_file():
                zf.write(src, str(arc))
                total += 1
                print(f"  + {arc}")
            elif src.is_dir():
                count = add_directory(zf, src, arc)
                total += count
                print(f"  + {arc}/ ({count} files)")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nCreated {OUTPUT_ZIP.name}: {size_mb:.1f} MB ({total} files)")
    print(f"Location: {OUTPUT_ZIP}")
    print("\nNext steps:")
    print("  1. Upload fpl_rl_colab.zip to your Google Drive")
    print("  2. Open colab/train_rl_colab.ipynb in Google Colab")
    print("  3. Run all cells")


if __name__ == "__main__":
    main()
