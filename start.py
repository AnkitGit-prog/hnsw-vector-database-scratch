"""
start.py — One-click startup script.

Checks prerequisites, optionally generates data, then starts the API server.
The React frontend must be started separately with: cd frontend && npm run dev

Usage:
  python start.py                  # start API (use cached data)
  python start.py --generate       # regenerate dataset, then start API
  python start.py --no-server      # generate data only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def check_dependencies() -> bool:
    """Check that required packages are installed."""
    missing = []
    for pkg in ["fastapi", "uvicorn", "numpy", "sentence_transformers"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True


def data_exists() -> bool:
    """Check if dataset files exist."""
    np_path = ROOT / "data" / "generated" / "dataset_vectors.npz"
    meta_path = ROOT / "data" / "generated" / "dataset_metadata.json"
    return np_path.exists() and meta_path.exists()


def run_generate() -> None:
    print("\n[1/3] Generating dataset (50,000 vectors)...")
    subprocess.run([sys.executable, "scripts/generate_dataset.py"], cwd=ROOT, check=True)


def run_api() -> None:
    print("\n[✓] Starting FastAPI server at http://localhost:8000")
    print("    API docs: http://localhost:8000/docs")
    print("    Frontend: cd frontend && npm run dev\n")
    os.chdir(ROOT / "backend")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], cwd=ROOT / "backend")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="(Re)generate dataset")
    parser.add_argument("--no-server", action="store_true", help="Generate data only, don't start server")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vector Database From Scratch")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    if args.generate or not data_exists():
        run_generate()
    else:
        print("[✓] Dataset found. Use --generate to regenerate.")

    if not args.no_server:
        run_api()
