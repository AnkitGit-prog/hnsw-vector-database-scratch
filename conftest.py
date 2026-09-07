"""conftest.py — Shared pytest fixtures."""
import sys
from pathlib import Path

# Make sure backend is importable from all test files
sys.path.insert(0, str(Path(__file__).parent / "backend"))
