"""Put the package source on sys.path so the tests run without `pip install -e .`.

The package lives under robotics/src (see pyproject.toml: packages.find where=src).
This lets `import ot2...` work straight from a checkout.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
