"""Shared pytest fixtures/configuration.

Ensures the ``src`` layout is importable even when the package has not been
installed, so tests can run in a plain checkout.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
