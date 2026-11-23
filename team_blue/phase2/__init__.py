"""Phase II: opening the TiRex-2 black box on the TIME benchmark."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "output" / "phase2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
