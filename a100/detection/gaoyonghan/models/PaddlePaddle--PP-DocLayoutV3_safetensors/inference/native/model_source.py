"""Select this model's bundled framework before importing model code.

Paths are relative to this model directory; the same loader supports NVIDIA and MX.
"""
from pathlib import Path
import os
import sys

MODEL_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = MODEL_ROOT / 'inference/native/_ppdeps'
if not (SOURCE_ROOT / 'transformers' / "__init__.py").is_file():
    raise FileNotFoundError(f"Missing model source: {SOURCE_ROOT}")
source = str(SOURCE_ROOT)
if source in sys.path:
    sys.path.remove(source)
sys.path.insert(0, source)
# Carry the same source location into child Python processes.
previous = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = source + (os.pathsep + previous if previous else "")
