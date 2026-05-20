from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


INNER_PROJECT_DIR = Path(__file__).resolve().parent / "pharma_intelligence_fastapi"
INNER_APP_PATH = INNER_PROJECT_DIR / "app.py"

if not INNER_APP_PATH.exists():
    raise RuntimeError(f"Could not find FastAPI app at {INNER_APP_PATH}")

sys.path.insert(0, str(INNER_PROJECT_DIR))

spec = importlib.util.spec_from_file_location("_pharma_intelligence_fastapi_app", INNER_APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load FastAPI app from {INNER_APP_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app
