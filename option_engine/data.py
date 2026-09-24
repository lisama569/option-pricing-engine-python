"""Load bundled local volatility grids into a SurfaceStore at bootstrap.

File naming convention: ``<surface_id>_v<version>.json``. Files sharing a
surface id are published in filename order, so their published version
numbers follow the file ordering deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import SurfaceGridPayload
from .surface import SurfaceStore

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "surfaces"


def load_bundled_surfaces(store: SurfaceStore | None = None,
                          data_dir: Path | str = DEFAULT_DATA_DIR) -> SurfaceStore:
    store = store or SurfaceStore()
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))
    # Group by surface id (filename prefix before "_v").
    for path in files:
        surface_id = path.stem.rsplit("_v", 1)[0] if "_v" in path.stem else path.stem
        with path.open("r", encoding="utf-8") as fh:
            payload = SurfaceGridPayload(**json.load(fh))
        store.publish(surface_id, payload)
    return store
