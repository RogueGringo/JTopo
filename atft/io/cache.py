"""HDF5 serialization for intermediate results."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from atft.core.types import PersistenceDiagram


def save_persistence_diagram(pd: PersistenceDiagram, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for degree, diagram in pd.diagrams.items():
            f.create_dataset(f"degree_{degree}", data=diagram, compression="gzip")
        for k, v in pd.metadata.items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)


def load_persistence_diagram(path: Path) -> PersistenceDiagram:
    with h5py.File(path, "r") as f:
        diagrams = {}
        for key in f.keys():
            if key.startswith("degree_"):
                degree = int(key.split("_")[1])
                diagrams[degree] = f[key][:].astype(np.float64)
        metadata = {k: v for k, v in f.attrs.items()}
    return PersistenceDiagram(diagrams=diagrams, metadata=dict(metadata))
