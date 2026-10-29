"""Offline sanity checks for the tile-level diurnal layer."""
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "data" / "benchmark"


def test_diurnal_files_are_complete():
    files = sorted(BENCH.glob("*/diurnal_tbol.npz"))
    if not files:
        pytest.skip("diurnal layer not built")
    for f in files:
        d = np.load(f, allow_pickle=False)
        assert d["tbol"].shape == (16, 96)
        assert d["local_time"].shape == (96,)
        v = d["tbol"][np.isfinite(d["tbol"])]
        assert 30 < v.min() and v.max() < 450
