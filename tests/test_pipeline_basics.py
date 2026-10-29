"""Fast, offline sanity tests for the pipeline's geometry and helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import list_regions, load_region, wac_tile, lola_tile
from src.pipeline.io_utils import fill_gaps


def test_all_regions_load_with_expected_windows():
    for rid in list_regions():
        r = load_region(rid)
        assert (r["lola_height"], r["lola_width"]) == (512, 512)
        assert (r["diviner_height"], r["diviner_width"]) == (256, 256)
        assert 0 <= r["lola_row_start"] and r["lola_row_end"] <= 23040
        assert 0 <= r["lola_col_start"] and r["lola_col_end"] <= 46080
        assert 0 <= r["diviner_row_start"] and r["diviner_row_end"] <= 17920
        assert 0 <= r["diviner_col_start"] and r["diviner_col_end"] <= 46080


def test_pilot_lola_window_matches_known_indices():
    r = load_region("lacus_mortis")
    assert (r["lola_row_start"], r["lola_col_start"]) == (11264, 6912)


def test_tile_selection():
    assert wac_tile(44, 27) == "E300N0450"
    assert wac_tile(-44.5, 347.5) == "E300S3150"
    assert lola_tile(-44.5, -42.5, 347.5)[0] == "ldem_256_90s_00s_180_360_float"


def test_splits_are_disjoint_and_cover_registry():
    import yaml
    s = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"))
    parts = [set(s["train"]), set(s["val"]), set(s["test"])]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set().union(*parts) == set(list_regions())


def test_fill_gaps_flags_filled_pixels():
    a = np.ones((5, 5), dtype=np.float32); a[2, 2] = np.nan
    filled, valid = fill_gaps(a)
    assert np.isfinite(filled).all() and not valid[2, 2] and valid.sum() == 24
