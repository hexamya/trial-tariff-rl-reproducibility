"""Tests for the LCL smart-meter loader on a small synthetic zip in the source format."""
import zipfile

import duckdb
import numpy as np
import pandas as pd

from ttrl.data import lcl_meters as lm

HEADER = "LCLid,stdorToU,DateTime,KWH/hh (per half hour) \n"


def _write_zip(path, pieces: dict[str, list[str]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, lines in pieces.items():
            z.writestr(f"Small LCL Data/{name}", HEADER + "".join(line + "\n" for line in lines))


def test_convert_and_build_panel(tmp_path):
    p0 = ["MAC000002,Std,2013-01-01 00:00:00.0000000, 0.25 ",
          "MAC000002,Std,2013-01-01 00:30:00.0000000,Null",
          "MAC000002,Std,2013-01-01 15:21:58.0000000,Null",
          "MAC000003,ToU,2013-01-01 00:00:00.0000000, 1.5 "]
    # MAC000003 continues in the next piece, with one exact duplicate row.
    p1 = ["MAC000003,ToU,2013-01-01 00:00:00.0000000, 1.5 ",
          "MAC000003,ToU,2013-01-01 00:30:00.0000000, 0 "]
    zpath = tmp_path / "lcl.zip"
    _write_zip(zpath, {"LCL-June2015v2_1.csv": p1, "LCL-June2015v2_0.csv": p0})

    stats = lm.convert_zip(zpath, tmp_path / "pieces", n_pieces=2, verbose=False)
    assert stats["piece"].tolist() == [0, 1]
    assert stats[["n_rows", "n_null", "n_bad_ts"]].sum().tolist() == [6, 2, 1]
    assert stats[["n_bad_kwh", "n_bad_tariff", "n_bad_id"]].to_numpy().sum() == 0

    counts = lm.build_panel(tmp_path / "pieces", tmp_path / "panel.parquet", tmp_path / "hh.parquet")
    assert counts == {"n_raw": 6, "n_off_grid": 1, "n_off_grid_null": 1, "n_exact_dup": 1, "n_panel": 4,
                      "n_conflicting_dup": 0}

    panel = pd.read_parquet(tmp_path / "panel.parquet")
    assert panel["hh_id"].tolist() == [2, 2, 3, 3]
    assert panel["tou"].tolist() == [False, False, True, True]
    assert panel["ts_label"].iloc[1] == pd.Timestamp("2013-01-01 00:30")
    np.testing.assert_allclose(panel["kwh"].to_numpy(), [0.25, np.nan, 1.5, 0.0])
    assert panel["kwh"].dtype == np.float32

    hh = pd.read_parquet(tmp_path / "hh.parquet")
    assert hh["lclid"].tolist() == ["MAC000002", "MAC000003"]
    assert hh["n_rows"].tolist() == [2, 2] and hh["n_null"].tolist() == [1, 0]


def test_convert_piece_flags_format_problems(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(HEADER + "MAC000002,Xyz,2013-01-01 00:10:00.0000000, abc \n")
    stats = lm.convert_piece(duckdb.connect(), csv, tmp_path / "bad.parquet")
    assert stats == {"n_rows": 1, "n_null": 0, "n_bad_kwh": 1, "n_bad_tariff": 1, "n_bad_id": 0, "n_bad_ts": 1}
