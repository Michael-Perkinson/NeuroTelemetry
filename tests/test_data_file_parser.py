from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.data_file_parser import (
    detect_skip_rows,
    read_and_process_photometry_file,
    safe_get_df,
)


def test_read_and_process_photometry_file_keeps_numeric_signal_columns(
    local_tmpdir: Path,
) -> None:
    photometry_path = local_tmpdir / "photometry.csv"
    photometry_path.write_text(
        "\n".join(
            [
                "# t_min,dFoF_465,dFoF_405,Z_465,Ignored",
                "0.0,1.0,2.0,3.0,a",
                "0.5,bad,2.5,3.5,b",
                "1.0,4.0,5.0,6.0,c",
            ]
        )
    )

    df = read_and_process_photometry_file(photometry_path)

    assert df.to_dict("list") == {
        "TimeSinceReference": [0.0, 60.0],
        "dFoF_465": [1.0, 4.0],
        "dFoF_405": [2.0, 5.0],
        "Z_465": [3.0, 6.0],
    }


def test_read_and_process_photometry_file_requires_time_column(
    local_tmpdir: Path,
) -> None:
    photometry_path = local_tmpdir / "photometry.csv"
    photometry_path.write_text("time,dFoF_465\n0,1\n")

    with pytest.raises(ValueError, match="# t_min"):
        read_and_process_photometry_file(photometry_path)


def test_detect_skip_rows_returns_first_time_row() -> None:
    raw = pd.DataFrame(
        {
            0: [
                "# header",
                "# Col 1:,Pressure,,500",
                " Time,Pressure,Temp",
                "01/01/2025 01:00:00 PM,1.0,20.0",
            ]
        }
    )

    assert detect_skip_rows(raw) == 2


def test_safe_get_df_requires_pressure_and_normalizes_missing_optional_data() -> None:
    with pytest.raises(ValueError, match="Pressure data is required"):
        safe_get_df({}, "Pressure")

    assert safe_get_df({"Temp": None}, "Temp").empty
    assert safe_get_df({"Activity": pd.DataFrame()}, "Activity").empty

    pressure_df = pd.DataFrame({"Pressure": [1.0]})
    assert safe_get_df({"Pressure": pressure_df}, "Pressure").equals(pressure_df)
