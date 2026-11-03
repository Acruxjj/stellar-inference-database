import numpy as np
import pandas as pd

from src.data_diagnostic import (
    inspect_structure,
    preview_data,
    check_missing_values,
    check_duplicate_rows,
    summarize_numeric_columns,
    check_non_finite_values,
    general_diagnostics,
)


def test_check_missing_values():
    df = pd.DataFrame({
        "Teff_K": [3000.0, np.nan],
        "FeH_dex": [-0.2, 0.1],
    })

    result = check_missing_values(df)

    assert result["Teff_K"] == 1
    assert result["FeH_dex"] == 0


def test_check_duplicate_rows():
    df = pd.DataFrame({
        "Teff_K": [3000.0, 3000.0, 3500.0],
        "FeH_dex": [-0.2, -0.2, 0.1],
    })

    assert check_duplicate_rows(df) == 1


def test_check_non_finite_values():
    df = pd.DataFrame({
        "Teff_K": [3000.0, np.inf, -np.inf],
        "FeH_dex": [-0.2, 0.1, 0.3],
    })

    result = check_non_finite_values(df)

    assert result["Teff_K"] == 2
    assert result["FeH_dex"] == 0


def test_inspect_structure():
    df = pd.DataFrame({
        "Teff_K": [3000.0, 3500.0],
        "FeH_dex": [-0.2, 0.1],
    })

    result = inspect_structure(df)

    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["shape"] == (2, 2)
    assert result["memory_mb"] > 0
    assert result["column_types"]["Teff_K"] == "float64"


def test_preview_data():
    df = pd.DataFrame({
        "Teff_K": [3000.0, 3200.0, 3400.0, 3600.0],
    })

    result = preview_data(df, n=2)

    assert len(result) == 2
    assert result["Teff_K"].tolist() == [3000.0, 3200.0]


def test_summarize_numeric_columns():
    df = pd.DataFrame({
        "Teff_K": [3000.0, 4000.0],
        "spectral_type": ["M4", "M0"],
    })

    result = summarize_numeric_columns(df)

    assert "Teff_K" in result.index
    assert "spectral_type" not in result.index
    assert result.loc["Teff_K", "mean"] == 3500.0


def test_general_diagnostics():
    df = pd.DataFrame({
        "Teff_K": [3000.0, 3000.0, np.nan],
        "FeH_dex": [-0.2, -0.2, 0.1],
    })

    report = general_diagnostics(df)

    assert "structure" in report
    assert "preview" in report
    assert "missing_values" in report
    assert "duplicate_rows" in report
    assert "numeric_summary" in report
    assert "non_finite_values" in report

    assert report["structure"]["shape"] == (3, 2)
    assert report["missing_values"]["Teff_K"] == 1
    assert report["duplicate_rows"] == 1