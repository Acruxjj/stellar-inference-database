import numpy as np
import pandas as pd


def inspect_structure(df: pd.DataFrame) -> dict:
    # Return basic information about a DataFrame.
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "shape": df.shape,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "column_types": df.dtypes.astype(str).to_dict(),
    }


def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    # Return the first n rows.
    return df.head(n)


def check_missing_values(df: pd.DataFrame) -> dict:
    # Count missing values in each column.
    return df.isna().sum().to_dict()


def check_duplicate_rows(df: pd.DataFrame) -> int:
    # Count rows that are entirely duplicated.
    return int(df.duplicated().sum())


def summarize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Return descriptive statistics for numeric columns.
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T


def check_non_finite_values(df: pd.DataFrame) -> dict:
    # Count positive and negative infinity in numeric columns.
    # Missing values are reported separately.
    
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {}

    values = numeric_df.to_numpy(dtype=float, na_value=np.nan)

    counts = np.isinf(values).sum(axis=0)

    return {
        column: int(count)
        for column, count in zip(numeric_df.columns, counts)
    }


def general_diagnostics(df: pd.DataFrame) -> dict:
    # Run all general diagnostics and return their results.
    return {
        "structure": inspect_structure(df),
        "preview": preview_data(df),
        "missing_values": check_missing_values(df),
        "duplicate_rows": check_duplicate_rows(df),
        "numeric_summary": summarize_numeric_columns(df),
        "non_finite_values": check_non_finite_values(df),
    }