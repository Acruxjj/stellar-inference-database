from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STARS_PATH = PROJECT_ROOT / "data" / "all_stars.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "stellar_window_comparison.csv"

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "stellar_inference.db"

# Load and validate the source files.
def load_source_data():
    stars = pd.read_parquet(STARS_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    required_star_columns = [
        "Teff_K",
        "logg_cgs",
        "FeH_dex",
        "Mass_Msun",
        "Radius_Rsun",
    ]

    required_metric_columns = [
        "param",
        "window",
        "R2",
        "MAE",
        "RMSE",
        "bias",
        "bias_std",
    ]

    missing_stars = set(required_star_columns) - set(stars.columns)
    missing_metrics = set(required_metric_columns) - set(metrics.columns)

    if missing_stars:
        raise ValueError(
            f"Missing columns in stellar dataset: {sorted(missing_stars)}"
        )

    if missing_metrics:
        raise ValueError(
            f"Missing columns in metrics CSV: {sorted(missing_metrics)}"
        )

    if stars[required_star_columns].isna().any().any():
        raise ValueError("Stellar parameters contain missing values.")

    if metrics[required_metric_columns].isna().any().any():
        raise ValueError("Model metrics contain missing values.")

    if metrics.duplicated(subset=["param", "window"]).any():
        raise ValueError(
            "Duplicate parameter/window combinations found in metrics."
        )

    return stars[required_star_columns], metrics[required_metric_columns]

# Create the database schema.
def create_tables(connection: sqlite3.Connection):
    connection.executescript(
        """
        DROP TABLE IF EXISTS model_metrics;
        DROP TABLE IF EXISTS stars;

        CREATE TABLE stars (
            star_id INTEGER PRIMARY KEY,
            Teff_K REAL NOT NULL,
            logg_cgs REAL NOT NULL,
            FeH_dex REAL NOT NULL,
            Mass_Msun REAL NOT NULL,
            Radius_Rsun REAL NOT NULL
        );

        CREATE TABLE model_metrics (
            metric_id INTEGER PRIMARY KEY,
            parameter TEXT NOT NULL,
            window TEXT NOT NULL,
            r2 REAL NOT NULL,
            mae REAL NOT NULL,
            rmse REAL NOT NULL,
            bias REAL NOT NULL,
            bias_std REAL NOT NULL,

            UNIQUE (parameter, window)
        );
        """
    )

# Import stellar parameters into the stars table.
def import_stars(connection: sqlite3.Connection, stars: pd.DataFrame):
    records = [
        (i, *row)
        for i, row in enumerate(stars.itertuples(index=False, name=None), start=1)
    ]

    connection.executemany(
        """
        INSERT INTO stars (
            star_id,
            Teff_K,
            logg_cgs,
            FeH_dex,
            Mass_Msun,
            Radius_Rsun
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        records,
    )

# Import model evaluation metrics.
def import_metrics(
    connection: sqlite3.Connection,
    metrics: pd.DataFrame,
):
    records = [
        (
            row["param"],
            row["window"],
            row["R2"],
            row["MAE"],
            row["RMSE"],
            row["bias"],
            row["bias_std"],
        )
        for _, row in metrics.iterrows()
    ]

    connection.executemany(
        """
        INSERT INTO model_metrics (
            parameter,
            window,
            r2,
            mae,
            rmse,
            bias,
            bias_std
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

# Print basic database verification results.
def verify_database(connection: sqlite3.Connection):
    for table in ("stars", "model_metrics"):
        count = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count} records")

    print("\nMetrics preview:")

    preview = pd.read_sql_query(
        """
        SELECT parameter, window, r2, mae, rmse, bias
        FROM model_metrics
        ORDER BY window, parameter
        LIMIT 5
        """,
        connection,
    )

    print(preview.to_string(index=False))


def main():
    stars, metrics = load_source_data()

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        create_tables(connection)
        import_stars(connection, stars)
        import_metrics(connection, metrics)
        verify_database(connection)

    print(f"\nDatabase created at: {DATABASE_PATH}")


if __name__ == "__main__":
    main()