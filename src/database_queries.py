import sqlite3

import pandas as pd


def get_star_count(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return the total number of stars in the database.
    return pd.read_sql_query(
        """
        SELECT COUNT(*) AS total_stars
        FROM stars
        """,
        connection,
    )


def get_hottest_stars(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return the five hottest stars.
    return pd.read_sql_query(
        """
        SELECT star_id, Teff_K, logg_cgs, FeH_dex
        FROM stars
        ORDER BY Teff_K DESC
        LIMIT 5
        """,
        connection,
    )


def get_model_metrics(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return model performance metrics for all parameters and windows.
    return pd.read_sql_query(
        """
        SELECT parameter, window, r2, mae, rmse, bias
        FROM model_metrics
        ORDER BY parameter, mae ASC
        """,
        connection,
    )


def get_best_windows(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return the best spectral window for each parameter based on MAE.
    return pd.read_sql_query(
        """
        SELECT
            m.parameter,
            m.window,
            m.mae,
            m.r2,
            m.rmse
        FROM model_metrics AS m
        WHERE m.mae = (
            SELECT MIN(m2.mae)
            FROM model_metrics AS m2
            WHERE m2.parameter = m.parameter
        )
        ORDER BY m.parameter
        """,
        connection,
    )


def get_mean_mae(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return the mean MAE across spectral windows for each parameter.
    return pd.read_sql_query(
        """
        SELECT
            parameter,
            AVG(mae) AS mean_mae
        FROM model_metrics
        GROUP BY parameter
        ORDER BY parameter
        """,
        connection,
    )


def get_mamajek_distribution(connection: sqlite3.Connection) -> pd.DataFrame:
    # Return the stellar distribution by approximate Mamajek subtype.
    # https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    return pd.read_sql_query(
        """
        SELECT
            CASE
                WHEN Teff_K >= 3755 THEN 'M0V'
                WHEN Teff_K >= 3610 THEN 'M1V'
                WHEN Teff_K >= 3495 THEN 'M2V'
                WHEN Teff_K >= 3320 THEN 'M3V'
                WHEN Teff_K >= 3135 THEN 'M4V'
                WHEN Teff_K >= 2935 THEN 'M5V'
                ELSE 'M6V'
            END AS approx_spectral_type,
            COUNT(*) AS number_of_stars,
            AVG(FeH_dex) AS mean_metallicity
        FROM stars
        GROUP BY approx_spectral_type
        ORDER BY MIN(Teff_K) DESC
        """,
        connection,
    )

def get_all_stars(connection: sqlite3.Connection) -> pd.DataFrame:
    # Retrieve all stellar parameters from the database.
    return pd.read_sql_query(
        """
        SELECT
            star_id,
            Teff_K,
            logg_cgs,
            FeH_dex,
            Mass_Msun,
            Radius_Rsun
        FROM stars
        """,
        connection,
    )


def get_best_windows(connection: sqlite3.Connection) -> pd.DataFrame:
    # Retrieve the spectral window with the lowest MAE for each parameter.
    return pd.read_sql_query(
        """
        SELECT
            m.parameter,
            m.window,
            m.mae,
            m.r2,
            m.rmse
        FROM model_metrics AS m
        WHERE m.mae = (
            SELECT MIN(m2.mae)
            FROM model_metrics AS m2
            WHERE m2.parameter = m.parameter
        )
        ORDER BY m.parameter
        """,
        connection,
    )