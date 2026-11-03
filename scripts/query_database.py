from pathlib import Path
import sqlite3

from src.database_queries import (
    get_star_count,
    get_hottest_stars,
    get_model_metrics,
    get_best_windows,
    get_mean_mae,
    get_mamajek_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "stellar_inference.db"


def main():
    with sqlite3.connect(DATABASE_PATH) as connection:

        # 1. Count the stars
        star_count = get_star_count(connection)
        print("Number of Stars:")
        print(star_count.to_string(index=False))

        # 2. Inspect the hottest stars
        hottest_stars = get_hottest_stars(connection)
        print("\nHottest Stars")
        print(hottest_stars.to_string(index=False))

        # 3. Compare model metrics
        metrics = get_model_metrics(connection)
        print("\nModel Metrics:")
        print(metrics.to_string(index=False))

        # 4. Identify the best spectral window for each parameter
        best_windows = get_best_windows(connection)
        print("\nBest Window per parameter")
        print(best_windows.to_string(index=False))

        # 5. Calculate mean MAE for each parameter
        mean_mae = get_mean_mae(connection)
        print("\nMean MAE per parameter")
        print(mean_mae.to_string(index=False))

        # 6. Group stars by approximate Mamajek spectral subtype
        mamajek_distribution = get_mamajek_distribution(connection)
        print("\nStellar Distribuition by approximate Mamajek spectral subtype")
        print(mamajek_distribution.to_string(index=False))


if __name__ == "__main__":
    main()