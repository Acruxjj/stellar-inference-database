from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_visualization import (
    plot_histograms,
    plot_scatter,
    plot_correlation_matrix,
    plot_hr_diagram,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "all_stars.parquet"
FIGURES_PATH = PROJECT_ROOT / "figures"


def main():
    df = pd.read_parquet(DATA_PATH)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_histograms(
        df,
        columns=["Teff_K", "logg_cgs", "FeH_dex"],
        labels=[
            r"$T_{\rm eff}$ (K)",
            r"$\log g$ (dex)",
            "[Fe/H] (dex)",
        ],
        title="Distribution of stellar parameters",
    )
    fig.savefig(
        FIGURES_PATH / "stellar_parameter_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig, _ = plot_hr_diagram(df)
    fig.savefig(
        FIGURES_PATH / "hr_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig, _ = plot_scatter(
        df,
        x="Mass_Msun",
        y="Radius_Rsun",
        xlabel=r"Mass ($M_\odot$)",
        ylabel=r"Radius ($R_\odot$)",
        title="Stellar mass–radius relation",
    )
    fig.savefig(
        FIGURES_PATH / "mass_radius_relation.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig, _ = plot_correlation_matrix(
        df,
        columns=[
            "Teff_K",
            "logg_cgs",
            "FeH_dex",
            "Mass_Msun",
            "Radius_Rsun",
        ],
        labels=[
            r"$T_{\rm eff}$",
            r"$\log g$",
            "[Fe/H]",
            r"$M/M_\odot$",
            r"$R/R_\odot$",
        ],
        title="Stellar parameter correlations",
    )
    fig.savefig(
        FIGURES_PATH / "parameter_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()