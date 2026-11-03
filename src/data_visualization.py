from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_STYLE = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
}

plt.rcParams.update(_STYLE)


def plot_histograms(
    df: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str] | None = None,
    bins: int = 30,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """
    Plot histograms for one or more DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : sequence of str
        Columns to visualize.
    labels : sequence of str, optional
        Axis labels, in the same order as columns.
    bins : int, default=30
        Number of histogram bins.
    ncols : int, default=3
        Number of subplot columns.
    figsize : tuple of float, optional
        Figure dimensions.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig, axes
    """
    columns = list(columns)

    if not columns:
        raise ValueError("At least one column must be provided.")

    if ncols < 1:
        raise ValueError("ncols must be at least 1.")

    if labels is not None and len(labels) != len(columns):
        raise ValueError("'labels' must have the same length as 'columns'.")

    labels = list(labels) if labels is not None else columns
    nrows = int(np.ceil(len(columns) / ncols))

    if figsize is None:
        figsize = (5 * ncols, 3.8 * nrows)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax, column, label in zip(axes_flat, columns, labels):
            values = pd.to_numeric(df[column], errors="coerce").dropna()

            ax.hist(values, bins=bins, edgecolor="black")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")

        for ax in axes_flat[len(columns):]:
            ax.set_visible(False)

        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.tight_layout()

    return fig, axes


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    color_by: str | None = None,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] = (7, 5),
    point_size: float = 14,
    alpha: float = 0.7,
    xscale: str = "linear",
    yscale: str = "linear",
    invert_xaxis: bool = False,
    invert_yaxis: bool = False,
):
    """
    Create a scatter plot with optional color mapping and axis controls.

    Returns
    -------
    fig, ax
    """
    required = [x, y] + ([color_by] if color_by else [])
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        if color_by:
            scatter = ax.scatter(
                df[x],
                df[y],
                c=df[color_by],
                s=point_size,
                alpha=alpha,
                cmap="viridis",
            )
            fig.colorbar(
                scatter,
                ax=ax,
                label=colorbar_label or color_by,
            )
        else:
            ax.scatter(
                df[x],
                df[y],
                s=point_size,
                alpha=alpha,
            )

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if title:
            ax.set_title(title)

        if invert_xaxis:
            ax.invert_xaxis()

        if invert_yaxis:
            ax.invert_yaxis()

        fig.tight_layout()

    return fig, ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    method: str = "spearman",
    figsize: tuple[float, float] = (7, 6),
    title: str | None = None,
    cmap: str = "coolwarm",
    annotate: bool = True,
    annotation_size: int = 9,
):
    """
    Plot a correlation matrix for selected numeric columns.

    Parameters
    ----------
    method : {"pearson", "spearman", "kendall"}
        Correlation method.
    """
    if columns is None:
        numeric = df.select_dtypes(include="number")
    else:
        missing = [column for column in columns if column not in df.columns]

        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

        numeric = df[list(columns)].select_dtypes(include="number")

    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    corr = numeric.corr(method=method)

    if labels is not None and len(labels) != len(corr.columns):
        raise ValueError("'labels' must match the numeric columns selected.")

    tick_labels = list(labels) if labels is not None else list(corr.columns)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        image = ax.imshow(
            corr.to_numpy(),
            vmin=-1,
            vmax=1,
            cmap=cmap,
        )

        positions = np.arange(len(tick_labels))
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)

        if annotate:
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    value = corr.iloc[i, j]
                    text_color = (
                        "white"
                        if np.isfinite(value) and abs(value) > 0.6
                        else "black"
                    )
                    text = f"{value:.2f}" if np.isfinite(value) else "—"

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=annotation_size,
                        color=text_color,
                    )

        fig.colorbar(image, ax=ax, label=f"{method.capitalize()} correlation")

        if title:
            ax.set_title(title)

        fig.tight_layout()

    return fig, ax


def calculate_luminosity(
    df: pd.DataFrame,
    teff_column: str = "Teff_K",
    radius_column: str = "Radius_Rsun",
    solar_teff: float = 5772.0,
) -> pd.Series:
    """
    Estimate luminosity in solar units using the Stefan-Boltzmann relation.

    The result is derived from effective temperature and stellar radius.
    """
    if solar_teff <= 0:
        raise ValueError("'solar_teff' must be positive.")

    missing = [
        column
        for column in (teff_column, radius_column)
        if column not in df.columns
    ]

    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    teff = pd.to_numeric(df[teff_column], errors="coerce")
    radius = pd.to_numeric(df[radius_column], errors="coerce")

    luminosity = radius**2 * (teff / solar_teff) ** 4
    luminosity = luminosity.where(
        np.isfinite(luminosity) & (luminosity > 0)
    )

    return luminosity.rename("Luminosity_Lsun")


def plot_hr_diagram(
    df: pd.DataFrame,
    teff_column: str = "Teff_K",
    radius_column: str = "Radius_Rsun",
    color_by: str | None = "FeH_dex",
    xlabel: str = r"Effective temperature $T_{\rm eff}$ (K)",
    ylabel: str = r"$\log_{10}(L/L_\odot)$",
    colorbar_label: str = "[Fe/H] (dex)",
    title: str = "HR diagram (derived luminosity)",
    figsize: tuple[float, float] = (7, 5),
):
    """
    Plot an HR diagram using luminosity derived from Teff and radius.
    """
    luminosity = calculate_luminosity(
        df,
        teff_column=teff_column,
        radius_column=radius_column,
    )

    valid = (
        np.isfinite(pd.to_numeric(df[teff_column], errors="coerce"))
        & np.isfinite(luminosity)
        & (luminosity > 0)
    )

    plot_df = df.loc[valid].copy()
    plot_df["_log_luminosity"] = np.log10(luminosity.loc[valid])

    if color_by is not None and color_by not in plot_df.columns:
        raise KeyError(f"Column not found in DataFrame: {color_by}")

    fig, ax = plot_scatter(
        plot_df.assign(_log_luminosity=plot_df["_log_luminosity"]),
        x=teff_column,
        y="_log_luminosity",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        color_by=color_by,
        colorbar_label=colorbar_label,
        figsize=figsize,
        invert_xaxis=True,
    )

    return fig, ax