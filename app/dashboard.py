from pathlib import Path
import sys
import sqlite3
from contextlib import closing

import matplotlib.pyplot as plt
import streamlit as st


# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database_queries import (
    get_star_count,
    get_model_metrics,
    get_all_stars,
    get_best_windows,
)

from src.data_visualization import (
    plot_histograms,
    plot_scatter,
    plot_correlation_matrix,
    plot_hr_diagram,
)

DATABASE_PATH = PROJECT_ROOT / "database" / "stellar_inference.db"

st.set_page_config(
    page_title="Stellar Data Explorer",
    page_icon="⭐",
    layout="wide",
)


# Dashboard header
st.title("⭐ Stellar Data Explorer")

st.write(
    "An interactive interface for exploring stellar parameters "
    "and spectral window model performance."
)

# Check database availability
if not DATABASE_PATH.exists():
    st.error("Database not found. Run scripts/build_database.py first.")
    st.stop()

# Retrieve data from the database
with closing(sqlite3.connect(DATABASE_PATH)) as connection:
    star_count = get_star_count(connection)
    stars = get_all_stars(connection)
    metrics = get_model_metrics(connection)
    best_windows = get_best_windows(connection)

# Dataset overview
total_stars = int(star_count.loc[0, "total_stars"])
total_windows = metrics["window"].nunique()
total_parameters = metrics["parameter"].nunique()

st.subheader("Dataset overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total stars", f"{total_stars:,}")
col2.metric("Spectral windows", total_windows)
col3.metric("Stellar parameters", total_parameters)

st.divider()

# Stellar properties
st.header("Stellar properties")

st.caption(
    "Explore the distributions, correlations, and "
    "physical relationships between stellar parameters."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Distributions",
    "Parameter relationships",
    "HR diagram",
    "Correlation matrix",
])

# Parameter distributions
with tab1:

    fig, axes = plot_histograms(
        stars,
        columns=[
            "Teff_K",
            "logg_cgs",
            "FeH_dex",
            "Mass_Msun",
            "Radius_Rsun",
        ],
        labels=[
            r"$T_{\mathrm{eff}}$ (K)",
            r"$\log g$ (dex)",
            "[Fe/H] (dex)",
            r"Mass ($M_\odot$)",
            r"Radius ($R_\odot$)",
        ],
        bins=30,
        ncols=3,
        figsize=(15, 7),
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)

# Parameter relationships
with tab2:

    parameter_options = {
        "Effective temperature (K)": "Teff_K",
        "Surface gravity (dex)": "logg_cgs",
        "Metallicity (dex)": "FeH_dex",
        "Mass (solar masses)": "Mass_Msun",
        "Radius (solar radii)": "Radius_Rsun",
    }

    col_x, col_y = st.columns(2)

    x_label = col_x.selectbox(
        "X-axis",
        options=list(parameter_options.keys()),
    )

    y_label = col_y.selectbox(
        "Y-axis",
        options=list(parameter_options.keys()),
        index=2,
    )

    color_parameter = st.selectbox(
        "Color points by",
        options=list(parameter_options.keys()),
        index=1,
    )

    fig, ax = plot_scatter(
        stars,
        x=parameter_options[x_label],
        y=parameter_options[y_label],
        color_by=parameter_options[color_parameter],
        xlabel=x_label,
        ylabel=y_label,
        colorbar_label=color_parameter,
        point_size=12,
        alpha=0.7,
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


# Hertzsprung-Russell diagram
with tab3:

    fig, ax = plot_hr_diagram(
        stars,
        color_by="FeH_dex",
        figsize=(8, 6),
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


# Correlation matrix
with tab4:

    fig, ax = plot_correlation_matrix(
        stars,
        columns=[
            "Teff_K",
            "logg_cgs",
            "FeH_dex",
            "Mass_Msun",
            "Radius_Rsun",
        ],
        labels=[
            r"$T_{\mathrm{eff}}$",
            r"$\log g$",
            "[Fe/H]",
            "Mass",
            "Radius",
        ],
        method="spearman",
        figsize=(8, 7),
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


st.divider()


# Model performance
st.header("Model performance")

st.caption(
    "Compare the performance of convolutional neural networks "
    "across the evaluated spectral windows."
)


parameter_labels = {
    "Teff": "Effective temperature (K)",
    "logg": "Surface gravity (dex)",
    "FeH": "Metallicity (dex)",
}


selected_parameter = st.selectbox(
    "Select stellar parameter",
    options=metrics["parameter"].unique(),
    format_func=lambda value: parameter_labels.get(value, value),
)


filtered_metrics = metrics[
    metrics["parameter"] == selected_parameter
].sort_values("mae")


best_result = best_windows[
    best_windows["parameter"] == selected_parameter
]


if not best_result.empty:

    best_window = best_result.iloc[0]

    st.metric(
        "Lowest MAE",
        f"{best_window['mae']:.4f}",
    )

    st.write(
        f"Best-performing window: `{best_window['window']}`"
    )


col1, col2 = st.columns(2)


with col1:

    st.subheader("Mean absolute error (MAE)")

    st.bar_chart(
        filtered_metrics,
        x="window",
        y="mae",
    )


with col2:

    st.subheader("Coefficient of determination (R²)")

    st.bar_chart(
        filtered_metrics,
        x="window",
        y="r2",
    )


st.subheader("Detailed model metrics")

st.dataframe(
    metrics,
    width="stretch",
    hide_index=True,
)


st.divider()


# Data explorer
st.header("Data explorer")

st.write(
    "Browse the stellar parameter dataset."
)


st.dataframe(
    stars,
    width="stretch",
    hide_index=True,
)

st.caption(
    f"Displaying {len(stars):,} stellar records."
)