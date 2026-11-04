# ⭐ Stellar Data Explorer

An interactive data exploration project based on synthetic stellar parameters and machine learning model performance.

The project uses data from an undergraduate research project in stellar astrophysics to demonstrate data processing, database management, SQL querying, scientific visualization, and interactive dashboard development.

## Project Overview

The project explores a synthetic dataset of 5,000 M-dwarf stars and evaluates the performance of convolutional neural networks (CNNs) in estimating stellar atmospheric parameters from different spectral windows.

The database stores stellar parameters and model evaluation metrics, allowing the data to be queried and explored through Python and SQL.

## Features

- **Data diagnostics:** inspection of dataset structure, missing values, duplicate records, and numerical properties.
- **Data visualization:** stellar parameter distributions, scatter plots, correlation matrices, and Hertzsprung-Russell diagrams.
- **Database management:** relational data storage using SQLite.
- **SQL queries:** retrieval and comparison of stellar parameters and model performance metrics.
- **Interactive dashboard:** exploration of stellar properties and spectral window performance using Streamlit.

## Dataset

The project uses two input datasets:

### Stellar parameters

`all_stars.parquet`

Contains 5,000 synthetic stellar records with the following parameters:

| Column | Description |
|---|---|
| `Teff_K` | Effective temperature (K) |
| `logg_cgs` | Surface gravity (cgs) |
| `FeH_dex` | Metallicity ([Fe/H], dex) |
| `Mass_Msun` | Stellar mass (solar masses) |
| `Radius_Rsun` | Stellar radius (solar radii) |

### Model evaluation metrics

`stellar_window_comparison.csv`

Contains model evaluation metrics for four spectral windows and three stellar parameters:

- Effective temperature (`Teff`)
- Surface gravity (`logg`)
- Metallicity (`FeH`)

The evaluation metrics include R², MAE, RMSE, bias, and bias standard deviation.

Let \(y_i\) be the reference value, \(\hat{y}_i\) the predicted value, \(\bar{y}\) the mean of the reference values, and \(n\) the number of evaluated samples. The residual is defined as:

\[
e_i = \hat{y}_i-y_i
\]

The metrics are calculated as follows:

**Coefficient of determination (R²)**

\[
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

Measures how well the predictions explain the variance in the reference values. Higher values generally indicate better predictive performance.

**Mean Absolute Error (MAE)**

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|e_i|
\]

Measures the average absolute prediction error.

**Root Mean Squared Error (RMSE)**

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}e_i^2}
\]

Measures the typical magnitude of prediction errors, giving greater weight to larger errors.

**Bias**

\[
\mathrm{Bias}=\frac{1}{n}\sum_{i=1}^{n}e_i
\]

Measures the mean signed prediction error. Positive values indicate systematic overestimation, while negative values indicate underestimation.

**Bias standard deviation**

\[
\sigma_{\mathrm{bias}}=
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(e_i-\mathrm{Bias})^2}
\]

Measures the dispersion of residuals around their mean, describing the variability of prediction errors after accounting for the mean bias.

**Note:** The stellar dataset is synthetic and reflects the parameter distributions used in its construction. It should not be interpreted as a representation of the actual Galactic M-dwarf population.

## Database

The project uses SQLite with two main tables:

- `stars`: stellar parameters.
- `model_metrics`: CNN evaluation metrics for each parameter and spectral window.

The database is generated from the input files using the database construction script.

## Usage

### 1. Build the database

From the project root, run:

```bash
python -m scripts.build_database
```

This creates the SQLite database at:

```text
database/stellar_inference.db
```

### 2. Run database queries

```bash
python -m scripts.query_database
```

### 3. Generate data visualizations

```bash
python -m scripts.visualize_dataset
```

### 4. Launch the interactive dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will be available locally at:

```text
http://localhost:8501
```

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- SQLite
- SQL
- Streamlit
- Pytest

## Project Context

This project is based on data generated and analyzed during an undergraduate research project in stellar astrophysics.

It was developed as a practical application of data analysis and software development techniques, connecting scientific data with database systems and interactive visualization.
