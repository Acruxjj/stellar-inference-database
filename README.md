# Stellar Data Explorer

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

**Note:** The stellar dataset is synthetic and reflects the parameter distributions used in its construction. It should not be interpreted as a representation of the actual Galactic M-dwarf population.

## Database

The project uses SQLite with two main tables:

- `stars`: stellar parameters.
- `model_metrics`: CNN evaluation metrics for each parameter and spectral window.

The database is generated from the input files using the database construction script.

## Project Structure

```text
stellar-inference-database/
├── app/
│   └── dashboard.py
├── data/
│   ├── all_stars.parquet
│   └── stellar_window_comparison.csv
├── database/
├── figures/
├── scripts/
│   ├── build_database.py
│   ├── query_database.py
│   └── visualize_dataset.py
├── src/
│   ├── data_diagnostic.py
│   ├── data_visualization.py
│   └── database_queries.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd stellar-inference-database
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

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

## Tests

Run the automated tests from the project root:

```bash
python -m pytest
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