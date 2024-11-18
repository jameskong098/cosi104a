# Clustering Analysis on Portfolio Data

COSI 116A - Introduction to Machine Learning

HW7

James Kong

11/21/24

## Overview

This project performs clustering analysis on a dataset `HW7Portfolio25.csv`, which contains the daily returns of 25 portfolios over 25438 trading days. Additionally, it evaluates the potential of using clustering results to predict the NBER recession index from the dataset `USREC.csv`.

## Files

- `data_loader.py`: Contains functions to load portfolio data from CSV files.
- `model.py`: Contains functions to perform hierarchical and K-means clustering analysis, and to plot clustering results.
- `main.py`: Orchestrates the data loading, clustering analysis, and evaluation steps.
- `HW7Portfolio25.csv`: Portfolio data file.
- `USREC.csv`: NBER recession index data file.

## Functions

### `data_loader.py`

- `load_portfolio_data(portfolio_file)`: Loads portfolio data from a CSV file.

### `model.py`

- `hierarchical_clustering(data, n_clusters=3)`: Performs hierarchical clustering analysis.
- `kmeans_clustering(data, n_clusters=3)`: Performs K-means clustering analysis.
- `plot_clusters(data, labels, title)`: Plots the clustering results.

### `main.py`

- `main()`: Orchestrates the data loading, clustering analysis, and evaluation steps.

## Tasks

### (a) Perform hierarchical clustering analysis

Hierarchical clustering is performed on the portfolio data with 3 clusters. This choice is based on the assumption that there may be three distinct groups of portfolios with similar return patterns.

### (b) Perform K-means clustering analysis

K-means clustering is performed on the portfolio data with 3 clusters. This choice is consistent with the hierarchical clustering analysis to compare the results and identify any similarities or differences.

### (c) Can the above clustering analysis results be used to predict the NBER recession index?

The clustering analysis results can provide insights into the patterns and relationships within the portfolio data. However, using clustering results directly to predict the NBER recession index requires further analysis and validation. The clustering labels can be compared with the recession index to identify any correlations or patterns that may indicate a predictive relationship.