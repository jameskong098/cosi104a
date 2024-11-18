"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW7
Date: 11/21/24

Description:
This script serves as the main entry point for performing clustering analysis on portfolio data and evaluating its potential to predict the NBER recession index.
"""
from data_loader import load_portfolio_data
from model import hierarchical_clustering, kmeans_clustering, plot_clusters, determine_optimal_clusters
import pandas as pd

def main():
    # Load portfolio data
    portfolio_file = "HW7Portfolio25.csv"
    portfolio_data = load_portfolio_data(portfolio_file)

    # Determine the optimal number of clusters
    determine_optimal_clusters(portfolio_data)

    # Perform hierarchical clustering
    hier_labels = hierarchical_clustering(portfolio_data)
    plot_clusters(portfolio_data, hier_labels, "Hierarchical Clustering", "hierarchical_clustering.png")

    # Perform K-means clustering
    kmeans_labels = kmeans_clustering(portfolio_data)
    plot_clusters(portfolio_data, kmeans_labels, "K-means Clustering", "kmeans_clustering.png")

    # Load recession data
    recession_file = "USREC.csv"
    recession_data = pd.read_csv(recession_file, index_col=0)

    # Evaluate if clustering results can predict recession
    # This part requires further analysis and explanation

if __name__ == '__main__':
    main()
