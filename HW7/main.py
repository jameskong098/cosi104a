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

def main():
    # Load portfolio data
    portfolio_file = "HW7Portfolio25.csv"
    portfolio_data = load_portfolio_data(portfolio_file)

    # Determine the optimal number of clusters
    determine_optimal_clusters(portfolio_data, run=False)

    features = [
        "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM", "ME2 BM1", "ME2 BM2", "ME2 BM3",
        "ME2 BM4", "ME2 BM5", "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5", "ME4 BM1",
        "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5", "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"
    ]

    # Perform hierarchical clustering
    hier_labels = hierarchical_clustering(portfolio_data, n_clusters=3)
    plot_clusters(portfolio_data, hier_labels, "Hierarchical Clustering", "hierarchical_clustering.png", features)

    # Perform K-means clustering
    kmeans_labels = kmeans_clustering(portfolio_data, n_clusters=3)
    plot_clusters(portfolio_data, kmeans_labels, "K-means Clustering", "kmeans_clustering.png", features)


if __name__ == '__main__':
    main()
