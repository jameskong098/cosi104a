"""
Author: James Kong
Course: COSI 116A - Introduction to Machine Learning
Assignment: HW8
Date: 11/28/24

Description:
This script serves as the main entry point for performing clustering analysis on portfolio data using PCA and evaluating its potential to predict the NBER recession index.
"""
from sklearn.metrics import normalized_mutual_info_score
from data_loader import load_data
from model import determine_optimal_clusters, hierarchical_clustering, kmeans_clustering, apply_pca, calculate_cumulative_explained_variance

def main():
    # Load portfolio data
    portfolio_file = "HW7Portfolio25.csv"
    portfolio_data = load_data(portfolio_file)

    # Load NBER recession index data
    nber_file = "USREC.csv"
    nber_data = load_data(nber_file)

    # Ensure the lengths of the datasets match
    min_length = min(len(portfolio_data), len(nber_data))
    portfolio_data = portfolio_data.iloc[:min_length]
    nber_labels = nber_data.iloc[:min_length].values.ravel()

    # Determine the optimal number of principal components
    calculate_cumulative_explained_variance(portfolio_data, max_components=10, run=False)
    optimal_components = 4
    print(f"\nOptimal number of principal components: {optimal_components}")

    # Apply PCA with the optimal number of components
    pca_data, explained_variance_ratio = apply_pca(portfolio_data, optimal_components)
    print(f"\nExplained variance ratio by each component: {explained_variance_ratio}")

    # Determine the optimal number of clusters
    determine_optimal_clusters(pca_data, run=False)

    # Perform hierarchical clustering
    hier_labels = hierarchical_clustering(pca_data, n_clusters=3)

    # Perform K-means clustering
    kmeans_labels = kmeans_clustering(pca_data, n_clusters=4, random_state=42, n_init="auto")

    # Compare clustering results with NBER recession index using NMI
    nmi_hierarchical = normalized_mutual_info_score(nber_labels, hier_labels)
    nmi_kmeans = normalized_mutual_info_score(nber_labels, kmeans_labels)

    print(f"NMI for Hierarchical Clustering: {nmi_hierarchical}")
    print(f"NMI for K-means Clustering: {nmi_kmeans}")

if __name__ == '__main__':
    main()
