from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 'Agg' backend for non-GUI environments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def hierarchical_clustering(data, n_clusters):
    """
    Perform hierarchical clustering analysis.

    Parameters:
    - data (DataFrame): The data to cluster.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - labels (ndarray): Cluster labels for each data point.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    print(f"\nRunning hierarchical clustering with {n_clusters} clusters...")
    labels = model.fit_predict(data)
    return labels

def kmeans_clustering(data, n_clusters, random_state, n_init):
    """
    Perform K-means clustering analysis.

    Parameters:
    - data (DataFrame): The data to cluster.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - labels (ndarray): Cluster labels for each data point.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    print(f"\nRunning K-means clustering with {n_clusters} clusters...\n")
    labels = model.fit_predict(data)
    return labels

def apply_pca(data, n_components):
    """
    Apply PCA to the data and return the transformed data.

    Parameters:
    - data (DataFrame): The data to transform.
    - n_components (int): The number of principal components to keep.

    Returns:
    - pca_data (ndarray): The data transformed by PCA.
    - explained_variance_ratio (ndarray): The amount of variance explained by each of the selected components.
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return pca_data, explained_variance_ratio

def calculate_cumulative_explained_variance(data, max_components=10, run=False):
    """
    Calculate and plot the cumulative explained variance for the given data.

    Parameters:
    - data (DataFrame): The data to analyze.
    - max_components (int): The maximum number of principal components to consider.
    """
    if not run:
        return
    
    pca = PCA(n_components=max_components)
    pca.fit(data)
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

    # Plot the cumulative explained variance
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, max_components + 1), cumulative_explained_variance, marker='o')
    plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('cumulative_explained_variance.png')
    plt.close()

def determine_optimal_clusters(data, max_clusters=10, run=False):
    """
    Determine the optimal number of clusters using the Elbow Method and Silhouette Score.

    Parameters:
    - data (DataFrame): The data to cluster.
    - max_clusters (int): The maximum number of clusters to test.

    Returns:
    - None
    """
    if not run:
        return
    print("\nDetermining the optimal number of clusters...")
    sse = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_init='auto', n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    # Plot the Elbow Method
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.savefig('./cluster_optimization/elbow_method.png')
    plt.close()

    # Plot the Silhouette Scores
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('./cluster_optimization/silhouette_scores.png')
    plt.close()
