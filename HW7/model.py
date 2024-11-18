import matplotlib
matplotlib.use('Agg')  # 'Agg' backend for non-GUI environments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def hierarchical_clustering(data, n_clusters=3):
    """
    Perform hierarchical clustering analysis.

    Parameters:
    - data (DataFrame): The data to cluster.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - labels (ndarray): Cluster labels for each data point.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    print(f"\n Running hierarchical clustering with {n_clusters} clusters...")
    labels = model.fit_predict(data)
    return labels

def kmeans_clustering(data, n_clusters=3):
    """
    Perform K-means clustering analysis.

    Parameters:
    - data (DataFrame): The data to cluster.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - labels (ndarray): Cluster labels for each data point.
    """
    model = KMeans(n_init='auto', n_clusters=n_clusters, random_state=42)
    print(f"\n Running K-means clustering with {n_clusters} clusters...\n")
    labels = model.fit_predict(data)
    return labels

def plot_clusters(data, labels, title, filename):
    """
    Plot the clustering results and save as a PNG file.

    Parameters:
    - data (DataFrame): The data to plot.
    - labels (ndarray): Cluster labels for each data point.
    - title (str): The title of the plot.
    - filename (str): The filename to save the plot as.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def determine_optimal_clusters(data, max_clusters=10):
    """
    Determine the optimal number of clusters using the Elbow Method and Silhouette Score.

    Parameters:
    - data (DataFrame): The data to cluster.
    - max_clusters (int): The maximum number of clusters to test.

    Returns:
    - None
    """
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
    plt.savefig('elbow_method.png')
    plt.close()

    # Plot the Silhouette Scores
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_scores.png')
    plt.close()
