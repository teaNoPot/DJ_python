from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import matplotlib.pyplot as plt

def find_optimal_clusters(data, max_k=10):
    features = data[['Energy', 'Valence', 'Danceability', 'Tempo']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    ssd = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        ssd.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k+1), ssd, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k')
    plt.show()


def find_optimal_clusters_silhouette(data, max_k=10):
    features = data[['Energy', 'Valence', 'Danceability', 'Tempo']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()


def find_optimal_clusters_davies_bouldin(data, max_k=10):
    features = data[['Energy', 'Valence', 'Danceability', 'Tempo']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    db_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        db_score = davies_bouldin_score(features_scaled, cluster_labels)
        db_scores.append(db_score)
    
    plt.plot(range(2, max_k+1), db_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Index for Optimal k')
    plt.show()

