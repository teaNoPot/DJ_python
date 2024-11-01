from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from scipy.stats import pareto, gamma
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
from datetime import date
from find_optimal_cluster import find_optimal_clusters

## Dynamic scoring based on weights
def score_cluster(cluster_data, weights):
    cluster_data = cluster_data.copy()
    cluster_data['score'] = (
        weights.get('Energy', 0) * cluster_data.get('Energy', 0) +
        weights.get('Tempo', 0) * cluster_data.get('Tempo', 0) +
        weights.get('Danceability', 0) * cluster_data.get('Danceability', 0) +
        weights.get('Valence', 0) * cluster_data.get('Valence', 0) +
        weights.get('Loudness', 0) * cluster_data.get('Loudness', 0) +
        weights.get('Instrumentalness', 0) * (1 - cluster_data.get('Instrumentalness', 0)) +
        weights.get('release_ordinal', 0) * cluster_data.get('release_ordinal', 0)
    )
    return cluster_data.sort_values(by='score', ascending=False)

## SCORES with weights
mood_weights = {
    'energy_wave': {'Energy': 1.5, 'Tempo': 1.0, 'Danceability': 1.0, 'Loudness': 0.5},
    'mood_gradient': {'Valence': 1.5, 'Energy': 1.0, 'Tempo': 0.8, 'Loudness': 0.5},
    'tempo_run': {'Tempo': 2.0, 'Energy': 1.2, 'Danceability': 1.0, 'Loudness': 0.8},
    'nostalgia_mix': {'release_ordinal': 2.0, 'Energy': 0.5, 'Danceability': 0.5, 'Tempo': 0.3},
    'late_night_chill': {'Energy': 1.5, 'Loudness': 1.2, 'Instrumentalness': 1.0, 'Tempo': 0.8, 'Valence': 0.5}
}

## GENERATING PLAYLIST
def generate_playlist_combined(data, mood_category, switch_threshold=10, num_clusters=5):
    data = perform_clustering(data, num_clusters=num_clusters)
    ordered_clusters = sort_clusters_for_mood(data, mood_category)
    playlist_order = []

    # Process each cluster individually
    for cluster_data in ordered_clusters:
        current_song = cluster_data.iloc[0]
        playlist_order.append(current_song['Track ID'])

        # Repeat weighted selection within the cluster
        while len(cluster_data[~cluster_data['Track ID'].isin(playlist_order)]) > 0:
            # Create a copy to avoid modifying the original cluster_data slice
            potential_songs = cluster_data[~cluster_data['Track ID'].isin(playlist_order)].copy()
            
            # Calculate weighted distance for smooth transitions and assign without warning
            potential_songs['distance'] = (
                (potential_songs['Tempo'] - current_song['Tempo']).abs() +
                (potential_songs['Valence'] - current_song['Valence']).abs()
            )
            
            # If no close match is found, continue adding the remaining songs in order of score
            if all(potential_songs['distance'] > switch_threshold):
                playlist_order.extend(potential_songs.sort_values(by='score', ascending=False)['Track ID'].tolist())
                break
            
            # Select the closest match based on calculated distance
            next_song = potential_songs.sort_values(by='distance').iloc[0]
            playlist_order.append(next_song['Track ID'])
            current_song = next_song

    return playlist_order

def perform_clustering(data, num_clusters=5):
    required_columns = ['Energy', 'Valence', 'Danceability', 'Tempo']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for clustering: {missing_cols}")
    
    features = data[required_columns]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features_scaled)
    
    return data

def sort_clusters_for_mood(data, mood_category):
    if mood_category not in mood_weights:
        raise ValueError(f"Unknown mood category: {mood_category}")
    
    sorted_clusters = []
    weights = mood_weights[mood_category]

    for cluster in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster]
        sorted_cluster = score_cluster(cluster_data, weights)
        sorted_clusters.append(sorted_cluster)
    
    if mood_category in ['energy_wave', 'tempo_run']:
        sorted_clusters = sorted(sorted_clusters, key=lambda x: x['Energy'].mean() if mood_category == 'energy_wave' else x['Tempo'].mean())
    elif mood_category == 'mood_gradient':
        sorted_clusters = sorted(sorted_clusters, key=lambda x: x['Valence'].mean())
    elif mood_category == 'late_night_chill':
        sorted_clusters = sorted(sorted_clusters, key=lambda x: x['Energy'].mean(), reverse=False)

    return sorted_clusters

if __name__ == '__main__':
    filename = "moonz_weds.csv"
    data = pd.read_csv(filename, skipinitialspace=True)
    data.columns = data.columns.str.strip()

    print("GENERATING PLAYLIST")
    mood_category = 'tempo_run'  # This can be dynamically set by user input
    playlist_order = generate_playlist_combined(data, mood_category)

    ordered_playlist = data[data['Track ID'].isin(playlist_order)]
    ordered_playlist = ordered_playlist.set_index('Track ID').loc[playlist_order].reset_index()

    output_filename = "output4.csv"
    ordered_playlist.to_csv(output_filename, index=False)
