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

filename = "test_dataset.csv"

# read the data
data = pd.read_csv(filename, skipinitialspace=True)
data.columns = data.columns.str.strip()

features = data[['Tempo', 'Valence',  'Danceability', 'Energy', 'Key']].dropna()

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Elbow method to find optimal number of clusters
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    distortions.append(sum(np.min(cdist(features_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / features_scaled.shape[0])

# Clustering with the chosen number of clusters (adjust 'n_clusters' based on elbow method)
optimal_clusters = 5  # Replace with your elbow result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# IF data contains clusters calculated from previous stuff
def suggest_next_song(current_track_id):
    # Find current track's cluster
    current_song = data[data['Track ID'] == current_track_id]
    if current_song.empty:
        return "Track ID not found."
    
    cluster = current_song['Cluster'].values[0]
    # Find other songs in the same cluster, excluding the current song
    similar_songs = data[(data['Cluster'] == cluster) & (data['Track ID'] != current_track_id)]
    
    # Sort by tempo, valence, or any metric you choose for smooth transitions
    similar_songs = similar_songs.sort_values(by='Tempo', ascending=False)
    
    # Return a suggestion
    return similar_songs[['Track Name', 'Artist Name(s)', 'Tempo', 'Valence', 'Danceability']].head(5)

# This one generate the playlist 
def generate_playlist(data):
    # Make a copy to prevent modifying the original DataFrame
    playlist_data = data.copy()

    # List to hold the playlist order (track IDs)
    playlist_order = []

    # Select the first song (you could pick any starting song here; we'll use the first row)
    current_song = playlist_data.iloc[0]
    playlist_order.append(current_song['Track ID'])

    # Loop until we've added all songs to the playlist
    while len(playlist_order) < len(playlist_data):
        # Get the current song's cluster and filter for songs in the same cluster
        current_cluster = current_song['Cluster']
        potential_songs = playlist_data[(playlist_data['Cluster'] == current_cluster) & (~playlist_data['Track ID'].isin(playlist_order))]

        # If no songs are left in the cluster, switch to a new cluster
        if potential_songs.empty:
            potential_songs = playlist_data[~playlist_data['Track ID'].isin(playlist_order)]

        # Sort remaining songs by proximity to the current song's tempo or other feature (e.g., Valence)
        # Here we prioritize smooth transitions in `Tempo` and `Valence`
        potential_songs = potential_songs.copy()
        potential_songs.loc[:, 'distance'] = (potential_songs['Tempo'] - current_song['Tempo']).abs() + (potential_songs['Valence'] - current_song['Valence']).abs()
        next_song = potential_songs.sort_values(by='distance').iloc[0]
        
        # Add the selected song to the playlist and update the current song
        playlist_order.append(next_song['Track ID'])
        current_song = next_song

    # Return playlist order (Track IDs)
    return playlist_order


# Save playlist
# Generate the playlist order
playlist_order = generate_playlist(data)

# Output ordered playlist based on Track IDs
#ordered_playlist = data.set_index('Track ID').loc[playlist_order]

# Save the ordered playlist to a CSV file
output_filename = "ordered_playlist.csv"
playlist_order.to_csv(output_filename, index=False)