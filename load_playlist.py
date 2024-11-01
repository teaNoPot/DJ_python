import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import os
from dotenv import load_dotenv

load_dotenv()

# Spotify Developer Credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')

# Set up Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope='playlist-modify-private playlist-read-private'))

def get_playlist_details(playlist_id):
    # Get the playlist tracks
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    track_list = []
    
    # Extract track details
    for item in tracks:
        track = item['track']
        track_info = {
            'Track ID': track['id'],
            'Track Name': track['name'],
            'Album Name': track['album']['name'],
            'Artist Name(s)': ', '.join([artist['name'] for artist in track['artists']]),
            'Release Date': track['album']['release_date'],
            'Duration (ms)': track['duration_ms'],
            'Popularity': track['popularity'],
            'Added By': item['added_by']['id'] if item['added_by'] else None,
            'Added At': item['added_at'],
            'Genres': ', '.join(sp.artist(track['artists'][0]['id'])['genres']) if track['artists'] else None,
            'Record Label': track['album']['label'] if 'label' in track['album'] else None,
            'Danceability': None,  # You would fetch this with audio features
            'Energy': None,        # You would fetch this with audio features
            'Key': None,           # You would fetch this with audio features
            'Loudness': None,      # You would fetch this with audio features
            'Mode': None,          # You would fetch this with audio features
            'Speechiness': None,   # You would fetch this with audio features
            'Acousticness': None,  # You would fetch this with audio features
            'Instrumentalness': None, # You would fetch this with audio features
            'Liveness': None,      # You would fetch this with audio features
            'Valence': None,       # You would fetch this with audio features
            'Tempo': None,         # You would fetch this with audio features
            'Time Signature': None  # You would fetch this with audio features
        }
        track_list.append(track_info)
    
    # Create a DataFrame
    df = pd.DataFrame(track_list)
    
    # Optionally, fetch audio features for each track
    if not df.empty:
        features = sp.audio_features(df['Track ID'].tolist())
        features_df = pd.DataFrame(features)
        
        # Merge features into main DataFrame
        df = df.merge(features_df, left_on='Track ID', right_on='id', suffixes=('', '_features'))
        df.drop(columns=['id'], inplace=True)  # Remove redundant 'id' column
    
    print(df.head())
    return df

def replace_playlist_tracks(playlist_id, ordered_tracks_df):
    # First, remove all tracks from the playlist
    sp.playlist_remove_all_occurrences_of_items(playlist_id, ordered_tracks_df['Track ID'].tolist())
    
    # Add the ordered tracks back into the playlist
    sp.playlist_add_items(playlist_id, ordered_tracks_df['Track ID'].tolist())

# Example usage:
playlist_id = '5WjpkWpcz6oLwEcbcSEsGy'
ordered_playlist = get_playlist_details(playlist_id)

output_filename = f"playlists/{playlist_id}.csv"
ordered_playlist.to_csv(output_filename, index=False)
# ordered_playlist = ordered_playlist.sort_values(by='Popularity', ascending=False)  # Example sorting by popularity

