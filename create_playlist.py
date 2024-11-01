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

# Scope to allow modifying user's playlists
SCOPE = 'playlist-modify-public'  # or 'playlist-modify-private' if you want a private playlist

# Read Track IDs from the CSV file
output_filename = "output4.csv"
playlist_name = "Tempo moonz playlist"
playlist_description = "Generated by yours truly - Tiramisu"

def create_playlist(output_filename, playlist_name, playlist_description):
    # Authenticate with Spotify
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                client_secret=CLIENT_SECRET,
                                                redirect_uri=REDIRECT_URI,
                                                scope=SCOPE))

    playlist_data = pd.read_csv(output_filename)
    track_ids = playlist_data['Track ID'].tolist()

    user_id = sp.me()['id']  # Get your own Spotify user ID

    # Create new playlist
    new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
    playlist_id = new_playlist['id']

    # Spotify's API only allows adding up to 100 tracks at once, so we add in batches
    for i in range(0, len(track_ids), 100):
        sp.playlist_add_items(playlist_id, track_ids[i:i + 100])

    print(f"Playlist '{playlist_name}' created successfully!")


create_playlist(output_filename=output_filename, playlist_name=playlist_name, playlist_description=playlist_description)
