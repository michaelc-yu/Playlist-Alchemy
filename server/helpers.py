import os
import re
import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from gensim.models import KeyedVectors


def authenticate_spotify():
    """Authenticate with Spotify API using the secret access tokens."""
    client_id = os.environ["client_id_key"]
    client_secret = os.environ["client_secret_key"]
    redirect_uri = "http://localhost:3000"
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return spotify

def get_word_vectors():
    """Get word vector embeddings from the GloVe dataset."""
    word_vectors = KeyedVectors.load_word2vec_format('../glove.6B/glove.6B.50d.txt', binary=False)
    return word_vectors

def tokenize(text):
    """Get tokens array from an input string."""
    text_without_apostrophies = re.sub(r"'", '', text)
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text_without_apostrophies.lower())

# def get_genre_idx_map(df):
#     num_songs = len(df)
#     s = set()

#     for i in range(num_songs):
#         genres = df.iloc[i]['genres']
#         s.add(str(genres))

#     print(f"num combination of genres: {len(s)}")

#     genre_to_idx = {}
#     for i, genre in enumerate(s):
#         genre_to_idx[genre] = i

#     return genre_to_idx

# def query_features(df, spotify):
#     danceability = []
#     energy = []
#     loudness = []
#     tempo = []
#     valence = []
#     for i in range(8675):
#         song_id = df.iloc[i]['song_id'].split(':')[2]
#         audio_features = spotify.audio_features([song_id])[0]
#         # print(audio_features)

#         danceability.append(audio_features['danceability'])
#         energy.append(audio_features['energy'])
#         loudness.append(audio_features['loudness'])
#         tempo.append(audio_features['tempo'])
#         valence.append(audio_features['valence'])

#     min_danceability = np.min(danceability)
#     max_danceability = np.max(danceability)

#     min_energy = np.min(energy)
#     max_energy = np.max(energy)

#     min_loudness = np.min(loudness)
#     max_loudness = np.max(loudness)

#     min_tempo = np.min(tempo)
#     max_tempo = np.max(tempo)

#     min_valence = np.min(valence)
#     max_valence = np.max(valence)

#     print(f"min danceability: {min_danceability}")
#     print(f"max danceability: {max_danceability}")

#     print(f"min energy: {min_energy}")
#     print(f"max energy: {max_energy}")

#     print(f"min loudness: {min_loudness}")
#     print(f"max loudness: {max_loudness}")

#     print(f"min tempo: {min_tempo}")
#     print(f"max tempo: {max_tempo}")

#     print(f"min valence: {min_valence}")
#     print(f"max valence: {max_valence}")
