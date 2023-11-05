import os
import re
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from gensim.models import KeyedVectors


def authenticate_spotify():
    client_id = os.environ["client_id_key"]
    client_secret = os.environ["client_secret_key"]
    redirect_uri = "http://localhost:3000"
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return spotify

def get_df(path):
    df = pd.read_csv(path)
    return df

def get_word_vectors():
    word_vectors = KeyedVectors.load_word2vec_format('glove.6B/glove.6B.50d.txt', binary=False)
    return word_vectors

def tokenize(text):
    text_without_apostrophies = re.sub(r"'", '', text)
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text_without_apostrophies.lower())

def get_genre_idx_map(df):
    num_songs = len(df)
    s = set()

    for i in range(num_songs):
        genres = df.iloc[i]['genres']
        s.add(str(genres))

    print(f"num combination of genres: {len(s)}")

    genre_to_idx = {}
    for i, genre in enumerate(s):
        genre_to_idx[genre] = i

    return genre_to_idx

