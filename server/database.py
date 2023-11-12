import re
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import helpers


MIN_DANCEABILITY = 0.0
MAX_DANCEABILITY = 0.961
MIN_ENERGY = 0.0000203
MAX_ENERGY = 0.998
MIN_LOUDNESS = -30.227
MAX_LOUDNESS = -0.713
MIN_TEMPO = 0
MAX_TEMPO = 208.046
MIN_VALENCE = 0
MAX_VALENCE = 0.977


word_vectors = helpers.get_word_vectors()

df = pd.read_csv('../lyrics_10k.csv')


df = df[:10:]
num_songs = len(df)
print(f"only looking at the first {num_songs} songs")


spotify = helpers.authenticate_spotify()


def get_lyric_embeddings():
    """Return embeddings for all song lyrics as training data for model."""
    inputs = []
    for i in range(num_songs):
        lyrics = df.iloc[i]['lyrics']
        lyric_tokens = helpers.tokenize(lyrics)

        embeddings = []
        for token in lyric_tokens:
            if token in word_vectors:
                embedding = word_vectors[token]
                embeddings.append(embedding)

        inputs.append(embeddings)

    max_sequence_length = max(len(one_input) for one_input in inputs)
    print(f"max sequence length: {max_sequence_length}")

    padded_inputs = []

    for one_input in inputs:
        pad_width = ((0, max_sequence_length - len(one_input)), (0, 0))
        padded_input = np.pad(one_input, pad_width, mode='constant', constant_values=0)
        padded_inputs.append(padded_input)

    inputs = np.stack(padded_inputs)
    print(f"inputs shape: {inputs.shape}") # torch.Size([3, 390, 50]) -> 3 songs, 390 is the max words in lyrics, 50 floats for each word as the vector embedding
    
    return inputs


def get_feature_embeddings():
    """Return embeddings of features for each song as training data for the model."""
    # give each song a score on these features, based on how often those words occur / total number of words in genre string
    key_genres = ['blues', 'dance', 'folk', 'hip', 'hop', 'indie', 'jazz', 'k-pop', 'lilith', 'metal', 'pop', 'rap', 'rock', 'soul', 'trap', 'wave']
    outputs = []
    for i in range(num_songs):
        one_output = []

        song_genres = df.iloc[i]['genres']

        if isinstance(song_genres, str):
            song_genres = re.split(r'[; ]', song_genres)
            song_genres = [word for word in song_genres if word in key_genres]
            print(f"genres: {song_genres}")

            for genre in key_genres:
                if len(song_genres) > 0:
                    ratio = song_genres.count(genre) / len(song_genres)
                    one_output.append(ratio)
                else:
                    one_output.append(0)
        else:
            for genre in key_genres:
                one_output.append(0)


        # audio features
        song_id = df.iloc[i]['song_id'].split(':')[2]
        audio_features = spotify.audio_features([song_id])[0]

        danceability = audio_features['danceability']
        normalized_danceability = (danceability - MIN_DANCEABILITY) / (MAX_DANCEABILITY - MIN_DANCEABILITY)

        energy = audio_features['energy']
        normalized_energy = (energy - MIN_ENERGY) / (MAX_ENERGY - MIN_ENERGY)

        loudness = audio_features['loudness']
        normalized_loudness = (loudness - MIN_LOUDNESS) / (MAX_LOUDNESS - MIN_LOUDNESS)

        tempo = audio_features['tempo']
        normalized_tempo = (tempo - MIN_TEMPO) / (MAX_TEMPO - MIN_TEMPO)

        valence = audio_features['valence']
        normalized_valence = (valence - MIN_VALENCE) / (MAX_VALENCE - MIN_VALENCE)

        one_output.append(normalized_danceability)
        one_output.append(normalized_energy)
        one_output.append(normalized_loudness)
        one_output.append(normalized_tempo)
        one_output.append(normalized_valence)

        outputs.append(one_output)

    outputs = np.stack(outputs)
    print(f"outputs shape: {outputs.shape}") # torch.Size([50, 20]) -> 50 songs, 21 features each

    return outputs


def get_user_tensor(user_input):
    """Generate a tensor object from the user input string."""
    user_tokens = helpers.tokenize(user_input)
    user_embeddings = []
    for token in user_tokens:
        if token in word_vectors:
            embedding = word_vectors[token]
            user_embeddings.append(embedding)

    user_array = np.array(user_embeddings)
    user_tensor = torch.tensor(user_array).unsqueeze(0)
    
    return user_tensor


def get_predictions(closest_indices, outputs):
    """Generate song predictions based on closest Euclidean distance between vectors."""
    for index in closest_indices:
        print(df.iloc[index]['song'])
        print(df.iloc[index]['artists'])
        print(df.iloc[index]['song_id'].split(':')[2])

        print(outputs[index])

        print("---")

        closest_vectors = outputs[closest_indices]
        print (f"closest vectors: {closest_vectors}")

