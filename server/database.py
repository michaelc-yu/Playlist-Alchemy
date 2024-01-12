#
# Created by Michael Yu on 11/4/2023
#

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


MAX_SONGS = 100
df = df[:MAX_SONGS:]
num_songs = MAX_SONGS

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
    # print(f"max sequence length: {max_sequence_length}")

    padded_inputs = []

    for one_input in inputs:
        pad_width = ((0, max_sequence_length - len(one_input)), (0, 0))
        padded_input = np.pad(one_input, pad_width, mode='constant', constant_values=0)
        padded_inputs.append(padded_input)

    inputs = np.stack(padded_inputs)
    # print(f"inputs shape: {inputs.shape}") # torch.Size([3, 390, 50]) -> 3 songs, 390 is the max words in lyrics, 50 floats for each word as the vector embedding
    
    return inputs


def get_feature_embeddings():
    """Return embeddings of features for each song as training data for the model."""
    # give each song a score on these features, based on how often those words occur / total number of words in genre string
    key_genres = ['blues', 'dance', 'folk', 'hip', 'hop', 'indie', 'jazz', 'k-pop', 'lilith', 'metal', 'pop', 'rap', 'rock', 'soul', 'trap', 'wave'] #  16 genres
    outputs = []

    for i in range(num_songs):
        one_output = []
        song_genres = df.iloc[i]['genres']

        if isinstance(song_genres, str):
            song_genres = re.split(r'[; ]', song_genres)
            song_genres = [word for word in song_genres if word in key_genres]
            # print(f"genres: {song_genres}")

            for genre in key_genres:
                if len(song_genres) > 0:
                    ratio = song_genres.count(genre) / len(song_genres)
                    one_output.append(ratio)
                else:
                    one_output.append(0)
        else:
            for genre in key_genres:
                one_output.append(0)

        outputs.append(one_output)

    tracks = []
    audio_features_arr = []
    for i in range(num_songs):
        # audio features
        song_id = df.iloc[i]['song_id'].split(':')[2]

        tracks.append(song_id)
        if len(tracks) == 100:
            # print ("making a spotify api call")
            # print (f"tracks: {tracks}")
            audio_features = spotify.audio_features(tracks=tracks)
            # print (f"audio_features: {audio_features}")
            # print (f"length audio features: {len(audio_features)}")

            for j in range(len(audio_features)):
                one_audio_feature = {}
                danceability = audio_features[j]['danceability']
                normalized_danceability = (danceability - MIN_DANCEABILITY) / (MAX_DANCEABILITY - MIN_DANCEABILITY)

                energy = audio_features[j]['energy']
                normalized_energy = (energy - MIN_ENERGY) / (MAX_ENERGY - MIN_ENERGY)

                loudness = audio_features[j]['loudness']
                normalized_loudness = (loudness - MIN_LOUDNESS) / (MAX_LOUDNESS - MIN_LOUDNESS)

                tempo = audio_features[j]['tempo']
                normalized_tempo = (tempo - MIN_TEMPO) / (MAX_TEMPO - MIN_TEMPO)

                valence = audio_features[j]['valence']
                normalized_valence = (valence - MIN_VALENCE) / (MAX_VALENCE - MIN_VALENCE)

                one_audio_feature['danceability'] = normalized_danceability
                one_audio_feature['energy'] = normalized_energy
                one_audio_feature['loudness'] = normalized_loudness
                one_audio_feature['tempo'] = normalized_tempo
                one_audio_feature['valence'] = normalized_valence

                audio_features_arr.append(one_audio_feature)
            tracks.clear()

    # print (f"audio features array: {audio_features_arr}")

    assert len(outputs) == len(audio_features_arr), "length of outputs should equal length of audio features array"

    for i in range(len(audio_features_arr)):
        one_output = []

        danceability = audio_features_arr[i]['danceability']
        if danceability > 1:
            danceability = 1
        if danceability < 0:
            danceability = 0
        energy = audio_features_arr[i]['energy']
        if energy > 1:
            energy = 1
        if energy < 0:
            energy = 0
        loudness = audio_features_arr[i]['loudness']
        if loudness > 1:
            loudness = 1
        if loudness < 0:
            loudness = 0
        tempo = audio_features_arr[i]['tempo']
        if tempo > 1:
            tempo = 1
        if tempo < 0:
            tempo = 0
        valence = audio_features_arr[i]['valence']
        if valence > 1:
            valence = 1
        if valence < 0:
            valence = 0

        one_output.append(danceability)
        one_output.append(energy)
        one_output.append(loudness)
        one_output.append(tempo)
        one_output.append(valence)

        outputs[i].extend(one_output)


    outputs = np.stack(outputs)
    # print(f"outputs shape: {outputs.shape}") # torch.Size([x, 21]) -> x songs, 21 features each

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
    for i, index in enumerate(closest_indices):
        print("---")
        print(f"{i+1}:")
        print(df.iloc[index]['song'])
        print(df.iloc[index]['artists'])

        # print(outputs[index])

    uris = []
    for index in closest_indices:
        uri = df.iloc[index]['song_id']
    
    return uris
