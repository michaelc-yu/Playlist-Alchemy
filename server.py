import helpers
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import re


word_vectors = helpers.get_word_vectors()

df = helpers.get_df('lyrics_10k.csv')



df = df[:100:]
num_songs = len(df)
print(f"only looking at the first {num_songs} songs")



# audio features
spotify = helpers.authenticate_spotify()

danceability = []
energy = []
loudness = []
tempo = []
valence = []
for i in range(num_songs):
    song_id = df.iloc[i]['song_id'].split(':')[2]
    audio_features = spotify.audio_features([song_id])[0]
    # print(audio_features)

    danceability.append(audio_features['danceability'])
    energy.append(audio_features['energy'])
    loudness.append(audio_features['loudness'])
    tempo.append(audio_features['tempo'])
    valence.append(audio_features['valence'])

min_danceability = np.min(danceability)
max_danceability = np.max(danceability)

min_energy = np.min(energy)
max_energy = np.max(energy)

min_loudness = np.min(loudness)
max_loudness = np.max(loudness)

min_tempo = np.min(tempo)
max_tempo = np.max(tempo)

min_valence = np.min(valence)
max_valence = np.max(valence)


# set up inputs for all songs
inputs = []
for i in range(num_songs):
    lyrics = df.iloc[i]['lyrics']
    lyric_tokens = helpers.tokenize(lyrics)
    # print(lyric_tokens)

    # get embeddings for each lyric token
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


# set up outputs for all songs
# give each song a score on these features, based on how often those words occur / total number of words in genre string
key_genres = ['blues', 'dance', 'folk', 'hip', 'hop', 'indie', 'jazz', 'k-pop', 'lilith', 'metal', 'pop', 'rap', 'rock', 'soul', 'trap', 'wave']
outputs = []
for i in range(num_songs):
    one_output = []

    # genres
    song_genres = df.iloc[i]['genres']
    
    song_genres = re.split(r'[; ]', song_genres)
    song_genres = [word for word in song_genres if word in key_genres]

    print(f"genres: {song_genres}")

    for genre in key_genres:
        if len(song_genres) > 0:
            ratio = song_genres.count(genre) / len(song_genres)
            one_output.append(ratio)
        else:
            one_output.append(0)


    # audio features

    song_id = df.iloc[i]['song_id'].split(':')[2]
    audio_features = spotify.audio_features([song_id])[0]
    # print(audio_features)

    danceability = audio_features['danceability']
    normalized_danceability = (danceability - min_danceability) / (max_danceability - min_danceability)

    energy = audio_features['energy']
    normalized_energy = (energy - min_energy) / (max_energy - min_energy)

    loudness = audio_features['loudness']
    normalized_loudness = (loudness - min_loudness) / (max_loudness - min_loudness)

    tempo = audio_features['tempo']
    normalized_tempo = (tempo - min_tempo) / (max_tempo - min_tempo)

    valence = audio_features['valence']
    normalized_valence = (valence - min_valence) / (max_valence - min_valence)

    one_output.append(normalized_danceability)
    one_output.append(normalized_energy)
    one_output.append(normalized_loudness)
    one_output.append(normalized_tempo)
    one_output.append(normalized_valence)

    outputs.append(one_output)


outputs = np.stack(outputs)
print(f"outputs shape: {outputs.shape}") # torch.Size([50, 20]) -> 50 songs, 21 features each


print(f"INPUTS: {inputs}")
print(f"OUTPUTS: {outputs}")


assert len(inputs) == len(outputs), "length of input should equal length of output"

assert np.all((outputs >= 0) & (outputs <= 1)), "each element in the output should have value between 0 and 1"



# Model

input_tensor = torch.tensor(inputs, dtype=torch.float32)
print(f"input tensor: {input_tensor}")
print(f"input shape: {input_tensor.shape}")


output_tensor = torch.tensor(outputs, dtype=torch.float32)
print(f"output tensor: {output_tensor}")
print(f"output shape: {output_tensor.shape}")


input_size = input_tensor.shape[2] # 50
print (f"input size {input_size}")
output_size = output_tensor.shape[1]  # 21
print (f"output size {output_size}")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # cell state

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc1(out)  # first FC layer
        out = self.relu(out)  # ReLU
        out = self.fc2(out)  # second FC layer -> Final Output
        return out


batch_size = 20
hidden_size = 64
learning_rate = 0.005 # 0.001 originally
num_epochs = 20
num_layers = 1
num_classes = 6


dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = LSTM(input_size, hidden_size, output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    total_loss = 0.0
    for input_batch, output_batch in dataloader:
        print (f"input batch: {input_batch}")
        print(f"output batch: {output_batch}")
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        model_outputs = model(input_batch)

        print("finished forward pass")
        print(f"model outputs: {model_outputs}")
        print(f"model outputs shape: {model_outputs.shape}")

        # Calculate loss
        loss = criterion(model_outputs, output_batch)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')



user_input = input("Enter text to generate playlist with: ")
user_tokens = helpers.tokenize(user_input)
user_embeddings = []
for token in user_tokens:
    if token in word_vectors:
        embedding = word_vectors[token]
        user_embeddings.append(embedding)

user_array = np.array(user_embeddings)

user_tensor = torch.tensor(user_array).unsqueeze(0)
print (f"user tensor shape: {user_tensor.shape}")

# Pass input through the model
model.eval()
with torch.no_grad():
    predictions = model(user_tensor)


print("Model Predictions:", predictions)


predicted_vector = predictions[0].numpy()
print(f"predicted vector: {predicted_vector}")

# Calculate Euclidean distance between predicted vector and each vector in our dataset
distances = np.linalg.norm(outputs - predicted_vector, axis=1)

closest_indices = np.argsort(distances)[:5]
print (f"closest indices: {closest_indices}")


for index in closest_indices:
    print(df.iloc[index]['song'])
    print(df.iloc[index]['artists'])
    print(outputs[index])


print("---")

closest_vectors = outputs[closest_indices]
print (f"closest vectors: {closest_vectors}")

