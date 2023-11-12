import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import model
import database


# Setup input and output tensors for training

inputs = database.get_lyric_embeddings()
outputs = database.get_feature_embeddings()

# print(f"INPUTS: {inputs}")
# print(f"OUTPUTS: {outputs}")

assert len(inputs) == len(outputs), "length of input should equal length of output"
assert np.all((outputs >= 0) & (outputs <= 1)), "each element in the output should have value between 0 and 1"


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




# Initialize hyperparameters for model training
batch_size = 2
hidden_size = 64
learning_rate = 0.005 # 0.001 originally
num_epochs = 20
num_layers = 1
num_classes = 6

# Instantiate objects used for model training and eval
dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = model.get_model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
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


# Get user input
user_input = input("Enter text to generate playlist with: ")

user_tensor = database.get_user_tensor(user_input)
print (f"user tensor shape: {user_tensor.shape}")

# Pass input through the model to get predictions
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


database.get_predictions(closest_indices=closest_indices, outputs=outputs)

