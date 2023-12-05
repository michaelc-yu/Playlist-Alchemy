import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable



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
        hn = hn[-1, :, :]
        out = self.relu(hn)
        out = self.fc1(out)  # first FC layer
        out = self.relu(out)  # ReLU
        out = self.fc2(out)  # second FC layer -> Final Output
        return out


def get_model(input_size, hidden_size, output_size, num_layers):
    """Instantiate a LSTM object and return it."""
    model = LSTM(input_size, hidden_size, output_size, num_layers=num_layers)
    return model

