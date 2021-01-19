import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Actor(nn.Module):

    def __init__(self, image_dim, n_image_inputs, output_dim):
        super(Actor, self).__init__()

        self.image_dim = image_dim
        self.n_image_inputs = n_image_inputs
        self.output_dim = output_dim
        self.encoders = []
        for i in range(n_image_inputs):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(self.image_dim[0], 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU()
            ))
        self.encoders.append(nn.Linear(in_features=1, out_features=64))
        self.fc_input_dim = self.feature_size()
        self.lstm_hidden = None
        self.lstm_layers = 1
        self.lstm_hidden_dim = 256
        self.lstm = nn.LSTM(self.fc_input_dim, self.lstm_hidden_dim, batch_first=True)
        self.mu = nn.Linear(self.lstm_hidden_dim, self.output_dim)
        self.sigma = nn.Linear(self.lstm_hidden_dim, self.output_dim)


    def init_hidden(self, batch_size):
        self.lstm_hidden = (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)),
                            Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)))


    def forward(self, X, seqlens):
        batch_size = X[0].size(0)
        seqlen = X[0].size(1)

        self.init_hidden(batch_size)
        device = next(self.parameters()).device

        encoded = []
        for x, enc in zip(X, self.encoders):
            seq_encoding = []
            for i in range(x.shape[1]):
                if not isinstance(enc, nn.Linear):
                    out = enc(x[:, i, ...].squeeze())
                    n_filters = out.shape[1]
                    seq_encoding.append(torch.mean(out.view(batch_size, n_filters, -1), dim=-1).view(batch_size, -1))
                else:
                    seq_encoding.append(enc(x[:, i, ...]).view(batch_size, -1))
            encoded.append(seq_encoding)
        encoded = [torch.stack(e) for e in encoded]
        x = torch.cat(encoded, dim=-1).view(batch_size, seqlen, -1)
        packed = pack_padded_sequence(x, torch.LongTensor(seqlens), batch_first=True, enforce_sorted=False)
        h_t, c_t = self.lstm_hidden
        hidden, _ = self.lstm(packed, (h_t, c_t))
        unpacked, seqlens = pad_packed_sequence(packed, batch_first=True)
        var = nn.functional.softplus(self.sigma(unpacked)) + 1e-5
        mu = self.mu(unpacked)
        return mu, var

    def inference(self, observation):
        batch_size = observation[0].size(0)

        encoded = []
        for x, enc in zip(observation, self.encoders):
            if not isinstance(enc, nn.Linear):
                out = enc(x)
                n_filters = out.shape[1]
                encoded.append(torch.mean(out.view(batch_size, n_filters, -1), dim=-1).view(batch_size, -1))
            else:
                encoded.append(enc(x).view(batch_size, -1))

        encoded = torch.cat(encoded, dim=-1).view(batch_size, 1, -1)
        hidden_init = self.lstm_hidden
        next_pred, (h_t, c_t) = self.lstm(encoded, hidden_init)
        self.lstm_hidden = (h_t, c_t)
        var = nn.functional.softplus(self.sigma(next_pred)) + 1e-5
        mu = self.mu(next_pred)
        return mu, var

    def feature_size(self):
        dummies = []
        dummies += [torch.zeros(1, *self.image_dim)] * self.n_image_inputs
        dummies += [torch.zeros((1, 1))]
        outs = []
        for d, en in zip(dummies, self.encoders):
            if not isinstance(en, nn.Linear):
                out = en(d)
                n_filters = out.shape[1]
                outs.append(torch.mean(out.view(1, n_filters, -1), dim=-1).view(1, -1).squeeze())
            else:
                outs.append(en(d).squeeze())
        outs = torch.stack(outs).view(1, -1)
        return outs.size(1)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.bias.data.fill_(0)
        nn.init.kaiming_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu', m))
        m.bias.data.fill_(0)

def get_architecture(image_dim, n_image_inputs, output_dim):
    network = Actor(image_dim, n_image_inputs, output_dim)
    network.apply(initialize_weights)
    return network