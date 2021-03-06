import torch.nn as nn
import torch

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
            ))
        self.encoders.append(nn.Linear(in_features=1, out_features=64))
        self.encoders = nn.ModuleList(self.encoders)
        self.fc_input_dim = self.feature_size()
        self.decoder = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, self.output_dim)
        self.sigma = nn.Linear(128, self.output_dim)

    def forward(self, X):
        batch_size = X[0].size(0)
        encoded = []
        for x, enc in zip(X, self.encoders):
            if not isinstance(enc, nn.Linear):
                out = enc(x)
                n_filters = out.shape[1]
                encoded.append(torch.mean(out.view(batch_size, n_filters, -1), dim=-1).view(batch_size, -1))
            else:
                encoded.append(enc(x).view(batch_size, -1))
        x = torch.stack(encoded).view(batch_size, -1)
        x = self.decoder(x)
        var = nn.functional.softplus(self.sigma(x)) + 1e-5
        return self.mu(x), var

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