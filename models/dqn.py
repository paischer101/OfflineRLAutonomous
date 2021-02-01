import torch.nn as nn
import torch
from torch.distributions import Normal
from copy import deepcopy

class Critic(nn.Module):
    def __init__(self, image_dim, n_image_inputs, action_dim):
        super(Critic, self).__init__()

        self.image_dim = image_dim
        self.n_image_inputs = n_image_inputs
        self.action_dim = action_dim
        self.encoders = []
        for i in range(n_image_inputs):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(self.image_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            ))
        self.encoders.append(nn.Linear(in_features=1, out_features=64))
        self.encoders.append(nn.Linear(in_features=action_dim, out_features=64))
        self.fc_input_dim = self.feature_size()
        self.decoder = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

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
        return self.decoder(x)

    def feature_size(self):
        dummies = []
        dummies += [torch.zeros(1, *self.image_dim)] * self.n_image_inputs
        dummies += [torch.zeros((1, 1))]
        dummies += [torch.zeros(self.action_dim, )]
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

class Actor(nn.Module):

    def __init__(self, image_dim, n_image_inputs, output_dim):
        super(Actor, self).__init__()

        self.image_dim = image_dim
        self.n_image_inputs = n_image_inputs
        self.output_dim = output_dim
        self.encoders = []
        for i in range(n_image_inputs):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(self.image_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            ))
        self.encoders.append(nn.Linear(in_features=1, out_features=64))
        self.fc_input_dim = self.feature_size()
        self.decoder = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
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

class ActorCritic(nn.Module):

    def __init__(self, image_dim, n_image_inputs, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = Actor(image_dim, n_image_inputs, n_actions)
        self.critic1 = Critic(image_dim, n_image_inputs, n_actions)
        self.critic2 = Critic(image_dim, n_image_inputs, n_actions)
        self.target1 = deepcopy(self.critic1)
        self.target2 = deepcopy(self.critic2)
        self.tau = 1e-2

    def forward(self, x):
        raise NotImplementedError

    def update_target(self, net, target):
        new_target = target.state_dict().copy()
        for (k_t, v_t), (k, v) in zip(target.state_dict().items(), net.state_dict().items()):
            new_target[k] = (1 - self.tau) * v_t + self.tau * v
        target.load_state_dict(new_target)

    def sample_n(self, state, n):
        mu, sigma = self.actor(state)
        policy_dist = Normal(loc=mu, scale=sigma)
        actions = policy_dist.sample_n(n)
        probs = policy_dist.log_prob(actions)
        return actions, probs

    def act(self, state):
        mu, sigma = self.actor(state)
        dist = Normal(loc=mu, scale=sigma)
        action = dist.sample().squeeze()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()

    def evaluate(self, state, action):
        q_vals = torch.min(self.critic1(state + [action]), self.critic2(state + [action]))
        return q_vals.detach()

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.bias.data.fill_(0)
        nn.init.kaiming_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu', m))
        m.bias.data.fill_(0)

def get_architecture(image_dim, n_image_inputs, output_dim):
    network = ActorCritic(image_dim, n_image_inputs, output_dim)
    network.apply(initialize_weights)
    return network