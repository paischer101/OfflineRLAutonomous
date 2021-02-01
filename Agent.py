import torch
from data import ReplayBuffer
import torch.nn as nn
from torch.distributions import Uniform
import numpy as np

class Agent:

    def __init__(self, network, gamma, demo_buffer, actor_lr, critic_lr, n_steps):
        self.network = network
        self.gamma = gamma
        self.alpha = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer = ReplayBuffer(demo_buffer, gamma, self.device)
        self.mse = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.network.critic1.parameters(), critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.network.critic2.parameters(), critic_lr)
        self.initial_lr_actor = actor_lr
        self.initial_lr_critic = critic_lr
        self.lr_actor = actor_lr
        self.lr_critic = critic_lr
        self.actor_lr_decay = (self.initial_lr_actor - 1e-7) / n_steps
        self.critic_lr_decay = (self.initial_lr_critic - 1e-7) / n_steps

    def update(self, batch_size, n_epochs):
        policy_losses = []
        q1_val_loss = []
        q2_val_loss = []
        entropies = []
        # optionally add entropy scaling loss for SAC and alpha lagrangian for CQL
        for e in range(n_epochs):

            states, actions, next_states, rewards, dones = self.buffer.sample(batch_size)

            # update for soft actor critic
            new_acts, log_probs, entropy = self.network.act(states)
            q_new_acts = self.network.evaluate(states, new_acts)
            policy_loss = (log_probs - q_new_acts).mean()

            # perform critic update
            q1_vals = self.network.critic1(states + [actions])
            q2_vals = self.network.critic2(states + [actions])
            next_actions, _ = self.network.sample_n(next_states, 10)
            new_next_states = [torch.stack([next_states[i]]*10).reshape(160, *next_states[i].shape[1:])
                               for i in range(len(next_states))]
            target_q1 = self.network.target1(new_next_states + [next_actions])
            target_q1 = target_q1.reshape(next_actions.shape).max(0)[0]
            target_q2 = self.network.target2(new_next_states + [next_actions])
            target_q2 = target_q2.reshape(next_actions.shape).max(0)[0]
            target_q = torch.min(target_q1, target_q2)
            done_mask = torch.where(dones == 1.0, 0, 1)
            td_target = rewards.unsqueeze(-1) + self.gamma * target_q.detach() * done_mask.unsqueeze(-1)
            td1_error = self.mse(q1_vals.squeeze(), td_target)
            td2_error = self.mse(q2_vals.squeeze(), td_target)

            curr_actions, curr_probs = self.network.sample_n(states, 10)
            next_actions, next_probs = self.network.sample_n(next_states, 10)
            rand_actions = Uniform(low=-1, high=1).rsample(sample_shape=next_actions.shape)
            rand_density = np.log(0.5 ** curr_actions.shape[-1])
            new_states = [torch.stack([states[i]] * 10).reshape(160, *states[i].shape[1:])
                               for i in range(len(states))]
            q1_rand = self.network.critic1(new_states + [rand_actions])
            q2_rand = self.network.critic2(new_states + [rand_actions])
            q1_rand = q1_rand.reshape(rand_actions.shape)
            q2_rand = q2_rand.reshape(rand_actions.shape)
            q1_curr_actions = self.network.critic1(new_states + [curr_actions])
            q2_curr_actions = self.network.critic2(new_states + [curr_actions])
            q1_curr_actions = q1_curr_actions.reshape(curr_actions.shape)
            q2_curr_actions = q2_curr_actions.reshape(curr_actions.shape)
            q1_next_actions = self.network.critic1(new_states + [next_actions])
            q2_next_actions = self.network.critic2(new_states + [next_actions])
            q1_next_actions = q1_next_actions.reshape(curr_actions.shape)
            q2_next_actions = q2_next_actions.reshape(curr_actions.shape)
            cat_q1 = torch.cat([q1_rand - rand_density, q1_next_actions - next_probs.detach(), q1_curr_actions - curr_probs.detach()], 1)
            cat_q2 = torch.cat([q2_rand - rand_density, q2_next_actions - next_probs.detach(), q2_curr_actions - curr_probs.detach()], 1)
            min_q1_loss = torch.logsumexp(cat_q1, dim=0).mean() * self.alpha
            min_q2_loss = torch.logsumexp(cat_q2, dim=0).mean() * self.alpha
            min_q1_loss = min_q1_loss - q1_vals.mean() * self.alpha
            min_q2_loss = min_q2_loss - q2_vals.mean() * self.alpha
            q1_loss = td1_error + min_q1_loss
            q2_loss = td2_error + min_q2_loss

            self.critic1_optimizer.zero_grad()
            q1_loss.backward(retain_graph=True)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            q2_loss.backward(retain_graph=True)
            self.critic2_optimizer.step()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.network.update_target(self.network.critic1, self.network.target1)
            self.network.update_target(self.network.critic2, self.network.target2)

            policy_losses.append(policy_loss.detach().item())
            q1_val_loss.append(q1_loss.detach().item())
            q2_val_loss.append(q2_loss.detach().item())
            entropies.append(entropy.detach().mean().item())

        return policy_losses, q1_val_loss, q2_val_loss, entropies
