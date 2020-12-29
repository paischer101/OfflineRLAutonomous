import torch
import torch.nn as nn
from data import DemoLoader, Dataloader
import os
import numpy as np
import utils
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('modelfile', metavar='MODELFILE', type=str, help='which model to load')
    parser.add_argument('--n-runs', metavar='N-RUNS', type=int, help='How often to run behavioral cloning')
    parser.add_argument('--n-steps', metavar='N-STEPS', type=int, help='Number of updates')
    return parser.parse_args()

def main():
    options = create_parser()
    demoloader = DemoLoader('./Demonstrations', 1, load_transitions=True)

    n_image_inputs = len(demoloader[0][0]) - 1
    image_dim = np.array(demoloader[0][0][0]).shape
    output_dim = len(demoloader[0][1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = utils.Environment('./Environment/deliveryservice')
    dataloader = Dataloader(demoloader, batch_size=64, device=device)

    # perform supervised learning for behavioral cloning
    for run in range(options.n_runs):

        step = 0
        initial_lr = 1e-3
        min_lr = 1e-7
        n_epochs = 5
        lr = initial_lr
        lr_decay = (initial_lr - min_lr) / options.n_steps
        selected_model = utils.import_model(options.modelfile)
        network = selected_model.get_architecture(image_dim, n_image_inputs, output_dim)
        network = network.to(device)
        loss = nn.MSELoss()
        optim = torch.optim.Adam(params=network.parameters(), lr=lr)

        if not os.path.exists(f'./results/BC_{run}'):
            os.makedirs(f'./results/BC_{run}')
        elif os.path.isfile(f'./results/BC_{run}/ckpt.pt'):
            # continue prior run
            state_dict = torch.load(f'./results/BC_{run}/ckpt.pt')
            network.load_state_dict(state_dict['network'])
            optim.load_state_dict(state_dict['optim'])
            step = state_dict['step']

        summary_writer = SummaryWriter(log_dir=f'./results/BC_{run}')
        batch_generator = dataloader.yield_batches(infinite=True)

        while step < options.n_steps:
            for epoch in range(n_epochs):
                batch_mean_loss = []
                batch_total_loss = []
                batch_entropy = []
                observation, action = next(batch_generator)
                mu, sigma = network(observation)
                mu_error = loss(mu, action)
                entropy = Normal(loc=mu, scale=sigma).entropy().mean()
                error = mu_error - (entropy * 1e-3)
                optim.zero_grad()
                error.backward()
                optim.step()
                batch_mean_loss.append(mu_error.cpu().detach().item())
                batch_entropy.append(entropy.cpu().detach().item())
                batch_total_loss.append(error.cpu().detach().item())

            step += 1
            summary_writer.add_scalar('Mean_Loss', np.mean(batch_mean_loss), step)
            summary_writer.add_scalar('Total_Loss', np.mean(batch_total_loss), step)
            summary_writer.add_scalar('Entropy', np.mean(batch_entropy), step)

            # evaluate agent in environment
            with torch.no_grad():
                done = False
                cum_reward = 0
                episode_length = 0
                obs = env.reset()
                while not done:
                    obs = [torch.tensor(o.transpose(2, 0, 1)).unsqueeze(0).to(device) if len(o.shape) > 2
                           else torch.tensor(o).unsqueeze(0).to(device) for o in obs]
                    mu, sigma = network(obs)
                    dist = Normal(loc=mu, scale=sigma)
                    action = dist.sample().squeeze().cpu().detach().numpy()
                    obs, rew, done, _ = env.step(action)
                    cum_reward += rew
                    episode_length += 1

            summary_writer.add_scalar('Cumulative_Reward', cum_reward, step)
            summary_writer.add_scalar('Episode_Length', episode_length, step)
            print(f"Step {step}")
            print(f"Cumulative Reward: {cum_reward}")
            print(f"Episode Length: {episode_length}")
            print(f"Total Loss: {error.cpu().detach().item()}")

            # linearly decay learning rate
            lr -= lr_decay
            for p in optim.param_groups:
                p['lr'] = lr

            if not step % 1000:
                # Save model and stop training
                checkpoint = {
                    'network': network.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step
                }
                torch.save(checkpoint, f'./results/BC_{run}/ckpt.pt')
                break


if __name__ == '__main__':
    main()
