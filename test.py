import torch
import numpy as np
import utils
from argparse import ArgumentParser
from torch.distributions import Normal

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('modelfile', metavar='MODELFILE', type=str, help='which model to load')
    parser.add_argument('--run', metavar='RUN', type=int, help='Which run to test')
    parser.add_argument('--test-runs', metavar='N-RUNS', type=int, help='How many runs to evaluate on environment')
    parser.add_argument('--env', metavar='ENV', type=str, help='Path to environment on which to evaluate')
    return parser.parse_args()

def main():
    options = create_parser()
    use_transitions = False if 'drqn' in options.modelfile else True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = utils.Environment(options.env, record_video=True)
    selected_model = utils.import_model(options.modelfile)
    network = selected_model.get_architecture((4, 84, 84), 3, 2)
    state_dict = torch.load(f'./results/BC_{options.modelfile}_{options.run}/ckpt.pt')
    network.load_state_dict(state_dict['network'])
    network = network.to(device)

    # perform supervised learning for behavioral cloning
    cum_rewards = []
    for _ in range(options.test_runs):

        # evaluate agent in environment
        with torch.no_grad():
            done = False
            cum_reward = 0
            episode_length = 0
            obs = env.reset()
            if not use_transitions:
                network.init_hidden(1)
            while not done:
                obs = [torch.tensor(o.transpose(2, 0, 1)).unsqueeze(0).to(device) if len(o.shape) > 2
                       else torch.tensor(o).unsqueeze(0).to(device) for o in obs]
                if use_transitions:
                    mu, sigma = network(obs)
                else:
                    mu, sigma = network.inference(obs)
                dist = Normal(loc=mu, scale=sigma)
                action = dist.sample().squeeze().cpu().detach().numpy()
                obs, rew, done, _ = env.step(action)
                cum_reward += rew
                episode_length += 1
            cum_rewards.append(cum_reward)

    print(f"AVERAGE CUMULATIVE REWARD OVER {options.test_runs} RUNS: {np.mean(cum_rewards)}")


if __name__ == '__main__':
    main()
