from Agent import Agent
import torch
from data import DemoLoader
import os
import numpy as np
import utils
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('modelfile', metavar='MODELFILE', type=str, help='which model to load')
    parser.add_argument('--n-runs', metavar='N-RUNS', type=int, help='How often to run behavioral cloning')
    parser.add_argument('--n-steps', metavar='N-STEPS', type=int, help='Number of updates')
    return parser.parse_args()

def main():
    options = create_parser()
    use_transitions = False if 'drqn' in options.modelfile else True
    demoloader = DemoLoader('./Demonstrations', 1, load_transitions=use_transitions, discard_incompletes=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_transitions:
        n_image_inputs = len(demoloader[0][0]) - 1
        image_dim = np.array(demoloader[0][0][0]).shape
        action_dim = len(demoloader[0][1])
    else:
        n_image_inputs = len(demoloader[0][0]) - 1
        image_dim = np.array(demoloader[0][0][0][0]).shape
        action_dim = len(demoloader[0][1][0])

    env = utils.Environment('./Environment/deliveryservice')

    # perform supervised learning for behavioral cloning
    for run in range(options.n_runs):

        step = 0
        n_epochs = 5
        selected_model = utils.import_model(options.modelfile)
        network = selected_model.get_architecture(image_dim, n_image_inputs, action_dim)
        network = network.to(device)
        target_entropy = -np.prod(env.env.action_space.shape).item()
        agent = Agent(network, 0.9, demoloader, 3e-4, 1e-3, options.n_steps, target_entropy)

        if not os.path.exists(f'./results/CQ_SAC_{options.modelfile}_{run}'):
            os.makedirs(f'./results/CQ_SAC_{options.modelfile}_{run}')
        elif os.path.isfile(f'./results/CQ_SAC_{options.modelfile}_{run}/ckpt.pt'):
            # continue prior run
            state_dict = torch.load(f'./results/CQ_SAC_{options.modelfile}_{run}/ckpt.pt')
            agent.network.load_state_dict(state_dict['network'])
            agent.actor_optimizer.load_state_dict(state_dict['policy_optimizer'])
            agent.critic1_optimizer.load_state_dict(state_dict['critic1_optimizer'])
            agent.critic2_optimizer.load_state_dict(state_dict['critic2_optimizer'])
            step = state_dict['step']

        summary_writer = SummaryWriter(log_dir=f'./results/CQ_SAC_{options.modelfile}_{run}')

        while step < options.n_steps:

            policy_losses, q1_val_loss, q2_val_loss, entropies, a_loss, alphas = \
                agent.update(batch_size=16, n_epochs=n_epochs)

            step += 1
            summary_writer.add_scalar('Policy_Loss', np.mean(policy_losses), step)
            summary_writer.add_scalar('Q1_Loss', np.mean(q1_val_loss), step)
            summary_writer.add_scalar('Q2_Loss', np.mean(q2_val_loss), step)
            summary_writer.add_scalar('Policy_Entropy', np.mean(entropies), step)
            summary_writer.add_scalar("Entropy_Tuning_Loss", np.mean(entropies), step)
            summary_writer.add_scalar("Entropy_Tuning", np.mean(alphas), step)

            # evaluate agent in environment
            with torch.no_grad():
                done = False
                cum_reward = 0
                episode_length = 0
                obs = env.reset()
                while not done and episode_length < 250:
                    obs = [torch.tensor(o.transpose(2, 0, 1)).unsqueeze(0).to(device) if len(o.shape) > 2
                           else torch.tensor(o).unsqueeze(0).to(device) for o in obs]
                    actions, _, _ = agent.network.act(obs)
                    action = actions.squeeze().cpu().detach().numpy()
                    obs, rew, done, _ = env.step(action)
                    cum_reward += rew
                    episode_length += 1

            summary_writer.add_scalar('Cumulative_Reward', cum_reward, step)
            summary_writer.add_scalar('Episode_Length', episode_length, step)
            print(f"Step {step}")
            print(f"Cumulative Reward: {cum_reward}")
            print(f"Episode Length: {episode_length}")

            if not step % 100:
                # Save model and stop training
                checkpoint = {
                    'network': agent.network.state_dict(),
                    'policy_optimizer': agent.actor_optimizer.state_dict(),
                    'critic1_optimizer': agent.critic1_optimizer.state_dict(),
                    'critic2_optimizer': agent.critic2_optimizer.state_dict(),
                    'step': step
                }
                torch.save(checkpoint, f'./results/CQ_SAC_{options.modelfile}_{run}/ckpt.pt')

        summary_writer.close()


if __name__ == '__main__':
    main()