import torch
import torch.nn as nn
from data import DemoLoader
import numpy as np
np.random.seed(101)
import utils
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from copy import deepcopy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('modelfile', metavar='MODELFILE', type=str, help='which model to load')
    parser.add_argument('--checkpoint', metavar='CHECKPOINT', required=True, help='path to checkpoint')
    return parser.parse_args()

def main():
    options = create_parser()
    export_path = os.path.join('/'.join(options.checkpoint.split('/')[:-1]), 'visualizations')
    if os.path.exists(export_path):
        print("Visualizations for model already created!")
        exit(0)

    demoloader = DemoLoader('./Demonstrations', 1, load_transitions=True)
    n_image_inputs = len(demoloader[0][0]) - 1
    image_dim = np.array(demoloader[0][0][0]).shape
    output_dim = len(demoloader[0][1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    selected_model = utils.import_model(options.modelfile)
    network = selected_model.get_architecture(image_dim, n_image_inputs, output_dim)
    network = network.to(device)
    checkpoint = torch.load(options.checkpoint)
    network.load_state_dict(checkpoint['network'])
    network_dummy = deepcopy(network)
    network_dummy.mu = Identity()
    network_dummy.sigma = Identity()
    summary_writer = SummaryWriter(export_path)

    # extract single trajectory out of demonstrations to predict on those
    done_inds = [i for i in range(len(demoloader)) if demoloader[i][-1]]
    ind = np.random.choice(done_inds, size=1)
    trajectory = [demoloader[i] for i in range(done_inds[done_inds.index(ind)-1], ind[0])]

    embeddings = []
    for i, (observation, action, next_state, reward, done) in enumerate(trajectory):
        summary_writer.add_images('Observation0', np.expand_dims(observation[0], 1), global_step=i + 1, dataformats='NCHW')
        summary_writer.add_images('Observation1', np.expand_dims(observation[1], 1), global_step=i + 1, dataformats='NCHW')
        summary_writer.add_images('Observation2', np.expand_dims(observation[2], 1), global_step=i + 1, dataformats='NCHW')
        summary_writer.add_scalar('Distance', observation[3][0], global_step=i + 1)

        obs_1 = torch.Tensor(observation[0]).to(device).unsqueeze(0)
        obs_2 = torch.Tensor(observation[1]).to(device).unsqueeze(0)
        obs_3 = torch.Tensor(observation[2]).to(device).unsqueeze(0)
        obs_4 = torch.Tensor(observation[3]).to(device).unsqueeze(0)

        mu, sigma = network([obs_1, obs_2, obs_3, obs_4])
        dist = Normal(loc=mu, scale=sigma)
        samples = dist.sample_n(1000).squeeze().cpu().detach().numpy()
        summary_writer.add_histogram('PredictedAction0', values=samples[:, 0], global_step=i + 1)
        summary_writer.add_histogram('PredictedAction1', values=samples[:, 1], global_step=i + 1)
        summary_writer.add_histogram('Action0', values=np.full((100,), fill_value=action[0]), global_step=i + 1)
        summary_writer.add_histogram('Action1', values=np.full((100,), fill_value=action[1]), global_step=i + 1)

        emb, _ = network_dummy([obs_1, obs_2, obs_3, obs_4])
        embeddings.append(emb.detach().cpu().numpy())

    embeddings = np.array(embeddings).squeeze()
    projection = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(projection[1:-1, 0], projection[1:-1, 1])
    plt.scatter(projection[0, 0], projection[0, 1], marker='v', color='green', s=100)
    plt.scatter(projection[-1, 0], projection[-1, 1], marker='x', color='red', s=100)
    for i in range(len(projection)):
        plt.annotate(i, (projection[i, 0], projection[i, 1]))
    plt.plot(projection[:, 0], projection[:, 1])
    plt.grid(True)
    plt.title("TSNE Projection of last layer")
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(export_path, 'tsne_proj.png'))


if __name__ == '__main__':
    main()
