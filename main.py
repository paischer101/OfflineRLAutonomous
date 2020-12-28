import torch
import torch.nn as nn
from data import DemoLoader
import os
import numpy as np
import utils
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('modelfile', metavar='MODELFILE', type=str, help='which model to load')
    return parser.parse_args()

def main():
    options = create_parser()
    demoloader = DemoLoader('./Demonstrations', 1, load_transitions=True)

    n_image_inputs = len(demoloader[0][0]) - 1
    image_dim = np.array(demoloader[0][0][0]).shape
    output_dim = len(demoloader[0][1])
    selected_model = utils.import_model(options.modelfile)
    network = selected_model.get_architecture(image_dim, n_image_inputs, output_dim)


if __name__ == '__main__':
    main()
