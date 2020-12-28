import os
import importlib
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import gym

def import_model(model_path):
    """
    import model from file path
    """
    model_file = os.path.basename(model_path).split(".")[0]
    model = importlib.import_module("models.%s" % model_file)
    return model

class Environment(object):

    def __init__(self, scenepath):
        unity_env = UnityEnvironment(scenepath)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset()

    def close(self):
        self.env.close()
