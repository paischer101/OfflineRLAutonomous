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

    def __init__(self, scenepath, record_video=False):
        unity_env = UnityEnvironment(scenepath, no_graphics=True)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
        if record_video:
            self.env = gym.wrappers.Monitor(self.env, "./results/videos", video_callable=lambda episode_id: True,
                                            force=True)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset()

    def close(self):
        self.env.close()
