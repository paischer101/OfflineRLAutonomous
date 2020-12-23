import os
from typing import List, Tuple
import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.timers import timed, hierarchical_timer
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
from collections import defaultdict
import matplotlib.pyplot as plt

INITIAL_POS = 33
SUPPORTED_DEMONSTRATION_VERSIONS = frozenset([0, 1])


class DemoLoader(object):

    def __init__(self, path, sequence_length, pad_sequences=False):
        if not os.path.exists(path):
            print(f"Directory {path} does not exist")
            exit(1)
        files = self.get_demo_files(path)
        filelist = [f for f in files]
        demo_buffer = AgentBuffer()
        print("Loading Demonstration data...")
        for f in filelist:
            _, buffer = self.demo_to_buffer(f, sequence_length)
            buffer.resequence_and_append(target_buffer=demo_buffer, training_length=sequence_length)
        del buffer
        self.buffer = dict()
        self.buffer['done'] = np.array(demo_buffer['done'])
        self.buffer['rewards'] = self.split_sequences(np.array(demo_buffer['rewards']), self.buffer['done'])
        self.buffer['obs_0'] = self.split_sequences(np.array(demo_buffer['obs_0']), self.buffer['done'])
        self.buffer['obs_1'] = self.split_sequences(np.array(demo_buffer['obs_1']), self.buffer['done'])
        self.buffer['obs_2'] = self.split_sequences(np.array(demo_buffer['obs_2']), self.buffer['done'])
        self.buffer['obs_3'] = self.split_sequences(np.array(demo_buffer['obs_3']), self.buffer['done'])
        self.buffer['action'] = self.split_sequences(np.array(demo_buffer['continuous_action']), self.buffer['done'])
        self.buffer['prev_action'] = self.split_sequences(np.array(demo_buffer['prev_action']), self.buffer['done'])
        del demo_buffer
        print("Demonstrations loaded!")
        self.visualize_sequence(self.buffer['obs_0'][0])

    def get_demo_files(self, path: str) -> List[str]:
        """
        Retrieves the demonstration file(s) from a path.
        :param path: Path of demonstration file or directory.
        :return: List of demonstration files
        Raises errors if |path| is invalid.
        """
        if os.path.isfile(path):
            if not path.endswith(".demo"):
                raise ValueError("The path provided is not a '.demo' file.")
            return [path]
        elif os.path.isdir(path):
            paths = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".demo")
            ]
            if not paths:
                raise ValueError("There are no '.demo' files in the provided directory.")
            return paths
        else:
            raise FileNotFoundError(
                f"The demonstration file or directory {path} does not exist."
            )

    def load_demonstration(self, file_path) -> Tuple[BehaviorSpec, List[AgentInfoActionPairProto], int]:
        """
        Loads and parses a demonstration file.
        :param file_path: Location of demonstration file (.demo).
        :return: BrainParameter and list of AgentInfoActionPairProto containing demonstration data.
        """

        # First 32 bytes of file dedicated to meta-data.
        file_paths = self.get_demo_files(file_path)
        behavior_spec = None
        brain_param_proto = None
        info_action_pairs = []
        total_expected = 0
        for _file_path in file_paths:
            with open(_file_path, "rb") as fp:
                with hierarchical_timer("read_file"):
                    data = fp.read()
                next_pos, pos, obs_decoded = 0, 0, 0
                while pos < len(data):
                    next_pos, pos = _DecodeVarint32(data, pos)
                    if obs_decoded == 0:
                        meta_data_proto = DemonstrationMetaProto()
                        meta_data_proto.ParseFromString(data[pos: pos + next_pos])
                        if (
                                meta_data_proto.api_version
                                not in SUPPORTED_DEMONSTRATION_VERSIONS
                        ):
                            raise RuntimeError(
                                f"Can't load Demonstration data from an unsupported version ({meta_data_proto.api_version})"
                            )
                        total_expected += meta_data_proto.number_steps
                        pos = INITIAL_POS
                    if obs_decoded == 1:
                        brain_param_proto = BrainParametersProto()
                        brain_param_proto.ParseFromString(data[pos: pos + next_pos])
                        pos += next_pos
                    if obs_decoded > 1:
                        agent_info_action = AgentInfoActionPairProto()
                        agent_info_action.ParseFromString(data[pos: pos + next_pos])
                        if behavior_spec is None:
                            behavior_spec = behavior_spec_from_proto(
                                brain_param_proto, agent_info_action.agent_info
                            )
                        info_action_pairs.append(agent_info_action)
                        if len(info_action_pairs) == total_expected:
                            break
                        pos += next_pos
                    obs_decoded += 1
        if not behavior_spec:
            raise RuntimeError(
                f"No BrainParameters found in demonstration file at {file_path}."
            )
        return behavior_spec, info_action_pairs, total_expected

    @timed
    def demo_to_buffer(self, file_path, sequence_length: int, expected_behavior_spec: BehaviorSpec = None
                       ) -> Tuple[BehaviorSpec, AgentBuffer]:
        """
        Loads demonstration file and uses it to fill training buffer.
        :param expected_behavior_spec:
        :param file_path: Location of demonstration file (.demo).
        :param sequence_length: Length of trajectories to fill buffer.
        :return:
        """
        behavior_spec, info_action_pair, _ = self.load_demonstration(file_path)
        demo_buffer = self.make_demo_buffer(info_action_pair, behavior_spec, sequence_length)
        if expected_behavior_spec:
            # check action dimensions in demonstration match
            if behavior_spec.action_spec != expected_behavior_spec.action_spec:
                raise RuntimeError(
                    "The actions {} in demonstration do not match the policy's {}.".format(
                        behavior_spec.action_spec, expected_behavior_spec.action_spec
                    )
                )
            # check observations match
            if len(behavior_spec.observation_shapes) != len(
                    expected_behavior_spec.observation_shapes
            ):
                raise RuntimeError(
                    "The demonstrations do not have the same number of observations as the policy."
                )
            else:
                for i, (demo_obs, policy_obs) in enumerate(
                        zip(
                            behavior_spec.observation_shapes,
                            expected_behavior_spec.observation_shapes,
                        )
                ):
                    if demo_obs != policy_obs:
                        raise RuntimeError(
                            f"The shape {demo_obs} for observation {i} in demonstration \
                            do not match the policy's {policy_obs}."
                        )
        return behavior_spec, demo_buffer

    @timed
    def make_demo_buffer(self,
                         pair_infos: List[AgentInfoActionPairProto],
                         behavior_spec: BehaviorSpec,
                         sequence_length: int,
                         ) -> AgentBuffer:
        # Create and populate buffer using experiences
        demo_raw_buffer = AgentBuffer()
        demo_processed_buffer = AgentBuffer()
        for idx, current_pair_info in enumerate(pair_infos):
            if idx > len(pair_infos) - 2:
                break
            next_pair_info = pair_infos[idx + 1]
            current_decision_step, current_terminal_step = steps_from_proto(
                [current_pair_info.agent_info], behavior_spec
            )
            next_decision_step, next_terminal_step = steps_from_proto(
                [next_pair_info.agent_info], behavior_spec
            )
            previous_action = (
                    np.array(
                        pair_infos[idx].action_info.vector_actions_deprecated, dtype=np.float32
                    )
                    * 0
            )
            if idx > 0:
                previous_action = np.array(
                    pair_infos[idx - 1].action_info.vector_actions_deprecated,
                    dtype=np.float32,
                )

            next_done = len(next_terminal_step) == 1

            if len(next_terminal_step) == 1:
                next_reward = next_terminal_step.reward[0]
            else:
                next_reward = next_decision_step.reward[0]

            if len(current_terminal_step) == 1:
                current_obs = list(current_terminal_step.values())[0].obs
            else:
                current_obs = list(current_decision_step.values())[0].obs

            demo_raw_buffer["done"].append(next_done)
            demo_raw_buffer["rewards"].append(next_reward)
            for i, obs in enumerate(current_obs):
                demo_raw_buffer[ObsUtil.get_name_at(i)].append(obs)
            if (
                    len(current_pair_info.action_info.continuous_actions) == 0
                    and len(current_pair_info.action_info.discrete_actions) == 0
            ):
                if behavior_spec.action_spec.continuous_size > 0:
                    demo_raw_buffer["continuous_action"].append(
                        current_pair_info.action_info.vector_actions_deprecated
                    )
                else:
                    demo_raw_buffer["discrete_action"].append(
                        current_pair_info.action_info.vector_actions_deprecated
                    )
            else:
                if behavior_spec.action_spec.continuous_size > 0:
                    demo_raw_buffer["continuous_action"].append(
                        current_pair_info.action_info.continuous_actions
                    )
                if behavior_spec.action_spec.discrete_size > 0:
                    demo_raw_buffer["discrete_action"].append(
                        current_pair_info.action_info.discrete_actions
                    )
            demo_raw_buffer["prev_action"].append(previous_action)
            if next_done:
                demo_raw_buffer.resequence_and_append(
                    demo_processed_buffer, batch_size=None, training_length=sequence_length
                )
                demo_raw_buffer.reset_agent()
        demo_raw_buffer.resequence_and_append(
            demo_processed_buffer, batch_size=None, training_length=sequence_length
        )
        return demo_processed_buffer

    def divide_into_chunks(self, a, chunk_size):
        for i in range(0,len(a), chunk_size):
            yield a[i:i+chunk_size, ...]

    def visualize_sequence(self, obs_sequence):
        concat_sequence = np.concatenate(obs_sequence, axis=-1).transpose(2,0,1)
        for img in concat_sequence:
            if not np.max(img):
                continue
            plt.imshow(img, cmap='gray')
            plt.show()

    def split_sequences(self, observations, done):
        obs = []
        tmp = []
        for o, d in zip(observations, done):
            tmp.append(o)
            if d:
                obs.append(tmp)
                tmp = []
        return obs

