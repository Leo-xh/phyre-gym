from pprint import pprint
from typing import Dict, Tuple

import gym
import numpy as np
import phyre
from gym import spaces
from phyre import FeaturizedObjects
from phyre.action_simulator import SimulationStatus
from phyre.metrics import Evaluator
from phyre.simulation import Simulation
from tqdm import tqdm


class PhyreEnv(gym.Env):
    """
    Vectorized feature Phyre Environment.
    The execution schema is
        env.seed(X)
        env.train()
        1. reset
            a. random select a task
            b. output the initial vectorized feature
        2. step
            a. simulate the task with the given action, record the vectorized feature trajectory and the result
            b. random select a task
            c. output the next initial vectorized feature, the result, whether the tasks are traversed, info: {"valid":the valid flag, "trajectory": the recorded trajectory}

            **users shuold judge whether the valid flag is True**
            **(1-done) is necessary since the final obs is randomly generated**
            **the observation shape (T,N,14)**

        env.test()

        :param render: return rgb arrays of the current task
    """

    metadata = {"render.modes": ["rgb_array"]}
    status_2_reward = {
        SimulationStatus.INVALID_INPUT: -5,
        SimulationStatus.NOT_SOLVED: -1,
        SimulationStatus.SOLVED: 0,
        SimulationStatus.UNSTABLY_SOLVED: 1,
        SimulationStatus.STABLY_SOLVED: 2,
    }

    def __init__(self, eval_setup: str = "ball_cross_template"):
        self.eval_setup = eval_setup
        self.action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        self._seed = 0
        self.train = True
        self.test = False

        self.reset()

        print(
            "PhyreEnv:",
            {
                "eval_setup": self.eval_setup,
                "action_tier": self.action_tier,
                "train_task": len(self.train_tasks),
                "test_task": len(self.test_tasks),
            },
        )
        self.action_space = spaces.Box(
            np.array([0] * self.train_simulator.action_space_dim),
            np.array([1] * self.train_simulator.action_space_dim),
            dtype=np.float,
        )

        # TODO: dict decision
        # self.observation_space = spaces.Dict(
        #     {
        #         "state": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float),
        #         "diameter": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float),
        #         "shape": spaces.Discrete(4),
        #         "color": spaces.Discrete(6),
        #     }
        # )
        self.observation_space = spaces.Box(
            np.array([0] * 14), np.array([1] * 14), dtype=np.float
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, bool, Dict]:

        assert (self.train_i < len(self.train_tasks) and self.train) or (
            self.test_i < len(self.test_tasks) and self.test
        ), "the env need to be reset"

        # DONE: stable

        if self.train:
            self.simulation: Simulation = self.train_simulator.simulate_action(
                self.train_task_indices[self.train_i],
                action,
                need_images=False,
                need_featurized_objects=True,
                stable=True,
            )
            self.train_i += 1
            if self.train_i >= len(self.train_tasks):
                done = True
                initial_feature = np.random.randn(3, self.observation_space.shape[0])
            else:
                done = False
                initial_feature = self.train_simulator.initial_featurized_objects[
                    self.train_task_indices[self.train_i]
                ].features.squeeze()
                # initial_feature = self.train_simulator.initial_featurized_objects[
                #     self.train_task_indices[self.train_i]
                # ]
        if self.test:
            self.simulation: Simulation = self.test_simulator.simulate_action(
                self.test_task_indices[self.test_i],
                action=action,
                need_images=True,
                need_featurized_objects=True,
                stable=True,
            )
            self.test_i += 1
            if self.test_i >= len(self.test_tasks):
                done = True
                initial_feature = np.random.randn(3, self.observation_space.shape[0])
            else:
                done = False
                initial_feature = self.test_simulator.initial_featurized_objects[
                    self.test_task_indices[self.test_i]
                ].features.squeeze()
                # initial_feature = self.test_simulator.initial_featurized_objects[
                #     self.test_task_indices[self.test_i]
                # ]

        if self.simulation.status.is_invalid():
            return (
                initial_feature,
                self.status_2_reward[self.simulation.status],
                done,
                {"valid": False, "trajectory": np.zeros((1, 1, 14))},
            )
        else:
            return (
                initial_feature,
                self.status_2_reward[self.simulation.status],
                done,
                {
                    "valid": True,
                    "trajectory": self.simulation.featurized_objects.features,
                },
            )

    def reset(self) -> FeaturizedObjects:
        self._seed = np.random.randint(10)
        self.train_tasks, dev_tasks, self.test_tasks = phyre.get_fold(
            self.eval_setup, self._seed
        )
        self.train_tasks += dev_tasks
        self.train_simulator = phyre.initialize_simulator(
            self.train_tasks, self.action_tier
        )
        self.test_simulator = phyre.initialize_simulator(
            self.test_tasks, self.action_tier
        )

        self.train_task_indices = np.random.permutation(
            np.arange(len(self.train_tasks))
        )
        self.test_task_indices = np.random.permutation(np.arange(len(self.test_tasks)))

        self.train_i = 0
        self.test_i = 0

        # simulation object
        self.simulation = None

        if self.train:
            return self.train_simulator.initial_featurized_objects[
                self.train_task_indices[self.train_i]
            ].features.squeeze()
            # return self.train_simulator.initial_featurized_objects[
            #     self.train_task_indices[self.train_i]
            # ]
        else:
            return self.test_simulator.initial_featurized_objects[
                self.test_task_indices[self.test_i]
            ].features.squeeze()
            # return self.test_simulator.initial_featurized_objects[
            #     self.test_task_indices[self.test_i]
            # ]

    def render(self, mode: str = "rgb_array"):
        if mode not in self.metadata["render.modes"]:
            raise (NotImplementedError, "Please use 'rgb_array' mode")
        assert self.simulation is not None, "the Env doesn't begin"
        return self.simulation.images

    def seed(self, seed: int = None):
        self._seed = seed

    def test(self):
        self.test = True
        self.train = False

    def train(self):
        self.test = False
        self.train = True

    def sample(self) -> np.ndarray:
        return self.train_simulator.sample()

    def evaluate(self, agent, save_gif=False, save_path="") -> Evaluator:
        # TODO: save gifs
        assert hasattr(agent, "predict"), "agent should have attribute 'predict'"
        evaluator = phyre.Evaluator(self.test_tasks)
        total_attempt = 0
        test_iter = tqdm(self.test_task_indices, desc="Evaluate tasks")
        for task_id in test_iter:
            # while evaluator.get_attempts_for_task(task_id) < phyre.MAX_TEST_ATTEMPTS:
            attempts = 0
            while (
                evaluator.get_attempts_for_task(task_id) < 10
                and attempts < phyre.MAX_TEST_ATTEMPTS
            ):
                initial_feature_obj = self.test_simulator.initial_featurized_objects[
                    task_id
                ]
                action = agent.predict(initial_feature_obj.features.squeeze())
                # print(action)
                status = self.test_simulator.simulate_action(
                    task_id, action, need_images=False
                ).status
                evaluator.maybe_log_attempt(task_id, status)
                total_attempt += 1
                attempts += 1
                test_iter.set_description(desc=f"{total_attempt} attempts")
        return evaluator


if __name__ == "__main__":
    # DONE: invalid sample
    env = PhyreEnv()
    env.reset()
    for _ in tqdm(range(5000)):
        action = env.sample()
        # print(action)
        _, r, done, _ = env.step(action)
        # print(r)
        if done:
            env.reset()

    class RandomAgent(object):
        def __init__(self, action_space):
            self.action_space = action_space

        def predict(self, state: np.ndarray):
            # def predict(self, state: FeaturizedObjects):
            return self.action_space.sample()

    r_agent = RandomAgent(env.action_space)
    evaluator = env.evaluate(r_agent)
    print(evaluator.get_auccess())
