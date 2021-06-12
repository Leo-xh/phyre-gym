from gym.envs.registration import register
from .envs import PhyreEnv
register(
    id='phyre-v0',
    entry_point='phyre_gym.envs:PhyreEnv',
)