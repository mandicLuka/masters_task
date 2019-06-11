import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='pomdp-v0',
    entry_point='search_env.envs:PomdpSearchEnv'
)
