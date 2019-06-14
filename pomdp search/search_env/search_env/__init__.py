from gym.envs.registration import register

register(
    id='multiagent_env-v0',
    entry_point='search_env.envs:MultiagentEnv',
)

