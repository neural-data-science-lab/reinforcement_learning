import gymnasium as gym

from sb3_contrib.common.wrappers.skill_observation import SkillObservationWrapper

env = gym.make("LunarLander-v2", render_mode="human")
env_wrapped = SkillObservationWrapper(env)

env_wrapped.reset(seed=42)

for _ in range(1000):
    action = env_wrapped.action_space.sample()  # this is where you would insert your policy

    observation, reward, terminated, truncated, info = env_wrapped.step(action)
    print(observation)

    if terminated or truncated:
        env_wrapped.reset()

env_wrapped.close()
