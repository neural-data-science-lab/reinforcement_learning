import os

import imageio
from PIL import Image
from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np

max_frame = 5

width = 480
height = 480
video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

domain_name = "point_mass"
task_name = "easy"
env_name = f"dm_control_{domain_name}_{task_name}"

# Load one task:
env = suite.load(domain_name=domain_name, task_name=task_name)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

exp_path = os.path.join("results", env_name, "random_agent")
os.makedirs(exp_path, exist_ok=True)
frames = []
while not time_step.last():
    for i in range(max_frame):
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        time_step = env.step(action)

        video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                              env.physics.render(height, width, camera_id=1)])

    for i in range(max_frame):
        frames.append(video[i])

imageio.mimsave(os.path.join(exp_path, "dm_control_example.gif"), frames, duration=20)
