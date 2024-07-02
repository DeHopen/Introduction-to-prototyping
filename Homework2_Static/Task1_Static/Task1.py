import mujoco
import numpy as np
import pandas as pd
from itertools import product


def deg2rad(degrees):
    return np.deg2rad(degrees)


joint_limits = np.arange(-90, 91, 15)  # -90 to 90 degrees in 15-degree increments

results = []

model = mujoco.MjModel.from_xml_path('Task1.xml')
data = mujoco.MjData(model)

all_configs = list(product(joint_limits, repeat=model.njnt))


for config in all_configs:

    data.qpos[:] = [deg2rad(angle) for angle in config]

    mujoco.mj_inverse(model, data)

    results.append({
        'joint1_angle': config[0],
        'joint2_angle': config[1],
        'joint3_angle': config[2],
        'torque1': data.qfrc_inverse[0],
        'torque2': data.qfrc_inverse[1],
        'torque3': data.qfrc_inverse[2],
    })

df = pd.DataFrame(results)
df.to_csv('inverse_dynamics_results.csv', index=False)

print('Results saved to inverse_dynamics_results.csv')
