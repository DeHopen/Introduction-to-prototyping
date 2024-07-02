import time
import mujoco
import numpy as np
import pandas as pd
from itertools import product


# Function to convert degrees to radians
def deg2rad(degrees):
    return np.deg2rad(degrees)


# Joint limits and step size
joint_limits = np.arange(-90, 91, 15)  # -90 to 90 degrees in 15-degree increments

# Prepare the CSV file
results = []

# Load the model
model = mujoco.MjModel.from_xml_path('Task1.xml')
data = mujoco.MjData(model)

# Get all possible joint angle configurations
all_configs = list(product(joint_limits, repeat=model.njnt))

# Iterate over all configurations
for config in all_configs:
    # Set joint angles
    data.qpos[:] = [deg2rad(angle) for angle in config]

    # Compute inverse dynamics
    mujoco.mj_inverse(model, data)

    # Save the configuration and resulting torques
    results.append({
        'joint1_angle': config[0],
        'joint2_angle': config[1],
        'joint3_angle': config[2],
        'torque1': data.qfrc_inverse[0],
        'torque2': data.qfrc_inverse[1],
        'torque3': data.qfrc_inverse[2],
    })

# Convert results to a DataFrame and save to a CSV file
df = pd.DataFrame(results)
df.to_csv('inverse_dynamics_results.csv', index=False)

print('Results saved to inverse_dynamics_results.csv')
