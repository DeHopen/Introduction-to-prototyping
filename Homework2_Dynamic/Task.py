import numpy as np
import matplotlib.pyplot as plt
import csv
import xml.etree.ElementTree as ET


def parse_robot_config(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    robot_config = {
        "links": [],
        "joints": []
    }
    for body in root.findall('worldbody/body'):
        for geom in body.findall('geom'):
            robot_config["links"].append({
                "name": geom.get('type'),
                "size": geom.get('size'),
                "material": geom.get('material')
            })
        for joint in body.findall('joint'):
            robot_config["joints"].append({
                "type": joint.get('type'),
                "axis": joint.get('axis')
            })
    return robot_config


xml_file = 'Task.xml'
robot_config = parse_robot_config(xml_file)

amplitude = 0.04  # 4 cm
max_velocity = 0.4  # m/s
dt = 0.01  # time step
steps_per_segment = int(amplitude / (max_velocity * dt))


x_trajectory = np.concatenate([np.linspace(0, amplitude, steps_per_segment), np.zeros(steps_per_segment)])
y_trajectory = np.concatenate([np.zeros(steps_per_segment), np.linspace(0, amplitude, steps_per_segment)])
steps = len(x_trajectory)

data = []

for t in range(steps):
    time = t * dt
    x_pos = x_trajectory[t]
    y_pos = y_trajectory[t]

    if t > 0:
        x_vel = (x_trajectory[t] - x_trajectory[t - 1]) / dt
        y_vel = (y_trajectory[t] - y_trajectory[t - 1]) / dt
        x_acc = (x_vel - ((x_trajectory[t - 1] - x_trajectory[t - 2]) / dt)) / dt
        y_acc = (y_vel - ((y_trajectory[t - 1] - y_trajectory[t - 2]) / dt)) / dt
    else:
        x_vel = y_vel = x_acc = y_acc = 0

    torque = np.sqrt(x_acc ** 2 + y_acc ** 2)

    data.append([time, x_pos, y_pos, x_vel, y_vel, x_acc, y_acc, torque])

csv_file = '../robot_data.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Time", "X Position", "Y Position", "X Velocity", "Y Velocity", "X Acceleration", "Y Acceleration", "Torque"])
    writer.writerows(data)

print(f"Data saved to {csv_file}")

data = np.array(data)
time = data[:, 0]
x_pos = data[:, 1]
y_pos = data[:, 2]
x_vel = data[:, 3]
y_vel = data[:, 4]
x_acc = data[:, 5]
y_acc = data[:, 6]
torque = data[:, 7]

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

axs[0].plot(time, x_pos, label='X Position')
axs[0].plot(time, y_pos, label='Y Position', linestyle='--')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time, x_vel, label='X Velocity')
axs[1].plot(time, y_vel, label='Y Velocity', linestyle='--')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(time, x_acc, label='X Acceleration')
axs[2].plot(time, y_acc, label='Y Acceleration', linestyle='--')
axs[2].set_ylabel('Acceleration (m/s^2)')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(time, torque, label='Torque', color='m')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Torque (Nm)')
axs[3].legend()
axs[3].grid(True)

plt.suptitle('Robot Cross Movement Trajectory')
plt.show()
