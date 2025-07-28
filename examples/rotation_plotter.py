import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def rotation_matrix(roll, pitch, yaw):
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# Define a unit cube centered at the origin
box = np.array([[0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5]])
edges = [(0, 1), (0, 2), (0, 4),
         (7, 3), (7, 5), (7, 6),
         (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]

# Colors for axes: X=red, Y=green, Z=blue
axis_colors = ['r', 'g', 'b']

# Initial roll, pitch, yaw
init_roll, init_pitch, init_yaw = 0, 0, 0

# Set up the figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Define update function
def update(val):
    roll = s_roll.val
    pitch = s_pitch.val
    yaw = s_yaw.val
    ax.cla()
    R = rotation_matrix(roll, pitch, yaw)
    rotated_axes = R @ np.eye(3)
    rotated_box = box @ R.T

    # Plot rotated axes with colors
    for i in range(3):
        vec = rotated_axes[:, i]
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=1.0, color=axis_colors[i], linewidth=2)

    # Plot box edges
    for edge in edges:
        v0 = rotated_box[edge[0]]
        v1 = rotated_box[edge[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color='k')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X (front)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    fig.canvas.draw_idle()

# Create sliders with range -180 to +180
ax_roll = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_pitch = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_yaw = plt.axes([0.25, 0.05, 0.65, 0.03])

s_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=init_roll)
s_pitch = Slider(ax_pitch, 'Pitch', -180, 180, valinit=init_pitch)
s_yaw = Slider(ax_yaw, 'Yaw', -180, 180, valinit=init_yaw)

s_roll.on_changed(update)
s_pitch.on_changed(update)
s_yaw.on_changed(update)

# Initial plot
update(None)
plt.show()
