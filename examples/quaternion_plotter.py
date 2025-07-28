import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def quaternion_to_matrix(q):
    # q = [qx, qy, qz, qw]
    qx, qy, qz, qw = q
    # normalize quaternion
    norm = np.linalg.norm(q)
    if norm == 0:
        qw = 1.0
        qx = qy = qz = 0.0
        norm = 1.0
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    # compute rotation matrix elements
    xx = qx*qx; yy = qy*qy; zz = qz*qz
    xy = qx*qy; xz = qx*qz; yz = qy*qz
    wx = qw*qx; wy = qw*qy; wz = qw*qz
    R = np.array([
        [1-2*(yy+zz),   2*(xy - wz),   2*(xz + wy)],
        [  2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)],
        [  2*(xz - wy),   2*(yz + wx), 1-2*(xx+yy)]
    ])
    return R

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

axis_colors = ['r', 'g', 'b']

# initial quaternion components
init_qx, init_qy, init_qz, init_qw = 0.0, 0.0, 0.0, 1.0

# set up figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.35)

def update(val):
    qx = s_qx.val
    qy = s_qy.val
    qz = s_qz.val
    qw = s_qw.val
    ax.cla()
    # build rotation matrix from quaternion
    R = quaternion_to_matrix([qx, qy, qz, qw])
    rotated_axes = R @ np.eye(3)
    rotated_box = box @ R.T
    # plot axes
    for i in range(3):
        vec = rotated_axes[:, i]
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=1.0,
                  color=axis_colors[i], linewidth=2)
    # plot cube
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

# sliders
ax_qx = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_qy = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_qz = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_qw = plt.axes([0.25, 0.10, 0.65, 0.03])

s_qx = Slider(ax_qx, 'qx', -1.0, 1.0, valinit=init_qx)
s_qy = Slider(ax_qy, 'qy', -1.0, 1.0, valinit=init_qy)
s_qz = Slider(ax_qz, 'qz', -1.0, 1.0, valinit=init_qz)
s_qw = Slider(ax_qw, 'qw', -1.0, 1.0, valinit=init_qw)

s_qx.on_changed(update)
s_qy.on_changed(update)
s_qz.on_changed(update)
s_qw.on_changed(update)

# initial draw
update(None)
plt.show()
