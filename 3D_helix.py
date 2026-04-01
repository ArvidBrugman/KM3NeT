import numpy as np
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
t_max = 10 * np.pi
points = 1000
r = 500
pitch = 50
bridge_step = 5   
# --------------------
# PARAMETER t
# --------------------
t = np.linspace(0, t_max, points)

# --------------------
# HELIX 1
# --------------------
x1 = r * np.cos(t)
y1 = r * np.sin(t)
z1 = pitch * t

# --------------------
# HELIX 2
# --------------------
x2 = r * np.cos(t + np.pi)
y2 = r * np.sin(t + np.pi)
z2 = z1

# --------------------
# PLOT
# --------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# alles blauw
ax.scatter(x1, y1, z1, color='blue', s=10)
ax.scatter(x2, y2, z2, color='blue', s=10)

# --------------------
# BRIDGES
# --------------------
for i in range(0, len(t), bridge_step):
    ax.plot(
        [x1[i], x2[i]],
        [y1[i], y2[i]],
        [z1[i], z2[i]],
        color='blue',
        linewidth=1
    )

# --------------------
# SETTINGS
# --------------------
ax.set_title("3D Double Helix (DNA Style)")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.set_box_aspect([1,1,1])

plt.show()

# --------------------
# COUNT
# --------------------
detector_count = len(x1) + len(x2)
print(f"Number of detectors (helix points only): {detector_count}")