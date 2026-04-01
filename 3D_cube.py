import numpy as np
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
afstand = 200
range_min = -1000
range_max = 1000

# --------------------
# GRID AXES
# --------------------
x = np.arange(range_min, range_max + afstand, afstand)
y = np.arange(range_min, range_max + afstand, afstand)
z = np.arange(range_min, range_max + afstand, afstand)

# --------------------
# 3D GRID
# --------------------
X, Y, Z = np.meshgrid(x, y, z)

# flatten voor plotting
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# --------------------
# PLOT
# --------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, color='blue', s=10)

ax.set_title("3D Cube Detector Grid")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.set_box_aspect([1,1,1])

plt.show()

# --------------------
# COUNT
# --------------------
detector_count = len(x) * len(y) * len(z)
print(f"Number of detectors: {detector_count}")