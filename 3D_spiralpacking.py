import numpy as np
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
detectors_per_layer = 300
c = 20
golden_angle = np.pi * (3 - np.sqrt(5))

z_layers = np.arange(0, 1001, 200)

# --------------------
# SETUP FIGURE
# --------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

detector_count = 0

# --------------------
# SPIRAL PER LAAG
# --------------------
for z in z_layers:
    for n in range(detectors_per_layer):
        theta = n * golden_angle
        r = c * np.sqrt(n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ax.scatter(x, y, z, color='purple', s=10)
        detector_count += 1


# --------------------
# PLOT SETTINGS
# --------------------
ax.set_title("3D Spiral Packing (Layered)")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.set_box_aspect([1,1,1])

plt.show()

print(f"Number of detectors: {detector_count}")