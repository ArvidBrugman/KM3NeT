import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# PARAMETERS
# --------------------------
distance_between_points = 100
radii = np.arange(distance_between_points, 1001, distance_between_points)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

detector_count = 0

# --------------------------
# 1. CENTRUM
# --------------------------
ax.scatter(0, 0, 0, color='blue', s=20)
detector_count += 1

# --------------------------
# 2. SPHERES NAAR BUITEN
# --------------------------
for r in radii:
    
    # aantal detectors gebaseerd op oppervlak
    n_points = int(4 * np.pi * r**2 / distance_between_points**2)
    
    for i in range(n_points):
        
        # verdeel punten over bol (belangrijk!)
        phi = np.arccos(1 - 2*(i + 0.5)/n_points)
        theta = 2 * np.pi * i / n_points
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        ax.scatter(x, y, z, color='blue', s=5)
        detector_count += 1

# --------------------------
# LABELS
# --------------------------
ax.set_title("Expanding 3D Sphere Detector Layout")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

print(f"Number of detectors: {detector_count}")