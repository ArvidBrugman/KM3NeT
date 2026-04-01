import numpy as np
import matplotlib.pyplot as plt

# decides distance between the rings and the points  
distance_between_points = 100
radii = np.arange(distance_between_points, 1001, distance_between_points)

# hight of the layers (e.g., every 200m)
z_layers = np.arange(0, 1001, 100)  

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# holds the count of number of detectors
detector_count = 0

for z in z_layers:
    # center point per laag
    ax.scatter(0, 0, z, color='blue', s=15)
    detector_count += 1

    for r in radii:
        points = int(2 * np.pi * r / distance_between_points)
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # makes an array of the same size as x and y, filled with the current z value (height)
        z_array = np.full_like(x, z)

        ax.scatter(x, y, z_array, color='blue', s=15)
        detector_count += points

# labels
ax.set_title("3D Cylinder Detector Layout")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

plt.show()

print(f"Number of detectors: {detector_count}")