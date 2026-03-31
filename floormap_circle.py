# This will be the floormap generator for the KM3NeT
# Acoustic Neutrino Detection experiment circle-version

import numpy as np
import matplotlib.pyplot as plt

# decides distance between the rings and the points 
distance_between_points = 50  
radii = np.arange(distance_between_points, 1001, distance_between_points)


plt.figure(figsize=(7,7))

detector_count = 0

plt.scatter(0, 0, color='blue', s=15)
detector_count += 1

for r in radii:
    # number of points based on the circumference of the circle 
    points = int(2 * np.pi * r / distance_between_points)
    
    theta = np.linspace(0, 2*np.pi, points, endpoint=False)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    plt.scatter(x, y, color='blue', s=15)

    detector_count += points

# center point

plt.axis('equal')
plt.title("Realistic Sensor Rings")

# Labels and title
plt.title("Floorplan circle/ring Detector")
plt.xlabel("X-plane [m]")
plt.ylabel("Y-plane [m]")

# equals axis distances
plt.axis('equal')

plt.show()

print(f"Number of detectors: {detector_count}")


