# This will be the floormap generator for the KM3NeT
# Acoustic Neutrino Detection experiment circle-version

import numpy as np
import matplotlib.pyplot as plt

radii = np.arange(100, 1001, 100)
afstand_tussen_punten = 100  # dit bepaal jij

plt.figure(figsize=(7,7))

for r in radii:
    # number of points based on the circumference of the circle 
    points = int(2 * np.pi * r / afstand_tussen_punten)
    
    theta = np.linspace(0, 2*np.pi, points, endpoint=False)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    plt.scatter(x, y, color='blue', s=15)

# center point
plt.scatter(0, 0, color='blue', s=50)
plt.axis('equal')
plt.title("Realistic Sensor Rings")

# Labels and title
plt.title("Floorplan circle/ring Detector")
plt.xlabel("X-plane [m]")
plt.ylabel("Y-plane [m]")

# equals axis distances
plt.axis('equal')

plt.show()


