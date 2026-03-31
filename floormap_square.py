# This will be the floormap generator for the KM3NeT
# Acoustic Neutrino Detection experiment square version


import numpy as np
import matplotlib.pyplot as plt

# parameters
# distances between the points
afstand = 200  
# starting point of the grid
range_min = -1000
# ending point of the grid
range_max = 1000

# creates points in grid
x = np.arange(range_min, range_max + afstand, afstand)
y = np.arange(range_min, range_max + afstand, afstand)

# creates grid
X, Y = np.meshgrid(x, y)

# Plot
plt.figure(figsize=(7,7))
plt.scatter(X, Y, color='blue', s=20)

# Labels and title
plt.title("Floorplan Square Detector")
plt.xlabel("X-plane [m]")
plt.ylabel("Y-plane [m]")

# equals axis distances
plt.axis('equal')

# Raster and boundaries
plt.xlim(range_min - 200, range_max + 200)
plt.ylim(range_min - 200, range_max + 200)

plt.show()


