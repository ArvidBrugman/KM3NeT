# This will be the floormap generator for the KM3NeT
# Acoustic Neutrino Detection experiment square-version


import numpy as np
import matplotlib.pyplot as plt

# parameters
# how big the area is (max radius)
r_max = 1000
# number of directions from the center
n_spokes = 20
# number of layers  
n_rings = 12

plt.figure(figsize=(7,7))

# spread angle of the spokes
angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)

# higher density of points near the center, lower density further out
radii = np.linspace(0, np.sqrt(r_max), n_rings)**2

# storage for points
x_points = []
y_points = []

# crossing points between spokes and rings
for r in radii:
    for theta in angles:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        x_points.append(x)
        y_points.append(y)

# plot points
plt.scatter(x_points, y_points, s=10, color='blue')

# gives polygonal effect
for r in radii:
    x_ring = r * np.cos(angles)
    y_ring = r * np.sin(angles)
    
    # closes the ring structure by adding the first point at the end
    x_ring = np.append(x_ring, x_ring[0])
    y_ring = np.append(y_ring, y_ring[0])
    
    plt.plot(x_ring, y_ring, color='lightgray', linewidth=0.5)

# 🕸️ teken spaken
for theta in angles:
    # lines from the middle to the outside
    x_line = [0, r_max * np.cos(theta)]
    y_line = [0, r_max * np.sin(theta)]
    
    plt.plot(x_line, y_line, color='lightgray', linewidth=0.5)

plt.axis('equal')
plt.title("Angular Spiderweb Layout")
plt.xlabel("X-plane [m]")
plt.ylabel("Y-plane [m]")

plt.show()

