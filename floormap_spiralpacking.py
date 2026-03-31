import numpy as np
import matplotlib.pyplot as plt

# how many points in total
detectors = 1000
# determines how far apart the points are (controls the density of the spiral)
c = 20  

# golden angle for optimal packing
golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.5°

x_points = []
y_points = []

# iteration to generate points in a sunflower spiral pattern
for n in range(detectors):
    theta = n * golden_angle
    r = c * np.sqrt(n)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    x_points.append(x)
    y_points.append(y)

# plot
plt.figure(figsize=(7,7))
plt.scatter(x_points, y_points, s=10, color='blue')

plt.axis('equal')
plt.title("Sunflower Spiral Packing")

plt.show()
print(f"Number of detectors: {detectors}")