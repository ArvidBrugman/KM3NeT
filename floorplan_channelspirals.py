import numpy as np
import matplotlib.pyplot as plt

# === INPUT ===
# maximal radius of the whole system
R = 100
# spacing between spirals (controls the density of the spiral, smaller, more dense)
c = 5
# number of spirals
k = 20          
# for this amount of spiral, we make a channel (every 5th spiral is a channel) 
spacing = 5      
# how many spirals together form a channel
channel_width = 1  

# golden angle formula for optimal packing
golden_angle = np.pi * (3 - np.sqrt(5))

# surface area (R) divded by the density (c), gives number of detectors
detectors = int((R / c)**2)

x_points = []
y_points = []

for n in range(detectors):
    
    # devides the points into k spirals, and determines which spiral this point belongs to
    spiral_id = n % k
    
    # decides if this point belongs to a channel or not, if it belongs to a channel, we skip it
    if spiral_id % spacing < channel_width:
        # skip this point, it belongs to a channel
        continue  
    
    theta = n * golden_angle
    r = c * np.sqrt(n)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    x_points.append(x)
    y_points.append(y)

plt.figure(figsize=(7,7))
plt.scatter(x_points, y_points, s=10, color='blue')

plt.axis('equal')
plt.title("Spiral Packing with Controlled Channels")

plt.show()