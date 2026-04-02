import numpy as np
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
R = 12600          # radius [m]
H = 2000           # height [m]

c = 500            # horizontal spacing (~meters)
z_step = 500       # vertical spacing (~meters)

k = 15             # number of spiral groups
spacing = 6        # every Nth spiral is a channel
channel_width = 1  # how many spirals form a channel

golden_angle = np.pi * (3 - np.sqrt(5))

# number of base points (lines before filtering)
detectors_xy = int((R / c)**2)

# vertical detector positions
z_layers = np.arange(0, H + 1, z_step)

# --------------------
# GENERATE BASE (XY)
# --------------------
x_base = []
y_base = []

for n in range(detectors_xy):
    
    spiral_id = n % k
    
    # remove spirals → create channels
    if spiral_id % spacing < channel_width:
        continue
    
    theta = n * golden_angle
    r = c * np.sqrt(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x_base.append(x)
    y_base.append(y)

# --------------------
# SETUP FIGURE
# --------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

detector_count = 0
total_cable_length = 0

# --------------------
# BUILD VERTICAL LINES
# --------------------
for x, y in zip(x_base, y_base):
    
    # each line contributes full height
    total_cable_length += H
    
    for z in z_layers:
        ax.scatter(x, y, z, color='blue', s=4)
        detector_count += 1

    # draw the cable/string
    ax.plot([x, x], [y, y], [0, H], color='black', linewidth=0.5)

# --------------------
# PLOT SETTINGS
# --------------------
ax.set_title("3D Detector Array with Vertical Strings and Channels")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_box_aspect([1,1, H/(2*R)])

plt.show()

# --------------------
# OUTPUT STATS
# --------------------
print("----- STATS -----")
print(f"Number of lines: {len(x_base)}")
print(f"Detectors per line: {len(z_layers)}")
print(f"Total detectors: {detector_count}")

print(f"Total cable length: {total_cable_length:.0f} m")
print(f"Total cable length: {total_cable_length/1000:.1f} km")