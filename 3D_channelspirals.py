import numpy as np
import matplotlib.pyplot as plt

# --------------------
# PARAMETERS
# --------------------
R = 12600
H = 2000

c = 500
z_step = 500

k = 15
spacing = 6
channel_width = 1

golden_angle = np.pi * (3 - np.sqrt(5))

detectors_xy = int((R / c)**2)
z_layers = np.arange(0, H + 1, z_step)

# --------------------
# GENERATE BASE (XY)
# --------------------
x_base = []
y_base = []

for n in range(detectors_xy):
    
    spiral_id = n % k
    
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

# --------------------
# BUILD VERTICAL LINES
# --------------------
for x, y in zip(x_base, y_base):
    
    for z in z_layers:
        ax.scatter(x, y, z, color='blue', s=4)
        detector_count += 1

    # optioneel: teken de lijn zelf
    ax.plot([x, x], [y, y], [0, H], color='black', linewidth=0.5)


# --------------------
# PLOT SETTINGS
# --------------------
ax.set_title("Vertical Detector Lines (Ocean Floor Anchored)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_box_aspect([1,1, H/(2*R)])

plt.show()

print(f"Total detectors: {detector_count}")
print(f"Number of lines: {len(x_base)}")
print(f"Detectors per line: {len(z_layers)}")