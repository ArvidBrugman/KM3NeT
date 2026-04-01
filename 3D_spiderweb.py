import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# SETTINGS
# --------------------------
# maximal radius of the sphere
r_max = 1000
# number of layers (concentric spheres)
n_rings = 15
# number of points per layer (evenly distributed)
n_theta = 20
# number of points in vertical direction (evenly distributed)
n_phi = 8

# --------------------------
# WAARDES MAKEN
# --------------------------
# makes nicely distributed values
radii = np.linspace(0, r_max, n_rings)
theta_vals = np.linspace(0, 2*np.pi, n_theta)
phi_vals = np.linspace(0, np.pi, n_phi)

# --------------------------
# POINTS GENERATOR
# --------------------------
# storage for points
x_points = []
y_points = []
z_points = []

# loops to generate points in a spherical grid pattern
for r in radii:
    for phi in phi_vals:
        for theta in theta_vals:
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)

# --------------------------
# STARTING PLOT
# --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Every point is blue
ax.scatter(x_points, y_points, z_points, s=5, color='blue')

# --------------------------
# 1. RADIAL SPOKES (centre → outwards)
# --------------------------
# take each direction and plot a line from the center to the outer layer (spokes)
# fix the properties and move on
for phi in phi_vals:
    for theta in theta_vals:
        
        x_line = []
        y_line = []
        z_line = []
        
        for r in radii:
            x_line.append(r * np.sin(phi) * np.cos(theta))
            y_line.append(r * np.sin(phi) * np.sin(theta))
            z_line.append(r * np.cos(phi))
        
        ax.plot(x_line, y_line, z_line, color='gray')

# --------------------------
# 2. RINGS (around the sphere)
# --------------------------
for r in radii:
    for phi in phi_vals:
        
        x_ring = []
        y_ring = []
        z_ring = []
        
        for theta in theta_vals:
            x_ring.append(r * np.sin(phi) * np.cos(theta))
            y_ring.append(r * np.sin(phi) * np.sin(theta))
            z_ring.append(r * np.cos(phi))
        
        ax.plot(x_ring, y_ring, z_ring, color='lightgray')

# --------------------------
# 3. VERTICAL LINES (up ↔ down)
# --------------------------
for r in radii:
    for theta in theta_vals:
        
        x_line = []
        y_line = []
        z_line = []
        
        for phi in phi_vals:
            x_line.append(r * np.sin(phi) * np.cos(theta))
            y_line.append(r * np.sin(phi) * np.sin(theta))
            z_line.append(r * np.cos(phi))
        
        ax.plot(x_line, y_line, z_line, color='lightgray')

# --------------------------
# LABELS
# --------------------------
ax.set_title("3D Spiderweb")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

plt.show()

# --------------------------
# INFO
# --------------------------
print("Aantal punten:", len(x_points))