import numpy as np
import matplotlib.pyplot as plt

# --------------------
# FUNCTIONS
# --------------------
def cylindrical_to_cartesian(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

# --------------------
# PARAMETERS (GESCHAALD!)
# --------------------
distance_between_points = 500   # 0.5 km spacing
radii = np.arange(distance_between_points, 3300, distance_between_points)  # tot 3.3 km radii
z_layers = np.arange(0, 3001, 500)  # tot 3 km hoogte

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

detector_count = 0

# --------------------
# DETECTOR CYLINDER (~100 km^3 orde)
# --------------------
for z in z_layers:
    ax.scatter(0, 0, z, color='blue', s=10)
    detector_count += 1

    for r in radii:
        points = int(2 * np.pi * r / distance_between_points)
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z_array = np.full_like(x, z)

        ax.scatter(x, y, z_array, color='blue', s=10)
        detector_count += points

# --------------------
# PARTICLE (startpositie)
# --------------------
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 1000

x_p, y_p, z_p = cylindrical_to_cartesian(r_particle, theta_particle, z_particle)

ax.scatter(x_p, y_p, z_p, color='orange', s=80, label='Start')

# --------------------
# VECTOR (cartesiaans!)
# --------------------
dx = 800
dy = 1000
dz = 1500

# nieuwe positie berekenen
x_new = x_p + dx
y_new = y_p + dy
z_new = z_p + dz





# vector tekenen
ax.quiver(
    x_p, y_p, z_p,
    dx, dy, dz,
    color='orange',
    normalize=False
)

# --------------------
# AFSTAND BEREKENEN
# --------------------
distance = np.sqrt(dx**2 + dy**2 + dz**2)
print(f"Distance traveled: {distance:.2f} meters")

# --------------------
# NIEUWE POSITIE (rood punt)
# --------------------
ax.scatter(x_new, y_new, z_new, color='red', s=80, label='End')

# --------------------
# TRAJECT LIJN
# --------------------
ax.plot(
    [x_p, x_new],
    [y_p, y_new],
    [z_p, z_new],
    color='orange'
)

# --------------------
# LABELS
# --------------------
ax.set_title("3D Detector (~100 km³) with Particle Track")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.legend()

plt.show()

print(f"Number of detectors: {detector_count}")