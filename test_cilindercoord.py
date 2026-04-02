import numpy as np
import matplotlib.pyplot as plt

# --------------------
# FUNCTIONS
# --------------------
def cylindrical_to_cartesian(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

def update_particle(r, theta, z, dr, dtheta, dz):
    r_new = r + dr
    theta_new = theta + dtheta
    z_new = z + dz
    return r_new, theta_new, z_new

# --------------------
# PARAMETERS
# --------------------
distance_between_points = 100
radii = np.arange(distance_between_points, 1001, distance_between_points)
z_layers = np.arange(0, 1001, 100)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

detector_count = 0

# --------------------
# DETECTOR CYLINDER
# --------------------
for z in z_layers:
    ax.scatter(0, 0, z, color='blue', s=15)
    detector_count += 1

    for r in radii:
        points = int(2 * np.pi * r / distance_between_points)
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z_array = np.full_like(x, z)

        ax.scatter(x, y, z_array, color='blue', s=15)
        detector_count += points

# --------------------
# INITIAL PARTICLE (cilindrisch)
# --------------------
r_particle = 300
theta_particle = np.pi / 4
z_particle = 400

# --------------------
# MOVEMENT (cilindrisch!)
# --------------------
dr = -50        # naar binnen
dtheta = 0.3    # draaien
dz = 150        # omhoog

# nieuwe positie berekenen
r_new, theta_new, z_new = update_particle(
    r_particle, theta_particle, z_particle,
    dr, dtheta, dz
)

# --------------------
# OUDE POSITIE (plotten)
# --------------------
x_old, y_old, z_old = cylindrical_to_cartesian(
    r_particle, theta_particle, z_particle
)

ax.scatter(x_old, y_old, z_old, color='orange', s=80, label='Start')

# --------------------
# NIEUWE POSITIE (plotten)
# --------------------
x_new, y_new, z_new = cylindrical_to_cartesian(
    r_new, theta_new, z_new
)

ax.scatter(x_new, y_new, z_new, color='red', s=80, label='End')

# --------------------
# LIJN (trajectory segment)
# --------------------
ax.plot(
    [x_old, x_new],
    [y_old, y_new],
    [z_old, z_new],
    color='orange'
)

# --------------------
# VECTOR PIJL (echte verplaatsing)
# --------------------
dx = x_new - x_old
dy = y_new - y_old
dz_vec = z_new - z_old

ax.quiver(
    x_old, y_old, z_old,
    dx, dy, dz_vec,
    color='orange',
    normalize=False
)

# --------------------
# LABELS
# --------------------
ax.set_title("Particle Motion in Cylindrical Coordinates with Vector")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.legend()

plt.show()

print(f"Number of detectors: {detector_count}")