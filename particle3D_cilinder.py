import numpy as np
import matplotlib.pyplot as plt

# =====================
# USER PARAMETERS
# =====================

# Detector
distance_between_points = 500
max_radius = 3300
max_height = 3000
z_step = 500

# Particle (cartesian via cylindrical startingpoint)
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 1000

# Movement (cartesian)
dx = 800 
dy = 1000
dz = 1500


# =====================
# FUNCTION 1: DETECTOR (MET LIJNEN)
# =====================
def build_detector(ax, distance_between_points, max_radius, max_height, z_step):
    radii = np.arange(distance_between_points, max_radius, distance_between_points)
    z_layers = np.arange(0, max_height + 1, z_step)

    detector_count = 0
    line_count = 0

    for r in radii:
        points = int(2 * np.pi * r / distance_between_points)
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        x_ring = r * np.cos(theta)
        y_ring = r * np.sin(theta)

        for i in range(len(x_ring)):
            x = x_ring[i]
            y = y_ring[i]

            # detectoren langs deze lijn
            z_vals = z_layers

            # plot detectoren
            ax.scatter(
                np.full_like(z_vals, x),
                np.full_like(z_vals, y),
                z_vals,
                color='blue',
                s=10
            )

            # plot lijn (string)
            ax.plot(
                [x]*len(z_vals),
                [y]*len(z_vals),
                z_vals,
                color='black',
                linewidth=1
            )

            detector_count += len(z_vals)
            line_count += 1

    detectors_per_line = len(z_layers)

    return detector_count, line_count, detectors_per_line


# =====================
# FUNCTION 2: PARTICLE
# =====================
def simulate_particle(ax, r, theta, z, dx, dy, dz):
    # cylindrical to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # startingpoint
    ax.scatter(x, y, z, color='orange', s=80, label='Start')

    # new position
    x_new = x + dx
    y_new = y + dy
    z_new = z + dz

    # endpoint
    ax.scatter(x_new, y_new, z_new, color='red', s=80, label='End')

    # vector
    ax.quiver(x, y, z, dx, dy, dz, color='orange', normalize=False)

    # distance
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    return distance


# =====================
# MAIN SCRIPT
# =====================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# build detector
detector_count, line_count, detectors_per_line = build_detector(
    ax,
    distance_between_points,
    max_radius,
    max_height,
    z_step
)

# simulate particle
distance = simulate_particle(
    ax,
    r_particle,
    theta_particle,
    z_particle,
    dx, dy, dz
)

# --------------------
# TOTAL LINE LENGTH
# --------------------
total_line_length_m = line_count * max_height
total_line_length_km = total_line_length_m / 1000

# labels
ax.set_title("3D Detector with Vertical Lines (Strings)")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.legend()

plt.show()

print("\n--- DETECTOR INFO ---")
print(f"Total number of detectors: {detector_count}")
print(f"Total number of lines: {line_count}")
print(f"Detectors per line: {detectors_per_line}")
print(f"Total cable length: {total_line_length_km:.2f} km")

print("\n--- PARTICLE INFO ---")
print(f"Distance traveled: {distance:.2f} meters")