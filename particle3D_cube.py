import numpy as np
import matplotlib.pyplot as plt

# =====================
# USER PARAMETERS
# =====================

# Cube detector (~100 km³)
cube_size = 4640  # meters
half_size = cube_size / 2
spacing = 400     # afstand tussen lijnen

# Particle
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 0

# Movement
dx = 800
dy = 1000
dz = 1500


# =====================
# FUNCTION 1: CUBE DETECTOR (MET LIJNEN)
# =====================
def build_cube_detector(ax, half_size, spacing, cube_size):

    x_vals = np.arange(-half_size, half_size + spacing, spacing)
    y_vals = np.arange(-half_size, half_size + spacing, spacing)
    z_vals = np.arange(-half_size, half_size + spacing, spacing)

    detector_count = 0
    line_count = 0

    # lijnen op elk (x,y)
    for x in x_vals:
        for y in y_vals:

            # detectoren langs z
            ax.scatter(
                np.full_like(z_vals, x),
                np.full_like(z_vals, y),
                z_vals,
                color='blue',
                s=10
            )

            # verticale lijn
            ax.plot(
                [x]*len(z_vals),
                [y]*len(z_vals),
                z_vals,
                color='black',
                linewidth=1
            )

            detector_count += len(z_vals)
            line_count += 1

    detectors_per_line = len(z_vals)

    return detector_count, line_count, detectors_per_line


# =====================
# FUNCTION 2: PARTICLE
# =====================
def simulate_particle(ax, r, theta, z, dx, dy, dz):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # start
    ax.scatter(x, y, z, color='orange', s=80, label='Start')

    # nieuwe positie
    x_new = x + dx
    y_new = y + dy
    z_new = z + dz

    # eind
    ax.scatter(x_new, y_new, z_new, color='red', s=80, label='End')

    # vector
    ax.quiver(x, y, z, dx, dy, dz, color='orange', normalize=False)

    # afstand
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    return distance


# =====================
# MAIN SCRIPT
# =====================
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# build cube detector
detector_count, line_count, detectors_per_line = build_cube_detector(
    ax,
    half_size,
    spacing,
    cube_size
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
total_line_length_m = line_count * cube_size
total_line_length_km = total_line_length_m / 1000

# labels
ax.set_title("3D Cube Detector (~100 km³) with Vertical Lines")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

ax.set_box_aspect([1,1,1])
ax.legend()

plt.show()

# =====================
# OUTPUT
# =====================
print("\n--- DETECTOR INFO ---")
print(f"Total number of detectors: {detector_count}")
print(f"Total number of lines: {line_count}")
print(f"Detectors per line: {detectors_per_line}")
print(f"Total cable length: {total_line_length_km:.2f} km")

print("\n--- PARTICLE INFO ---")
print(f"Distance traveled: {distance:.2f} meters")