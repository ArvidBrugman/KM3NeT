import numpy as np
import matplotlib.pyplot as plt

# =====================
# USER PARAMETERS
# =====================

# Detector (spiral)
detectors_per_layer = 300
golden_angle = np.pi * (3 - np.sqrt(5))

max_height = 3000
z_step = 500

# schaal fix → ~100 km³
max_radius = 3200  # meters
c = max_radius / np.sqrt(detectors_per_layer)

# Particle
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 1000

# Movement (cartesian)
dx = 800 
dy = 1000
dz = 1500


# =====================
# FUNCTION 1: SPIRAL DETECTOR (MET LIJNEN)
# =====================
def build_spiral_detector(ax, detectors_per_layer, c, golden_angle, max_height, z_step):
    z_layers = np.arange(0, max_height + 1, z_step)

    detector_count = 0
    line_count = detectors_per_layer

    x_lines = []
    y_lines = []

    # basisposities (spiral)
    for n in range(detectors_per_layer):
        theta = n * golden_angle
        r = c * np.sqrt(n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        x_lines.append(x)
        y_lines.append(y)

    # lijnen omhoog trekken
    for i in range(detectors_per_layer):
        x = x_lines[i]
        y = y_lines[i]

        z_vals = z_layers

        # detectoren
        ax.scatter(
            np.full_like(z_vals, x),
            np.full_like(z_vals, y),
            z_vals,
            color='purple',
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

    detectors_per_line = len(z_layers)

    return detector_count, line_count, detectors_per_line


# =====================
# FUNCTION 2: PARTICLE
# =====================
def simulate_particle(ax, r, theta, z, dx, dy, dz):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # startpunt
    ax.scatter(x, y, z, color='orange', s=80, label='Start')

    # nieuwe positie
    x_new = x + dx
    y_new = y + dy
    z_new = z + dz

    # eindpunt
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

# build detector
detector_count, line_count, detectors_per_line = build_spiral_detector(
    ax,
    detectors_per_layer,
    c,
    golden_angle,
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
ax.set_title("3D Spiral Detector (~100 km³) with Vertical Lines")
ax.set_xlabel("X-plane [m]")
ax.set_ylabel("Y-plane [m]")
ax.set_zlabel("Z-plane [m]")

# 🔥 belangrijk voor juiste verhoudingen
ax.set_box_aspect([1,1,1])

ax.legend()

plt.show()

# sanity check radius
print(f"Max radius reached: {c*np.sqrt(detectors_per_layer):.2f} m")

print("\n--- DETECTOR INFO ---")
print(f"Total number of detectors: {detector_count}")
print(f"Total number of lines: {line_count}")
print(f"Detectors per line: {detectors_per_line}")
print(f"Total cable length: {total_line_length_km:.2f} km")

print("\n--- PARTICLE INFO ---")
print(f"Distance traveled: {distance:.2f} meters")