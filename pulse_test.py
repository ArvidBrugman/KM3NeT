#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ACpulse import ACpulse
import time
import os

# =====================
# USER PARAMETERS
# =====================

distance_between_points = 1000
max_radius = 3300
max_height = 3000
z_step = 1000

# Particle positie
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 1000

# Energy
Edep = 1e20

# =====================
# NEUTRINO DIRECTION
# =====================

theta_dir = 0
phi_dir = np.pi / 3

# Calibration files
CALIB_NPY = "calibration.npy"
CALIB_TXT = "calibration.txt"


# =====================
# BUILD DETECTOR POSITIONS
# =====================
def build_detector_positions(distance_between_points, max_radius, max_height, z_step):
    radii = np.arange(distance_between_points, max_radius, distance_between_points)
    z_layers = np.arange(0, max_height + 1, z_step)

    positions = []

    for r in radii:
        points = int(2 * np.pi * r / distance_between_points)
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)

            for z in z_layers:
                positions.append([x, y, z])

    return np.array(positions)


# =====================
# DIRECTION VECTOR
# =====================
def get_direction_vector(theta, phi):
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)

    v = np.array([vx, vy, vz])
    return v / np.linalg.norm(v)


# =====================
# CALIBRATION (3 POINTS)
# =====================
def run_calibration(Edep):

    print("Running calibration (3 ACpulse calls)...")

    pulse = ACpulse()

    calibration_points = [
        (1000, 0),
        (2000, 0),
        (1000, 500)
    ]

    calibration_data = []

    for i, (R, Z) in enumerate(calibration_points):
        print(f"Calibration {i+1}/3 → R={R}, Z={Z}")

        pulse.hydrophonePosition([R, Z])
        pulse.shower_energy(Edep)

        _, signal = pulse.getSignal()
        amp = np.max(signal)

        calibration_data.append((R, Z, amp))

    calibration_data = np.array(calibration_data)

    # 💾 save binary (fast)
    np.save(CALIB_NPY, calibration_data)

    # 💾 save readable txt
    np.savetxt(
        CALIB_TXT,
        calibration_data,
        header="R(m)   Z(m)   Amplitude",
        fmt="%.6e"
    )

    print("Calibration saved as .npy and .txt!")

    return calibration_data


# =====================
# LOAD CALIBRATION
# =====================
def get_calibration(Edep):

    if os.path.exists(CALIB_NPY):
        print("Loading calibration from .npy file...")
        return np.load(CALIB_NPY)

    else:
        return run_calibration(Edep)


# =====================
# INTERPOLATION
# =====================
def interpolate_amplitude(R, Z, calibration_data):

    distances = np.sqrt(
        (calibration_data[:,0] - R)**2 +
        (calibration_data[:,1] - Z)**2
    )

    idx = np.argmin(distances)

    R_ref, Z_ref, A_ref = calibration_data[idx]

    scale = (R_ref + 1) / (R + 1)

    return A_ref * scale


# =====================
# SIGNAL COMPUTATION
# =====================
def compute_signals(detector_positions, source_pos, calibration_data, direction):

    signals = []

    for pos in detector_positions:

        r_vec = pos - source_pos
        dist = np.linalg.norm(r_vec)

        if dist == 0:
            amp = calibration_data[0,2]
        else:
            R = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            Z = r_vec[2]

            # interpolatie
            amp = interpolate_amplitude(R, Z, calibration_data)

            # afstandsafname
            amp *= np.exp(-dist / 2000)

            # pancake richting
            d = np.dot(r_vec, direction)
            pancake_width = 300

            amp *= np.exp(-(d / pancake_width)**2)

        signals.append(amp)

    return np.array(signals)


# =====================
# MAIN
# =====================
def main():

    start_time = time.time()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    detector_positions = build_detector_positions(
        distance_between_points,
        max_radius,
        max_height,
        z_step
    )

    # particle positie
    x0 = r_particle * np.cos(theta_particle)
    y0 = r_particle * np.sin(theta_particle)
    z0 = z_particle

    source_pos = np.array([x0, y0, z0])

    # richting
    direction = get_direction_vector(theta_dir, phi_dir)

    # calibratie
    calibration_data = get_calibration(Edep)

    # signalen berekenen
    print("Computing signals...")
    signals = compute_signals(detector_positions, source_pos, calibration_data, direction)

    # normaliseren
    if np.max(signals) > 0:
        signals = signals / np.max(signals)

    # plot
    x = detector_positions[:, 0]
    y = detector_positions[:, 1]
    z = detector_positions[:, 2]

    sc = ax.scatter(x, y, z, c=signals, cmap='viridis', s=40)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Normalized Signal Strength")

    ax.scatter(x0, y0, z0, color='red', s=120, label='Neutrino')

    ax.set_title("Directional Acoustic Detector Response")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.legend()
    plt.show()

    total_time = time.time() - start_time

    print("\n--- PERFORMANCE ---")
    print(f"Runtime: {total_time:.2f} seconds")


# =====================
# RUN
# =====================
if __name__ == "__main__":
    main()