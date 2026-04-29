import numpy as np
import matplotlib.pyplot as plt
from ACpulse import ACpulse
import time
import os

# =====================
# USER PARAMETERS
# =====================

# decides how dense the detector grid is
distance_between_points = 1000
max_radius = 3300
max_height = 3000
z_step = 1000

# Particle position (cylindrical coordinates)
r_particle = 1500
theta_particle = np.pi / 4
z_particle = 1000

# Energy shower
Edep = 1e20

# Detector position for single pulse analysis (cartesian coordinates)
detector_position = np.array([0, 0, 1000])

# =====================
# NEUTRINO DIRECTION
# =====================

# this together gives the start of the pancake shape
# thats namely perpendicular to the direction of the neutrino
# angle w.r.t. z-axis
theta_dir = 0
# rotation angle in xy-plane (fill in with e.g. np.pi / 4)
phi_dir = np.pi / 2

# Calibration files
CALIB_NPY = "calibration.npy"
CALIB_TXT = "calibration.txt"


# =====================
# BUILD DETECTOR POSITIONS
# =====================
def build_detector_positions(distance_between_points, max_radius, max_height, z_step):
    # makes rings of detectors at different radii 
    radii = np.arange(distance_between_points, max_radius, distance_between_points)
    # makes layers of detectors at different heights
    z_layers = np.arange(0, max_height + 1, z_step)

    positions = []

    for r in radii:
        # decides how many detectors on every ring/circle
        # = circumference / distance between points
        points = int(2 * np.pi * r / distance_between_points)
        # distributes the detectors evenly on the circle
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)

        # coordinate transformation from cylindrical to cartesian
        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)

            for z in z_layers:
                # makes a 3D grid of detector points
                positions.append([x, y, z])

    return np.array(positions)


# =====================
# DIRECTION VECTOR
# =====================
def get_direction_vector(theta, phi):
    # spherical to cartesian coordinate transformation
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    # theta = 0 gives arrow pointing downwards
    vz = -np.cos(theta)

    v = np.array([vx, vy, vz])
    # normalize the vector to have length 1
    return v / np.linalg.norm(v)


# =====================
# CALIBRATION (3 POINTS)
# =====================
def run_calibration(Edep):

    print("Running calibration (3 ACpulse calls)...")

    pulse = ACpulse()

    # these are the 3 points we will use for calibration in (r,z) coordinates
    calibration_points = [
        (1000, 0),
        (2000, 0),
        (1000, 500)
    ]

    calibration_data = []
    # R = distance from shower axis in xy-plane
    # Z = distance along the shower axis (z-axis)
    for i, (R, Z) in enumerate(calibration_points):
        print(f"Calibration {i+1}/3 → R={R}, Z={Z}")

        # determines the amplitude of the signal    
        pulse.hydrophonePosition([R, Z])
        pulse.shower_energy(Edep)
        _, signal = pulse.getSignal()
        amp = np.max(signal)

        calibration_data.append((R, Z, amp))

    calibration_data = np.array(calibration_data)

    # save binary (fast)
    np.save(CALIB_NPY, calibration_data)

    # save readable .txt
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
# checks if calibration file exists, if not runs the calibration
def get_calibration(Edep):

    if os.path.exists(CALIB_NPY):
        print("Loading calibration from .npy file...")
        return np.load(CALIB_NPY)

    else:
        return run_calibration(Edep)


# =====================
# INTERPOLATION
# =====================
# guessing strategy to guess the amplitude at any point (R,Z) based on the 3 calibration points
# e.g. for 1m --> amplitude = 10
# for 2m --> amplitude = 5
# for 1m, 500m --> amplitude = 7.5
def interpolate_amplitude(R, Z, calibration_data):

    # calculates the distance from the point (R,Z) to each of the calibration points
    distances = np.sqrt(
        (calibration_data[:,0] - R)**2 +
        (calibration_data[:,1] - Z)**2
    )

    # finds the index of the closest calibration point
    # which calibration seems to like mostly the same as my own point
    idx = np.argmin(distances)

    # take that amplitude
    R_ref, Z_ref, A_ref = calibration_data[idx]

    # scale the amplitude based on the distance from the reference point
    # e.g. 2x more distance --> 0.5x amplitude
    scale = (R_ref + 1) / (R + 1)

    return A_ref * scale


# =====================
# SIGNAL COMPUTATION
# =====================
def compute_signals(detector_positions, source_pos, calibration_data, direction):

    signals = []
    # calculates the signal for every detector
    for pos in detector_positions:

        # vector from source to detector
        r_vec = pos - source_pos
        # distance from source to detector
        dist = np.linalg.norm(r_vec)

        # special case for the detector at the source position (to avoid division by zero)
        if dist == 0:
            amp = calibration_data[0,2]
        else:
            # distance to shower axis (radial)
            R = np.sqrt(r_vec[0]**2 + r_vec[1]**2)

            # distance along the shower axis 
            Z = np.dot(r_vec, direction)

            # Z-cut: skip regions where signal is negligible
            if Z < -10 or Z > 50:
                amp = 0
            else:
                # what will ACpulse give us at that point based on the calibration data
                amp = interpolate_amplitude(R, Z, calibration_data)

                # distance damping as a consequence of the medium (sea water)
                amp *= np.exp(-dist / 2000)

                # projects detector on shower plane to get the distance along the pancake shape
                d = np.dot(r_vec, direction)
                pancake_width = 300

                # gaussian shape, inside the shape is strong signal, outside the shape is weak signal
                amp *= np.exp(-(d / pancake_width)**2)

        signals.append(amp)

    return np.array(signals)

# =====================
# SINGLE PULSE AT POSITION
# =====================
def compute_pulse_at_position(detector_pos, source_pos, direction, Edep):
    """
    Compute the full acoustic pulse (time vs amplitude) at a single detector position.
    """
    # gives pulse
    pulse = ACpulse()

    # vector from source to detector
    r_vec = detector_pos - source_pos

    # distance along the shower axis 
    Z = np.dot(r_vec, direction)

    # apply same Z-cut as in signal model
    if Z < -10 or Z > 50:
        print("Chosen detector is outside shower region → no signal")
        return None, None

    # radial distance to shower axis
    R = np.sqrt(r_vec[0]**2 + r_vec[1]**2)

    # set detector position in ACpulse coordinates
    pulse.hydrophonePosition([R, Z])
    # decides how strong the pulse is
    pulse.shower_energy(Edep)

    # gets the signal/ waveform at that position
    time_array, signal = pulse.getSignal()

    return time_array, signal

# =====================
# MAIN
# =====================
def main():
    # stamples the runtime of the whole program to get an idea of the performance
    start_time = time.time()

    # makes 3D plot of the detector positions and the signal strength at each position
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # builds the detector geometry
    detector_positions = build_detector_positions(
        distance_between_points,
        max_radius,
        max_height,
        z_step
    )

    # converts neutrino position from cylindrical to cartesian coordinates
    x0 = r_particle * np.cos(theta_particle)
    y0 = r_particle * np.sin(theta_particle)
    z0 = z_particle

    # source of the signal (neutrino interaction point)
    source_pos = np.array([x0, y0, z0])

    # decides the direction of the shower and how the pancake will be oriented in space
    direction = get_direction_vector(theta_dir, phi_dir)

    # calibration data if its there
    calibration_data = get_calibration(Edep)

    # calculate for every detector position what the signal strength will be based on the distance to the shower and the orientation of the shower
    print("Computing signals...")
    signals = compute_signals(detector_positions, source_pos, calibration_data, direction)

    # convert to mPa for better visualization
    signals_mPa = signals * 1000

    # ordens the data for plotting
    x = detector_positions[:, 0]
    y = detector_positions[:, 1]
    z = detector_positions[:, 2]

    # makes a scatter plot of the detector positions, colored by the signal strength
    sc = ax.scatter(
    x, y, z,
    c=signals_mPa,
    cmap='plasma',
    s=40, vmin=0, vmax=np.percentile(signals_mPa, 95))


    # makes a colorbar to show the scale of the signal strength
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signal Amplitude [mPa]")

    # plots the position of the neutrino
    ax.scatter(x0, y0, z0, color='red', s=120, label='Neutrino')

    # plot selected detector position (for pulse analysis)
    ax.scatter(detector_position[0], detector_position[1],
    detector_position[2], color='green', 
    edgecolors='black',
    s=120, label='Selected detector')

    # Plot shower direction as a vector (arrow)
    # length of the arrow (in meters)
    arrow_length = 2000  

    ax.quiver(
        x0, y0, z0,                        # startingpoint (neutrino)
        direction[0], direction[1], direction[2],  # direction
        length=arrow_length,
        color='cyan',
        linewidth=2,
        normalize=True, label='Shower direction')

    # shows the shower axis with a dashed line
    t = np.linspace(-2000, 2000, 100)
    x_line = x0 + t * direction[0]
    y_line = y0 + t * direction[1]
    z_line = z0 + t * direction[2]
    ax.plot(x_line, y_line, z_line, color='cyan', linestyle='--', label='Shower axis')


    # =====================
    # TEST: SINGLE PULSE AT USER-DEFINED POSITION
    # =====================

    print("\nComputing pulse at test position:", detector_position)

    time_array, signal = compute_pulse_at_position(
        detector_position,
        source_pos,
        direction,
        Edep
    )
    # only plot if inside shower region
    if time_array is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        time_array = time_array * 1e6
        signal = signal * 1e3

        ax2.plot(time_array, signal)
        ax2.set_xlim(-1000, 1000)
        ax2.set_xlabel("Time [µs]")
        ax2.set_ylabel("Amplitude [mPa]")
        ax2.set_title("Acoustic Pulse at Selected Detector Position")
        ax2.grid()

    
    # labels and title
    ax.set_title("Directional Acoustic Detector Response")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.legend(
    loc='upper left',
    # 0.90 is perfect, 0.95 is more to the right than 0.90
    bbox_to_anchor=(0.90, 1.17),
    borderaxespad=0.)

    plt.subplots_adjust(right=0.85, top=0.9)
    plt.show()


    # shows the runtime of the program to get an idea of the performance
    total_time = time.time() - start_time

    print("\n--- PERFORMANCE ---")
    print(f"Runtime: {total_time:.2f} seconds")


# =====================
# RUN
# =====================
if __name__ == "__main__":
    main()