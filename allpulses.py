import numpy as np
import matplotlib.pyplot as plt
from ACpulse import ACpulse

# =====================
# 1. DETECTOR
# =====================
def build_detector(distance_between_points, max_radius, max_height, z_step):
    """
    Build a 3D detector consisting of vertical strings arranged in concentric rings.

    - Rings at different radii (cylindrical geometry)
    - Each ring has hydrophones evenly spaced in phi
    - Each (x,y) position forms a vertical line in z
    """

    # radial positions of detector strings
    radii = np.arange(distance_between_points, max_radius, distance_between_points)

    # vertical layers along each string
    z_layers = np.arange(0, max_height + 1, z_step)

    positions = []

    for r in radii:
        # number of hydrophone strings on this ring
        n_points = int(2 * np.pi * r / distance_between_points)

        # angular positions around the circle
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

        for t in theta:
            # cylindrical → cartesian
            x = r * np.cos(t)
            y = r * np.sin(t)

            # build vertical line of hydrophones
            for z in z_layers:
                positions.append([x, y, z])

    return np.array(positions)


# =====================
# 2. NEUTRINO
# =====================
def create_neutrino(r, theta, z, theta_dir, phi_dir, energy):
    """
    Create a neutrino interaction (shower source).

    Returns a dictionary containing:
    - position (cartesian)
    - direction (unit vector)
    - energy
    """

    # position: cylindrical → cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pos = np.array([x, y, z])

    # direction vector (spherical → cartesian)
    vx = np.sin(theta_dir) * np.cos(phi_dir)
    vy = np.sin(theta_dir) * np.sin(phi_dir)
    vz = -np.cos(theta_dir)  # downward convention

    direction = np.array([vx, vy, vz])
    direction /= np.linalg.norm(direction)  # normalize

    # store everything as one physical event
    neutrino = {
        "position": pos,
        "direction": direction,
        "energy": energy
    }

    return neutrino


# =====================
# 3. ROTATION MATRIX
# =====================
def get_rotation_matrix(direction):
    """
    Compute rotation matrix that aligns the shower direction with the z-axis.

    This allows us to evaluate the acoustic model in the shower frame.
    """

    # desired axis (ACpulse assumes shower along z-axis)
    z_axis = np.array([0, 0, -1])

    # rotation axis and angle
    v = np.cross(direction, z_axis)
    c = np.dot(direction, z_axis)

    # already aligned → no rotation
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)

    # skew-symmetric matrix for Rodrigues rotation
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rodrigues' rotation formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v)**2))

    return R


# =====================
# 4. SIGNAL COMPUTATION
# =====================
def compute_all_signals(detector_positions, neutrino, rotation_matrix):
    """
    Compute full acoustic waveforms for all hydrophones.

    For each detector:
    - transform position into shower frame
    - compute (R, Z)
    - call ACpulse 

    Returns a list of detector data dictionaries:
    each containing R, Z, time and signal
    """

    pulse = ACpulse()

    source_pos = neutrino["position"]
    Edep = neutrino["energy"]

    detector_data = []

    for i, pos in enumerate(detector_positions):
        print(f"Detector {i+1}/{len(detector_positions)}")

        # vector from neutrino interaction to detector
        r_vec = pos - source_pos

        # rotate into shower coordinate system
        r_rot = rotation_matrix @ r_vec

        # cylindrical coordinates relative to shower axis
        R = np.sqrt(r_rot[0]**2 + r_rot[1]**2)  # radial distance
        Z = r_rot[2]                            # along shower axis

        # compute acoustic pulse at this position
        pulse.hydrophonePosition([R, Z])
        pulse.shower_energy(Edep)

        t, signal = pulse.getSignal()

        # store full information per hydrofoon
        detector_data.append({
            "position": pos,   # (x,y,z)
            "R": R,
            "Z": Z,
            "time": t,
            "signal": signal
        })

    return detector_data


# =====================
# 5. MAIN
# =====================
def main():

    # detector geometry parameters
    distance_between_points = 1000
    max_radius = 3300
    max_height = 3000
    z_step = 1000

    # neutrino energy
    Edep = 1e20

    # neutrino position (cylindrical)
    r_particle = 1500
    theta_particle = np.pi / 3
    z_particle = 1000

    # neutrino direction
    theta_dir = 0
    phi_dir = np.pi / 2

    # build detector array
    detector_positions = build_detector(
        distance_between_points,
        max_radius,
        max_height,
        z_step
    )

    # create neutrino event
    neutrino = create_neutrino(
        r_particle,
        theta_particle,
        z_particle,
        theta_dir,
        phi_dir,
        Edep
    )

    source_pos = neutrino["position"]
    direction = neutrino["direction"]

    # rotation to shower frame
    rotation_matrix = get_rotation_matrix(direction)

    # compute signals (full dataset)
    detector_data = compute_all_signals(
        detector_positions,
        neutrino,
        rotation_matrix
    )

    # =====================
    # VISUALIZATION (example)
    # =====================
    # plot a single detector pulse as example
    example = detector_data[0]
    np.save("detector_data.npy", detector_data)

    plt.plot(example["time"], example["signal"])
    plt.title(f"Example pulse (R={example['R']:.1f}, Z={example['Z']:.1f})")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.show()


if __name__ == "__main__":
    main()