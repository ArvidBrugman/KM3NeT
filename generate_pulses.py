import numpy as np
from ACpulse import ACpulse
import multiprocessing

E = 1e20
Ncores = 10

print("Starting generation")

def worker(args):
    R, Z, Edep = args

    pulse = ACpulse()
    pulse.hydrophonePosition([R, Z])
    pulse.shower_energy(Edep)

    t, signal = pulse.getSignal()

    return R, Z, t, signal


if __name__ == "__main__":

    params = [(R, Z, E) for R in range(100, 5001, 500)
                           for Z in np.arange(-50, 51, 5)]
    

    print("Total number of simulations:", len(params))

    results = []
    time_axis = None

    with multiprocessing.Pool(Ncores) as pool:
        for R, Z, t, signal in pool.imap_unordered(worker, params):
            results.append((R, Z, signal))

            if time_axis is None:
                time_axis = t

    # sort for easier grid filling later (R,Z) -> signal
    results.sort()

    # unique R and Z values for grid construction
    R_vals = sorted(set(r for r, _, _ in results))
    Z_vals = sorted(set(z for _, z, _ in results))

    # 3D array creation: (R, Z, t)
    signal_array = np.zeros((len(R_vals), len(Z_vals), len(time_axis)))

    # fill the grid
    for R, Z, signal in results:
        i = R_vals.index(R)
        j = Z_vals.index(Z)
        signal_array[i, j, :] = signal

    with open("pulses_preview.txt", "w") as f:
        for idx, (R, Z, signal) in enumerate(results[:5]):  # eerste 5
            f.write(f"R={R}, Z={Z}\n")
            f.write(" ".join(map(str, signal[:20])) + "\n\n")  # eerste 20 samples

    # save to .npz for later use
    np.savez("pulses.npz",
             R=np.array(R_vals),
             Z=np.array(Z_vals),
             t=time_axis,
             signal=signal_array)

    print("Saved to pulses.npz")