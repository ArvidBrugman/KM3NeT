import numpy as np

print("Generating Gaussian noise...")

# load pulse grid (we gebruiken alleen shape + axes)
data = np.load("pulses.npz")

R_vals = data["R"]
Z_vals = data["Z"]
t_vals = data["t"]

# shape = (R, Z, t)
shape = data["signal"].shape

# =====================
# NOISE PARAMETERS
# =====================
sigma = 0.01  # 10 mPa = 0.01 Pa

# =====================
# GENERATE NOISE
# =====================
noise = np.random.normal(
    loc=0.0,
    scale=sigma,
    size=shape
)

# =====================
# SAVE
# =====================
np.savez(
    "noise.npz",
    R=R_vals,
    Z=Z_vals,
    t=t_vals,
    noise=noise
)

print("Saved noise.npz")