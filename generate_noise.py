import numpy as np

print("Generating Gaussian noise...")

# load pulse grid 
data = np.load("pulses.npz")

# use these data to obtain a structure blueprint
R_vals = data["R"]
Z_vals = data["Z"]
t_vals = data["t"]

# determines the shape, shape = (R, Z, t)
shape = data["signal"].shape

# =====================
# NOISE PARAMETERS
# =====================
# mhu/average = 0
# spread sigma = 10 mPa = 0.01 Pa
sigma = 0.01  

# =====================
# GENERATE NOISE
# =====================
# generates random noise
noise = np.random.normal(
    loc=0.0,
    scale=sigma,
    size=shape
)

# =====================
# SAVE
# =====================
# saves the noise in  noise.npz file
np.savez(
    "noise.npz",
    R=R_vals,
    Z=Z_vals,
    t=t_vals,
    noise=noise
)

print("Saved noise.npz")