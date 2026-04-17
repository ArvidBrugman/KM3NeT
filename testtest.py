import numpy as np

data = np.load("pulses.npz")

R = data["R"]
Z = data["Z"]
t = data["t"]
signal = data["signal"]

# print structuur
print("=== DATA STRUCTUUR ===")
print("R:", R)
print("Z:", Z)
print("t (eerste 10):", t[:10])
print()

# voorbeeld van enkele pulses
print("=== VOORBEELD SIGNALEN ===")

for i in range(2):        # eerste 2 R waarden
    for j in range(2):    # eerste 2 Z waarden
        print(f"R = {R[i]}, Z = {Z[j]}")
        print("signal (eerste 10 samples):")
        print(signal[i, j, :10])
        print()