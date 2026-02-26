import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Setup Time and Input (40Hz Spikes)
t = np.linspace(0, 0.15, 5000) # 150 ms simulation
clock_freq = 40 
v_in = 5.0 * signal.square(2 * np.pi * clock_freq * t) # -5V to +5V

# 2. Simulate the Leaky Integrator (RC Circuit)
R = 1000  # 1k Ohm
C = 10e-6 # 10 uF
tau = R * C
v_cap = np.zeros_like(t)

# Numerical integration for the capacitor charge
dt = t[1] - t[0]
for i in range(1, len(t)):
    dv = (v_in[i-1] - v_cap[i-1]) / tau * dt
    v_cap[i] = v_cap[i-1] + dv

# 3. Inject Thermal Noise (kB T)
noise_amplitude = 0.4
v_noisy = v_cap + np.random.normal(0, noise_amplitude, len(t))

# 4. Simulate the Hysteretic Shield (Schmitt Trigger)
v_out = np.zeros_like(t)
current_state = 15.0  # Starts High (Gate Open, Manifold Flat)
threshold_high = 1.5  # Approximate trigger threshold
threshold_low = -1.5  # Approximate reset threshold

for i in range(len(t)):
    if v_noisy[i] >= threshold_high:
        current_state = -15.0 # Snaps Low (Gate Closed, Hyperbolic Plunge)
    elif v_noisy[i] <= threshold_low:
        current_state = 15.0  # Resets High
    v_out[i] = current_state

# 5. Plotting the Publication-Ready Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Top Plot: Leaky Integrator + Noise
ax1.plot(t, v_in, color='lightgray', linestyle='--', label='Incoming Somatic Spikes')
ax1.plot(t, v_noisy, color='mediumseagreen', alpha=0.8, label=r'Noisy Integration ($k_B T$)')
ax1.plot(t, v_cap, color='darkgreen', linewidth=2, label='Ideal Manifold Trajectory')
ax1.axhline(threshold_high, color='red', linestyle=':', label=r'Threshold ($\theta_{SST}$)')
ax1.set_ylabel('Local Voltage (V)', fontsize=12, fontweight='bold')
ax1.set_title('Dynamically Gated Analog Crossbar: Phase Transition Simulation', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom Plot: The Phase Transition (SST Gate)
ax2.plot(t, v_out, color='royalblue', linewidth=2.5, label='SST Shunt Gate State')
ax2.set_ylabel('Gate Voltage (V)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax2.set_ylim(-18, 18)
ax2.set_yticks([-15, 0, 15])
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("manifold_phase_transition.png", dpi=300)
plt.show()