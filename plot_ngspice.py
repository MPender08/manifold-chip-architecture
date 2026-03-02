import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 0.1, 10000) # 100ms
dt = t[1] - t[0]

# 1. Somatic Input (40Hz square wave, +/- 5V)
freq = 40
# Pulse: -5 to 5, period 25ms
v_in = 5 * np.sign(np.sin(2 * np.pi * freq * t))

# 2. RC Integrator (R=1k, C=10uF -> tau=10ms)
tau = 0.010
v_rc = np.zeros_like(t)
for i in range(1, len(t)):
    dv = (v_in[i] - v_rc[i-1]) / tau * dt
    v_rc[i] = v_rc[i-1] + dv

# 3. Schmitt Trigger
v_out = np.zeros_like(t)
v_ref = np.zeros_like(t)

# Initial state
v_out[0] = 15.0 if v_rc[0] < 0 else -15.0
v_ref[0] = v_out[0] * 0.1 # 9k/1k voltage divider hysteresis

for i in range(1, len(t)):
    # Calculate difference against the reference threshold
    diff = v_ref[i-1] - v_rc[i]
    
    # Infinite gain approximation with Op-Amp rails
    if diff > 0:
        v_out[i] = 15.0
    else:
        v_out[i] = -15.0
        
    # Update reference voltage (the hysteresis)
    v_ref[i] = v_out[i] * 0.1

# Plotting the Data
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t * 1000, v_in, label='Somatic Input (40Hz)', color='blue')
plt.ylabel('Voltage (V)')
plt.title('NGSpice Transient Analysis: Manifold Chip Phase Transition')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(t * 1000, v_rc, label='RC Integration (Proxy Density)', color='orange')
plt.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Upper Threshold (+1.5V)')
plt.axhline(-1.5, color='red', linestyle='--', alpha=0.5, label='Lower Threshold (-1.5V)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.plot(t * 1000, v_out, label='Schmitt Trigger Output (SST Gate)', color='green')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('manifold_transient_analysis.png', dpi=300)