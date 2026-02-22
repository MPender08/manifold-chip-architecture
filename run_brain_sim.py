import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

# Import our updated architecture and the new Diagnostic Hub
from dynamic_brain import DynamicCurvatureNet, VIP_MacroController

# --- SEED CONTROL ---
USE_LOCKED_SEED = True
LOCKED_SEED = 137

if USE_LOCKED_SEED:
    current_seed = LOCKED_SEED
else:
    current_seed = random.randint(1000, 99999)

print("\n" + "="*60)
print(f"SEED: {current_seed}")
print("="*60 + "\n")

torch.manual_seed(current_seed)

# --- Simulation Logic ---
x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_data = torch.tensor([[0.], [1.], [1.], [0.]])

def run_stamina_test(label, tax_rate, use_vip=False):
    print(f"Running Stamina Test: {label} (VIP Override: {use_vip})")
    
    model = DynamicCurvatureNet(input_dim=2, hidden_dim=2, output_dim=1)
    
    # Bias the brain to start slightly Hyperbolic
    nn.init.constant_(model.sst_neuron.sense[2].bias, 2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    mse = nn.MSELoss()
    
    history = {"gamma": [], "error": [], "vip": []}
    
    # Initialize the Global Diagnostic Hub if this run uses it
    vip_hub = VIP_MacroController(alpha=0.8, epsilon=0.15, delta=0.005, tau_vip=30.0) if use_vip else None
    current_vip_signal = 0.0
    
    for epoch in range(1500):
        optimizer.zero_grad()
        
        # Pass the global VIP signal into the forward pass
        pred, gamma_net = model(x_data, global_gamma_vip=current_vip_signal)
        
        # 1. Task Loss (Predator Pressure)
        task_loss = mse(pred, y_data)
        
        # 2. Metabolic Tax (Starvation Pressure)
        tax_loss = tax_rate * (gamma_net ** 2)
        
        # Total Loss
        total_loss = (task_loss * 20.0) + tax_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Update the VIP Controller with the new task loss
        if use_vip:
            current_vip_signal = vip_hub.step(task_loss)
        
        if epoch % 10 == 0:
            history["gamma"].append(gamma_net.item())
            history["error"].append(task_loss.item())
            history["vip"].append(current_vip_signal)
            
    return history

# --- Run Experiments ---
# We run a standard healthy model, and a healthy model with the VIP override active
healthy_standard = run_stamina_test("Healthy (Local Only)", tax_rate=0.001, use_vip=False)
healthy_vip = run_stamina_test("Healthy (VIP Attention)", tax_rate=0.001, use_vip=True)

# --- Advanced Visualization ---
plt.figure(figsize=(18, 6))
epochs_x = np.arange(0, 1500, 10)

# Plot A: Cognitive Performance (Error)
plt.subplot(1, 3, 1)
plt.plot(epochs_x, healthy_standard["error"], 'g--', lw=2, alpha=0.6, label="Local Only")
plt.plot(epochs_x, healthy_vip["error"], 'b', lw=3, label="VIP Override Active")
plt.axhline(0.25, color='k', linestyle=':', label="Random Guess Limit")
plt.ylim(-0.02, 0.52)
plt.title("Cognitive Performance (XOR Task)")
plt.xlabel("Epochs")
plt.ylabel("MSE Error")
plt.legend()
plt.grid(alpha=0.3)

# Plot B: Dendritic Curvature (Gamma)
plt.subplot(1, 3, 2)
plt.plot(epochs_x, healthy_standard["gamma"], 'g--', lw=2, alpha=0.6, label="Local Only")
plt.plot(epochs_x, healthy_vip["gamma"], 'b', lw=3, label="VIP Override Active")
plt.title("Network Curvature ($\gamma_{net}$)")
plt.xlabel("Epochs")
plt.ylabel("Geometric State (0=Flat, 1=Hyperbolic)")
plt.legend()
plt.grid(alpha=0.3)

# Plot C: The VIP Macro-Controller Signal
plt.subplot(1, 3, 3)
plt.fill_between(epochs_x, healthy_vip["vip"], color='orange', alpha=0.3)
plt.plot(epochs_x, healthy_vip["vip"], 'darkorange', lw=3, label="Global VIP Voltage")
plt.title("The Physics of Attention (Arousal Spike)")
plt.xlabel("Epochs")
plt.ylabel("Injected VIP Signal ($I_{VIP}$)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vip_attention_sim.png")
print("Saved visualization to 'vip_attention_sim.png'")
plt.show()