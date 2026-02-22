import torch
import torch.nn as nn
import geoopt

# ==========================================
# 1. THE GLOBAL DIAGNOSTIC HUB (NEW)
# ==========================================
class VIP_MacroController:
    def __init__(self, alpha=0.5, epsilon=0.1, delta=0.01, tau_vip=10.0, ema_alpha=0.1):
        """
        Monitors macroscopic task error to trigger a global hyperbolic plunge.
        alpha: Global gain scalar (intensity of the panic response)
        epsilon: Acceptable error threshold (only inject VIP if error > epsilon)
        delta: Stagnation threshold (inject if dE/dt >= -delta)
        tau_vip: The decay time constant (attention span / physical RC leak rate)
        ema_alpha: Smoothing factor for the error signal (Low-Pass Filter for batch noise)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.ema_alpha = ema_alpha 
        
        self.decay_factor = torch.exp(torch.tensor(-1.0 / tau_vip))
        
        self.current_gamma_vip = torch.tensor(0.0)
        self.smoothed_error = None
        self.prev_error = None

    def step(self, current_error):
        err_val = current_error.item() if torch.is_tensor(current_error) else current_error
        
        # 1. LOW-PASS FILTER: Smooth the incoming error to ignore batch noise
        if self.smoothed_error is None:
            self.smoothed_error = err_val
        else:
            self.smoothed_error = (self.ema_alpha * err_val) + ((1.0 - self.ema_alpha) * self.smoothed_error)

        if self.prev_error is None:
            self.prev_error = self.smoothed_error
            return self.current_gamma_vip.item()

        # 2. Calculate dE/dt using the SMOOTHED error
        dE_dt = self.smoothed_error - self.prev_error
        
        # 3. Continuous Error Injection (I_VIP)
        stagnating = 1.0 if (dE_dt >= -self.delta) else 0.0
        error_excess = max(0.0, self.smoothed_error - self.epsilon)
        i_vip = self.alpha * error_excess * stagnating
        
        # 4. Metabolic Decay Function
        self.current_gamma_vip = (self.current_gamma_vip * self.decay_factor) + i_vip
        self.current_gamma_vip = torch.clamp(self.current_gamma_vip, 0.0, 1.0)
        
        # Update state for the next step using the smoothed error
        self.prev_error = self.smoothed_error
        
        return self.current_gamma_vip.item()

# ==========================================
# 2. THE LOCAL HARDWARE MODULES (ORIGINAL)
# ==========================================
class HyperbolicLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.weight = geoopt.ManifoldParameter(torch.randn(out_features, in_features) * 0.2)
        self.bias = geoopt.ManifoldParameter(torch.zeros(out_features))

    def forward(self, x, current_c):
        # Ensure c is positive and stable
        c = torch.clamp(current_c, min=1e-4, max=5.0)
        temp_manifold = geoopt.PoincareBall(c=c)
        
        # Hyperbolic Transform
        x_hyp = temp_manifold.expmap0(x)
        output_hyp = temp_manifold.mobius_matvec(self.weight, x_hyp)
        output_hyp = temp_manifold.mobius_add(output_hyp, self.bias)
        
        return temp_manifold.logmap0(output_hyp)

class SST_Gate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sense = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.sense(x)).mean()


# ==========================================
# 3. THE UNIFIED ARCHITECTURE (UPDATED)
# ==========================================
class DynamicCurvatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.sst_neuron = SST_Gate(input_dim)
        self.dendrite = HyperbolicLayer(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, global_gamma_vip=0.0):
        # 1. Bottom-up local calculation (SST Gate)
        local_gamma = self.sst_neuron(x)
        
        # 2. Unified Topological Logic (The Physics of Attention)
        # The macro-controller acts as a baseline "floor" that elevates the network's geometric sensitivity
        gamma_net = torch.clamp(local_gamma + global_gamma_vip, 0.0, 1.0)
        
        # Multiply by 5.0 to scale the normalized gamma (0-1) up to a severe hyperbolic 'c' value (0-5)
        # This matches the 'Hyperbolic Plunge' parameters from the Stamina test
        c_mapped = gamma_net * 5.0 
        
        # 3. Route through the dynamic manifold
        hidden = self.dendrite(x, c_mapped)
        out = self.classifier(hidden)
        
        # Return both the prediction and the combined gamma state for the metabolic tax calculation
        return out, gamma_net