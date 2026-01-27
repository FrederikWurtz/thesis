import numpy as np
import torch
import math

class HapkeModel:
    def __init__(self, w=0.5, B0=0.5, h=0.1, phase_fun="hg", xi=0.2):
        """
        Initialize the HapkeModel with given parameters.

        Parameters:
        w (float): Single scattering albedo (0 < w < 1).
        B0 (float): Amplitude of the opposition effect.
        h (float): Angular width of the opposition surge.
        phase_fun (str): Phase function type ("hg" for Henyey-Greenstein).
        xi (float): Asymmetry parameter for the phase function.
        """
        self.w = w
        self.B0 = B0
        self.h = h
        self.phase_fun = phase_fun
        self.xi = xi

    def _H_function(self, mu):
        """
        Compute the Chandrasekhar H-function for multiple scattering.

        Parameters:
        mu (float or torch.Tensor): Cosine of incidence or emission angle.

        Returns:
        float or torch.Tensor: Value of the H-function.
        """
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        
        # Clamp mu to valid range [0, 1] for cosine
        mu_clamped = torch.clamp(mu, min=0, max=1)
        
        gamma = torch.sqrt(torch.clamp(torch.tensor(1.0 - self.w, dtype=torch.float32), min=1e-6))
        denominator = 1.0 + 2.0 * gamma * mu_clamped + 1e-12
        
        # Ensure denominator is positive
        denominator = torch.clamp(denominator, min=1e-6)
        
        return (1.0 + 2.0 * mu_clamped) / denominator

    def _B_SH(self, g_rad):
        """
        Compute the shadow-hiding opposition effect function.

        Parameters:
        g_rad (float or torch.Tensor): Phase angle in radians.

        Returns:
        float or torch.Tensor: Value of the opposition effect function.
        """
        if not isinstance(g_rad, torch.Tensor):
            g_rad = torch.tensor(g_rad, dtype=torch.float32)
        
        # Clamp g_rad to avoid tan(π/2) = inf
        # Phase angles typically range from 0 to π, but clamp to safe range
        g_clamped = torch.clamp(g_rad, min=0, max=torch.pi - 0.01)
        
        tan_term = torch.tan(0.5 * g_clamped)
        # Clamp tan_term to prevent extreme values
        tan_term = torch.clamp(tan_term, min=-1e6, max=1e6)
        
        denom = 1.0 + (1.0/self.h) * tan_term + 1e-12
        return self.B0 / denom

    def _P_phase(self, g_rad):
        """
        Compute the single-particle phase function.

        Parameters:
        g_rad (float or torch.Tensor): Phase angle in radians.

        Returns:
        float or torch.Tensor: Value of the phase function.
        """
        if not isinstance(g_rad, torch.Tensor):
            g_rad = torch.tensor(g_rad, dtype=torch.float32)
        if self.phase_fun == "hg":
            # Henyey-Greenstein phase function
            cg = torch.cos(g_rad)
            denominator = 1 + 2*self.xi*cg + self.xi**2
            
            # Ensure denominator stays positive (necessary for 1.5 power)
            denominator = torch.clamp(denominator, min=1e-6)
            
            result = (1 - self.xi**2) / torch.pow(denominator, 1.5)
            # Clamp result to reasonable values
            result = torch.clamp(result, min=0, max=1e6)
            return result
        else:
            # Isotropic scattering as default
            return torch.ones_like(g_rad)

    def radiance_factor(self, mu0, mu, g_rad):
        """
        Compute the radiance factor (I/F) using the Hapke model.

        Parameters:
        mu0 (float or torch.Tensor): Cosine of the incidence angle.
        mu (float or torch.Tensor): Cosine of the emission angle.
        g_rad (float or torch.Tensor): Phase angle in radians.

        Returns:
        float or torch.Tensor: Radiance factor (I/F).
        """
        if not isinstance(mu0, torch.Tensor):
            mu0 = torch.tensor(mu0, dtype=torch.float32)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(g_rad, torch.Tensor):
            g_rad = torch.tensor(g_rad, dtype=torch.float32)
            
        mu0c = torch.clamp(mu0, 0, 1)  # Ensure valid range
        muc = torch.clamp(mu, 0, 1)
        denom = (mu0c + muc) + 1e-12  # Avoid division by zero
        denom = torch.clamp(denom, min=1e-12)  # Extra safety
        
        P = self._P_phase(g_rad)      # Phase function
        B = self._B_SH(g_rad)         # Opposition effect
        H0 = self._H_function(mu0c)   # H-function for incidence
        H = self._H_function(muc)     # H-function for emission
        
        # Ensure no NaN/Inf in intermediate values
        P = torch.clamp(P, min=0, max=1e6)
        B = torch.clamp(B, min=0, max=1e6)
        H0 = torch.clamp(H0, min=0, max=1e6)
        H = torch.clamp(H, min=0, max=1e6)

        # Hapke reflectance equation
        r = (self.w / (4.0*torch.pi)) * (mu0c / denom) * ((1 + B) * P + H0*H - 1.0)
        R = torch.pi * r  # Convert to radiance factor
        
        # Clamp result to valid range
        R = torch.clamp(R, min=0, max=1e6)

        # Return R only where both mu0 and mu are positive, else 0
        # Use torch.maximum to clamp negative values to 0 (equivalent to np.maximum)
        return torch.where((mu0 > 0) & (mu > 0), torch.maximum(R, torch.tensor(0.0)), torch.tensor(0.0))

import torch
import math


# ============================================================
#  Roughness correction (Hapke 1984 / 1986 / 1993)
# ============================================================

def hapke_roughness(mu0, mu, g, theta_bar):
    """
    Hapke macroscopic roughness correction.

    Parameters
    ----------
    mu0 : torch.Tensor
        Cos(i)
    mu : torch.Tensor
        Cos(e)
    g : torch.Tensor
        Phase angle [rad]
    theta_bar : torch.Tensor or float
        Mean surface slope angle [rad]

    Returns
    -------
    mu0_eff, mu_eff, S
        Roughness-corrected cosines & shadowing term
    """
    theta_bar = torch.as_tensor(theta_bar, device=mu0.device, dtype=mu0.dtype)
    theta_bar = torch.clamp(theta_bar, 1e-6, math.radians(45.0))  # safe range

    t = torch.tan(theta_bar)
    sigma = t * torch.sqrt(torch.tensor(2.0 / math.pi, device=mu0.device))

    # Corrected incidence cosine
    mu0_eff = mu0 * (1.0 + (1.0/torch.cos(theta_bar) - 1.0) * torch.sqrt(1.0 - mu0**2))
    mu0_eff = torch.clamp(mu0_eff, min=0.0)

    # Corrected emission cosine
    mu_eff = mu * (1.0 + (1.0/torch.cos(theta_bar) - 1.0) * torch.sqrt(1.0 - mu**2))
    mu_eff = torch.clamp(mu_eff, min=0.0)

    # Shadowing probability (illumination × visibility)
    P0 = 1.0 - torch.exp(- (mu0 / sigma)**2)
    P  = 1.0 - torch.exp(- (mu / sigma)**2)

    S = P0 * P + 1e-6
    return mu0_eff, mu_eff, S



# ============================================================
#  Full Hapke Model Class
# ============================================================

class FullHapkeModel:
    """
    Full Hapke model for lunar-surface simulation.
    Supports:
      - spatial parameter maps
      - SHOE + CBOE
      - 1-term or 2-term HG
      - macroscopic roughness
    """

    def __init__(self,
                 # Albedo
                 w=0.12,

                 # Roughness
                 theta_bar=math.radians(20),

                 # SHOE
                 B0_sh=0.6,
                 h_sh=0.05,

                 # CBOE
                 B0_cb=0.2,
                 h_cb=0.02,

                 # Phase function
                 phase_fun="hg2",
                 xi=0.25,     # for hg1
                 b1=-0.3,     # for hg2
                 b2=0.2,      # for hg2
                 c=0.7        # for hg2
                 ):
        
        self.w = w
        self.theta_bar = theta_bar

        self.B0_sh = B0_sh
        self.h_sh = h_sh

        self.B0_cb = B0_cb
        self.h_cb = h_cb

        self.phase_fun = phase_fun.lower()
        self.xi = xi

        self.b1 = b1
        self.b2 = b2
        self.c = c

        self.eps = 1e-12


    # ------------------ helper ------------------

    def _to_tensor(self, val, like):
        if isinstance(val, torch.Tensor):
            return val.to(like.device, like.dtype)
        return torch.tensor(val, device=like.device, dtype=like.dtype)


    # ------------------ H-function ------------------

    def _H(self, mu, w):
        mu = torch.clamp(mu, 0.0, 1.0)
        w = torch.clamp(w, 1e-6, 1.0-1e-6)

        gamma = torch.sqrt(torch.clamp(1.0 - w, min=1e-6))
        denom = 1.0 + 2.0 * gamma * mu
        denom = torch.clamp(denom, min=1e-6)
        return (1.0 + 2.0 * mu) / denom


    # ------------------ Opposition effects ------------------

    def _B_SH(self, g, B0, h):
        g = torch.clamp(g, 0.0, math.pi - 1e-3)
        tanh = torch.tan(0.5 * g)
        h = torch.clamp(h, min=1e-6)
        denom = 1.0 + (1.0 / h) * tanh
        denom = torch.clamp(denom, min=1e-6)
        return B0 / denom

    def _B_CB(self, g, B0, h):
        g = torch.clamp(g, 0.0, math.pi - 1e-3)
        tanh = torch.tan(0.5 * g)
        h = torch.clamp(h, min=1e-6)
        denom = 1.0 + (tanh / h)**2
        return B0 / torch.clamp(denom, min=1e-6)


    # ------------------ Phase functions ------------------

    def _P(self, g, xi=None, b1=None, b2=None, c=None):
        cg = torch.cos(g)

        if self.phase_fun == "hg1":
            denom = 1 + 2*xi*cg + xi**2
            denom = torch.clamp(denom, min=1e-6)
            return (1 - xi**2) / denom**1.5

        # 2-term HG
        denom1 = 1 + 2*b1*cg + b1**2
        denom2 = 1 + 2*b2*cg + b2**2
        P1 = (1 - b1**2) / torch.clamp(denom1, min=1e-6)**1.5
        P2 = (1 - b2**2) / torch.clamp(denom2, min=1e-6)**1.5
        return c * P1 + (1 - c) * P2



    # ============================================================
    #  Main: Radiance factor I/F
    # ============================================================

    def radiance_factor(self, mu0, mu, g):
        # Convert inputs to tensors
        if not isinstance(mu0, torch.Tensor): mu0 = torch.tensor(mu0)
        if not isinstance(mu, torch.Tensor):  mu = torch.tensor(mu)
        if not isinstance(g, torch.Tensor):   g = torch.tensor(g)

        mu0, mu, g = torch.broadcast_tensors(mu0, mu, g)
        mu0 = mu0.to(g.device).float()
        mu  = mu.to(g.device).float()

        # Map support
        w      = self._to_tensor(self.w, mu0)
        theta  = self._to_tensor(self.theta_bar, mu0)
        B0_sh  = self._to_tensor(self.B0_sh, mu0)
        h_sh   = self._to_tensor(self.h_sh,  mu0)
        B0_cb  = self._to_tensor(self.B0_cb, mu0)
        h_cb   = self._to_tensor(self.h_cb,  mu0)

        # Phase params
        if self.phase_fun == "hg1":
            xi = self._to_tensor(self.xi, mu0)
            phase_params = {"xi": xi}
        else:
            b1 = self._to_tensor(self.b1, mu0)
            b2 = self._to_tensor(self.b2, mu0)
            c  = self._to_tensor(self.c,  mu0)
            phase_params = {"b1": b1, "b2": b2, "c": c}

        # ---------------- Roughness ----------------
        mu0_eff, mu_eff, S = hapke_roughness(mu0, mu, g, theta)

        denom = torch.clamp(mu0_eff + mu_eff, min=self.eps)

        # Phase + opposition effects
        P = self._P(g, **phase_params)
        Bsh = self._B_SH(g, B0_sh, h_sh)
        Bcb = self._B_CB(g, B0_cb, h_cb)
        Btot = Bsh + Bcb

        # H-functions use *unroughened* incident/emission
        H0 = self._H(mu0_eff, w)
        H  = self._H(mu_eff,  w)

        # --------- Bi-directional reflectance r(i,e,g) ---------
        r = (w / (4*math.pi)) * (mu0_eff / denom) * ((1 + Btot)*P + H0*H - 1)

        # Apply roughness shadowing
        r = S * r

        # Radiance factor I/F = π * r
        R = math.pi * r
        R = torch.clamp(R, min=0.0)

        # Apply physical visibility mask
        mask = (mu0 > 0) & (mu > 0)
        R = torch.where(mask, R, torch.zeros_like(R))
        
        return R