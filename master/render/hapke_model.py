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


import torch
import math

def _check_nan_hapke(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        return
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        idx = torch.nonzero(nan_mask, as_tuple=False)[0].tolist()
        msg = (
            f"[hapke_roughness] NaN detected in '{name}' at index {idx}.\n"
            f"  shape = {tuple(tensor.shape)}\n"
        )
        valid = tensor[~nan_mask]
        if valid.numel() > 0:
            msg += (
                f"  valid min={valid.min().item():.6e}, "
                f"max={valid.max().item():.6e}, "
                f"mean={valid.mean().item():.6e}\n"
            )
        raise ValueError(msg)

# ============================================================
#  Roughness correction (Hapke 1984 / 1986 / 1993)
# ============================================================

def hapke_roughness(mu0, mu, g, theta_bar, debug: bool = False):
    """
    Hapke macroscopic roughness correction.
    Returns mu0_eff, mu_eff, S.
    """

    # Sørg for tensors & fælles shape
    if not isinstance(mu0, torch.Tensor): mu0 = torch.tensor(mu0)
    if not isinstance(mu, torch.Tensor):  mu  = torch.tensor(mu)
    if not isinstance(g, torch.Tensor):   g   = torch.tensor(g)

    mu0, mu, g = torch.broadcast_tensors(mu0, mu, g)
    mu0 = mu0.to(dtype=torch.float32)
    mu  = mu.to(dtype=torch.float32)

    theta_bar = torch.as_tensor(theta_bar, device=mu0.device, dtype=mu0.dtype)
    theta_bar = torch.clamp(theta_bar, 1e-6, math.radians(45.0))  # safe range

    if debug:
        _check_nan_hapke(mu0,       "mu0 (input)")
        _check_nan_hapke(mu,        "mu (input)")
        _check_nan_hapke(g,         "g (input)")
        _check_nan_hapke(theta_bar, "theta_bar (input)")

    # *** VIGTIGT: clamp mu0 og mu til [-1, 1] ***
    mu0 = torch.clamp(mu0, -1.0, 1.0)
    mu  = torch.clamp(mu,  -1.0, 1.0)

    # Roughness-parametre
    t = torch.tan(theta_bar)
    sigma = t * torch.sqrt(torch.tensor(2.0 / math.pi,
                                        device=mu0.device,
                                        dtype=mu0.dtype))

    if debug:
        _check_nan_hapke(t,     "t = tan(theta_bar)")
        _check_nan_hapke(sigma, "sigma")

    # radikanter til sqrt
    rad0 = 1.0 - mu0**2
    rad  = 1.0 - mu**2

    # clamp til [0, ∞) for at undgå sqrt(negative)
    eps = 1e-6 # lille positiv værdi, så vi undgår 0 senere, hvor den er i nævneren
    rad0_clamped = torch.clamp(rad0, min=eps)
    rad_clamped  = torch.clamp(rad,  min=eps)

    if debug:
        _check_nan_hapke(rad0,         "rad0 = 1 - mu0^2 (før clamp)")
        _check_nan_hapke(rad,          "rad  = 1 - mu^2  (før clamp)")
        _check_nan_hapke(rad0_clamped, "rad0_clamped")
        _check_nan_hapke(rad_clamped,  "rad_clamped")

    sqrt0 = torch.sqrt(rad0_clamped)
    sqrt1 = torch.sqrt(rad_clamped)

    if debug:
        _check_nan_hapke(sqrt0, "sqrt0 = sqrt(1 - mu0^2)")
        _check_nan_hapke(sqrt1, "sqrt1 = sqrt(1 - mu^2)")

    # Corrected incidence cosine
    cos_theta = torch.cos(theta_bar)
    cos_theta = torch.clamp(cos_theta, min=1e-6)  # undgå 1/0

    factor = (1.0 / cos_theta - 1.0)

    mu0_eff = mu0 * (1.0 + factor * sqrt0)
    mu0_eff = torch.clamp(mu0_eff, min=0.0, max=1.0)

    # Corrected emission cosine
    mu_eff = mu * (1.0 + factor * sqrt1)
    mu_eff = torch.clamp(mu_eff, min=0.0, max=1.0)

    if debug:
        # Hvis der stadig er NaNs, print lokalt context
        nan_mask = torch.isnan(mu_eff)
        if nan_mask.any():
            idx = torch.nonzero(nan_mask, as_tuple=False)[0]
            y, x = idx[-2:].tolist() if mu_eff.dim() == 2 else idx[-3:].tolist()[1:]
            mu_val      = mu[idx]
            mu0_val     = mu0[idx]
            theta_val   = theta_bar[idx] if theta_bar.shape == mu.shape else theta_bar
            rad_val     = rad[idx]
            rad_cl_val  = rad_clamped[idx]
            print(
                "[hapke_roughness DEBUG] NaN i mu_eff ved pixel:\n"
                f"  index = {idx.tolist()}\n"
                f"  mu      = {mu_val.item():.6e}\n"
                f"  mu0     = {mu0_val.item():.6e}\n"
                f"  theta   = {theta_val.item():.6e} rad\n"
                f"  rad     = 1 - mu^2 = {rad_val.item():.6e}\n"
                f"  rad_cl  = {rad_cl_val.item():.6e}\n"
            )
            _check_nan_hapke(mu_eff, "mu_eff")

        _check_nan_hapke(mu0_eff, "mu0_eff")
        _check_nan_hapke(mu_eff,  "mu_eff")

    # Shadowing probability
    sigma_safe = torch.clamp(sigma, min=1e-6)
    P0 = 1.0 - torch.exp(- (mu0 / sigma_safe)**2)
    P  = 1.0 - torch.exp(- (mu  / sigma_safe)**2)

    if debug:
        _check_nan_hapke(P0, "P0 (shadowing-incidence)")
        _check_nan_hapke(P,  "P  (shadowing-emission)")

    S = P0 * P + 1e-6

    if debug:
        _check_nan_hapke(S, "S (shadowing term)")

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
                 c=0.7,        # for hg2

                 debug: bool = False
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

        self.debug = debug


    # ------------------ helper ------------------


    def _check_nan(self, tensor, name):
        if not isinstance(tensor, torch.Tensor):
            return
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            # Tag første NaN indeks
            idx = torch.nonzero(nan_mask, as_tuple=False)[0].tolist()
            msg = (
                f"[FullHapkeModel] NaN detected in '{name}' at index {idx}.\n"
                f"  tensor shape: {tuple(tensor.shape)}\n"
            )
            # prøv at printe nogle stats, hvis der også er valide værdier
            valid = tensor[~nan_mask]
            if valid.numel() > 0:
                msg += (
                    f"  valid min={valid.min().item():.6e}, "
                    f"max={valid.max().item():.6e}, "
                    f"mean={valid.mean().item():.6e}\n"
                )
            raise ValueError(msg)


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

        if self.debug:
            # --- Debug: check inputs ---
            self._check_nan(mu0, "mu0 (input)")
            self._check_nan(mu,  "mu (input)")
            self._check_nan(g,   "g (input)")


        # Map support
        w      = self._to_tensor(self.w, mu0)
        theta  = self._to_tensor(self.theta_bar, mu0)
        B0_sh  = self._to_tensor(self.B0_sh, mu0)
        h_sh   = self._to_tensor(self.h_sh,  mu0)
        B0_cb  = self._to_tensor(self.B0_cb, mu0)
        h_cb   = self._to_tensor(self.h_cb,  mu0)

        if self.debug:
            # Debug parameter maps
            self._check_nan(w,     "w (albedo map)")
            self._check_nan(theta, "theta_bar (roughness map)")
            self._check_nan(B0_sh, "B0_sh")
            self._check_nan(h_sh,  "h_sh")
            self._check_nan(B0_cb, "B0_cb")
            self._check_nan(h_cb,  "h_cb")


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
        mu0_eff, mu_eff, S = hapke_roughness(mu0, mu, g, theta, debug=self.debug)

        if self.debug:
            # Debug efter roughness
            self._check_nan(mu0_eff, "mu0_eff (after hapke_roughness)")
            self._check_nan(mu_eff,  "mu_eff (after hapke_roughness)")
            self._check_nan(S,       "S (roughness shadowing)")


        denom = torch.clamp(mu0_eff + mu_eff, min=self.eps)

        # Phase + opposition effects
        P = self._P(g, **phase_params)
        Bsh = self._B_SH(g, B0_sh, h_sh)
        Bcb = self._B_CB(g, B0_cb, h_cb)
        Btot = Bsh + Bcb

        if self.debug:
            # Debug efter fase + opposition
            self._check_nan(P,     "P (phase function)")
            self._check_nan(Bsh,  "B_SH (opposition)")
            self._check_nan(Bcb,  "B_CB (opposition)")
            self._check_nan(Btot, "Btot (B_SH + B_CB)")


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