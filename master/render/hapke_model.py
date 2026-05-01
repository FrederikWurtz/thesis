import numpy as np
import torch
import math


class LambertianModel:
    def __init__(self, w=0.5):
        """
        Initialize the LambertianModel with given parameters.

        Parameters:
        w (float): Albedo (0 < w < 1).
        """
        self.name = "lambertian"
        self.w = w

    def radiance_factor(self, mu0, mu):
        """
        Compute the radiance factor (I/F) using the Lambertian model.

        Parameters:
        mu0 (float or torch.Tensor): Cosine of the incidence angle.
        mu (float or torch.Tensor): Cosine of the emission angle.

        Returns:
        float or torch.Tensor: Radiance factor (I/F).
        """
        if not isinstance(mu0, torch.Tensor):
            mu0 = torch.tensor(mu0, dtype=torch.float32)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)

        # Radiance factor for Lambertian is simply w/π * cos(incidence)
        R = self.w / math.pi * mu0
        return torch.where((mu0 > 0) & (mu > 0), R, torch.zeros_like(R))

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
        self.name = "simplehapke"

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
#  Roughness correction (Hapke 1994, chapter 12)
# ============================================================

def cot(x):
    """
    Compute cotangent, handling both scalars and tensors of any shape.
    
    Parameters:
    x (float or torch.Tensor): Input value(s) in radians.
    
    Returns:
    float or torch.Tensor: cos(x) / sin(x), with special handling for small sin(x).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    sin_x = torch.sin(x)
    
    # Where sin(x) is too small, use a large number; otherwise compute cot normally
    result = torch.where(
        torch.abs(sin_x) < 1e-12,
        torch.full_like(x, 1e12),
        torch.cos(x) / sin_x
    )
    
    return result

def hapke_roughness(mu0, mu, i, e, psi, theta_bar, debug: bool = False):
    """
    Hapke macroscopic roughness correction.
    Returns mu0_eff, mu_eff, S.
    """

    # Sørg for tensors & fælles shape
    if not isinstance(mu0, torch.Tensor): mu0 = torch.tensor(mu0)
    if not isinstance(mu, torch.Tensor):  mu  = torch.tensor(mu)
    if not isinstance(i, torch.Tensor):   i   = torch.tensor(i)
    if not isinstance(e, torch.Tensor):   e   = torch.tensor(e)
    if not isinstance(psi, torch.Tensor): psi = torch.tensor(psi)

    mu0, mu, i, e, psi = torch.broadcast_tensors(mu0, mu, i, e, psi)
    mu0 = mu0.to(dtype=torch.float32)
    mu  = mu.to(dtype=torch.float32)

    theta_bar = torch.as_tensor(theta_bar, device=mu0.device, dtype=mu0.dtype)

    if debug:
        _check_nan_hapke(mu0,       "mu0 (input)")
        _check_nan_hapke(mu,        "mu (input)")
        _check_nan_hapke(i,         "i (input)")
        _check_nan_hapke(e,         "e (input)")
        _check_nan_hapke(psi,       "psi (input)")
        _check_nan_hapke(theta_bar, "theta_bar (input)")


    # Roughness parameters that are constant over map
    chi = torch.sqrt(1 + torch.pi*torch.tan(theta_bar)**2)
    
    if debug: # print values of roughness parameters and input angles (degrees and radians) for debugging
        # These are maps, so just print stats
        print(f"Input angles (degrees): i={torch.rad2deg(i).min().item():.2f} to {torch.rad2deg(i).max().item():.2f}, e={torch.rad2deg(e).min().item():.2f} to {torch.rad2deg(e).max().item():.2f}, psi={torch.rad2deg(psi).min().item():.2f} to {torch.rad2deg(psi).max().item():.2f}")
        print(f"Roughness parameter (constant): chi={chi.mean().item():.6f}, theta_bar={torch.rad2deg(theta_bar).mean().item():.2f} degrees")
        
    eps = 1e-6
    i_safe = torch.clamp(i, min=0.0, max=math.pi / 2 - eps)
    e_safe = torch.clamp(e, min=0.0, max=math.pi / 2 - eps)
    
    # Roughness parameters that vary over the map
    E1_e_map = torch.exp(-2 / torch.pi * cot(theta_bar) * cot(e_safe) )
    E1_i_map = torch.exp(-2 / torch.pi * cot(theta_bar) * cot(i_safe) )
    E2_e_map = torch.exp(-1 / torch.pi * (cot(theta_bar))**2 * (cot(e_safe))**2 )
    E2_i_map = torch.exp(-1 / torch.pi * (cot(theta_bar))**2 * (cot(i_safe))**2 )
    # if psi is 180, then tan(psi/2) is infinite and f_psi should be 0, so clamp psi to just below 180 degrees to avoid NaN
    psi_clamped = torch.clamp(psi, max=math.pi - 1e-6)
    f_psi = torch.exp(- 2 * torch.tan(psi_clamped / 2))
    
    if debug: # print values of roughness parameters that vary over the map for debugging
        print(f"Roughness parameters (maps): E1_e_map={E1_e_map.mean().item():.6f}, E1_i_map={E1_i_map.mean().item():.6f}, E2_e_map={E2_e_map.mean().item():.6f}, E2_i_map={E2_i_map.mean().item():.6f}, f_psi={f_psi.mean().item():.6f}")
    
    # compute effective cosines for incidence and emission depending on which is largest
    eps = 1e-6
    denom_ie = torch.clamp(2 - E1_e_map - (psi / torch.pi) * E1_i_map, min=eps)
    mu0_eff_i_leq_e = chi * (torch.cos(i) + torch.sin(i) * torch.tan(theta_bar) * (torch.cos(psi) * E2_e_map + torch.sin(psi / 2)**2 * E2_i_map) / denom_ie)
    
    denom_i0 = torch.clamp(2 - E1_i_map, min=eps)
    mu0_eff_i_leq_e_at_0 = chi * (torch.cos(i) + torch.sin(i) * torch.tan(theta_bar) * E2_i_map / denom_i0)
    
    mu_eff_i_leq_e = chi * (torch.cos(e) + torch.sin(e) * torch.tan(theta_bar) * (E2_e_map - torch.sin(psi / 2)**2 * E2_i_map) / denom_ie)
    
    denom_e0 = torch.clamp(2 - E1_e_map, min=eps)
    mu_eff_i_leq_e_at_0 = chi * (torch.cos(e) + torch.sin(e) * torch.tan(theta_bar) * E2_e_map / denom_e0)
    
    denom_ei = torch.clamp(2 - E1_i_map - (psi / torch.pi) * E1_e_map, min=eps)
    mu0_eff_e_lt_i = chi * (torch.cos(i) + torch.sin(i) * torch.tan(theta_bar) * (E2_i_map - torch.sin(psi / 2)**2 * E2_e_map) / denom_ei)
    mu0_eff_e_lt_i_at_0 = chi * (torch.cos(i) + torch.sin(i) * torch.tan(theta_bar) * E2_i_map / denom_i0)
    mu_eff_e_lt_i = chi * (torch.cos(e) + torch.sin(e) * torch.tan(theta_bar) * (torch.cos(psi) * E2_i_map + torch.sin(psi / 2)**2 * E2_e_map) / denom_ei)
    mu_eff_e_lt_i_at_0 = chi * (torch.cos(e) + torch.sin(e) * torch.tan(theta_bar) * E2_e_map / denom_e0)

    # for each pixel, select the correct effective cosines based on whether i <= e or not
    mu0_eff = torch.where(i <= e, mu0_eff_i_leq_e, mu0_eff_e_lt_i)
    mu_eff = torch.where(i <= e, mu_eff_i_leq_e, mu_eff_e_lt_i)

    # for each pixel, calculate the roughness shadowing factor S - clamping denominators to avoid NaN
    S_i_leq_e = mu_eff_i_leq_e / torch.clamp(mu_eff_i_leq_e_at_0, min=eps) * mu0 / torch.clamp(mu0_eff_i_leq_e_at_0, min=eps) * chi / torch.clamp(1 - f_psi + f_psi * chi * (mu0 / torch.clamp(mu0_eff_i_leq_e_at_0, min=eps)), min=eps)
    S_e_lt_i = mu_eff_e_lt_i / torch.clamp(mu_eff_e_lt_i_at_0, min=eps) * mu0 / torch.clamp(mu0_eff_e_lt_i_at_0, min=eps) * chi / torch.clamp(1 - f_psi + f_psi * chi * (mu / torch.clamp(mu_eff_e_lt_i_at_0, min=eps)), min=eps)

    S = torch.where(i <= e, S_i_leq_e, S_e_lt_i)

    return mu0_eff, mu_eff, S

# ============================================================
#  Full Hapke Model Class - following the 1994 book
# ============================================================

class FullHapkeModel:
    """
    Full Hapke model for lunar-surface simulation, following the 1994 book.
    Supports:
      - spatial parameter maps
      - Opposition effect
      - 1-term or 2-term HG
      - macroscopic roughness
    """

    def __init__(self,
                 # Albedo
                 w=0.12,

                 # Roughness
                 theta_bar_rad=math.radians(20),

                 # Opposition effect
                 B0=0.6,
                 h=0.05,

                 # Phase function
                 phase_fun="hg2",
                 xi=0.25,     # only for hg1
                 b=-0.3,     # for hg2
                 c=0.7,        # for hg2

                 debug: bool = False,
                 smooth_transition: bool = False # whether to apply a smooth transition at the shadow boundary to avoid killing gradients (should only be used during training, not for image rendering)
                 ):
        
        self.w = w
        self.theta_bar_rad = theta_bar_rad

        self.B0 = B0
        self.h = h


        self.phase_fun = phase_fun.lower()
        self.xi = xi

        self.b = b
        self.c = c

        self.eps = 1e-12

        self.debug = debug
        self.name = "fullhapke"
        
        self.smooth_transition = smooth_transition
        self.training = False
        self.k = 30  # steepness of sigmoid transition, adjust as needed

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


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

    def _H(self, x, w):
        
        x = torch.clamp(x, min=1e-6, max=1.0)
        w = torch.clamp(w, 1e-6, 1.0-1e-6)
    

        y = torch.sqrt(1.0 - w) # albedo factor
        r0 = (1.0 - y) / (1.0 + y) # diffuse reflectance
        
        bracket = r0 + (1 - 0.5 * r0 - x * r0) * torch.log((1.0 + x) / x)
        denom = 1.0 - (1 - y) * x * bracket
        
        denom = torch.clamp(denom, min=1e-6)
        return 1 / denom


    # ------------------ Opposition effect ------------------

    def _B(self, g, B0, h): # g in radians
        g = torch.clamp(g, 0.0, math.pi - 1e-4)
        tanh = torch.tan(0.5 * g)
        h = torch.clamp(h, min=1e-6, max=1.0)
        denom = 1.0 + (1.0 / h) * tanh
        denom = torch.clamp(denom, min=1e-6)
        return B0 / denom


    # ------------------ Phase functions ------------------

    def _P(self, g, xi=None, b=None, c=None):
        cg = torch.cos(g)

        if self.phase_fun == "hg1":
            denom = 1 + 2*xi*cg + xi**2
            denom = torch.clamp(denom, min=1e-6)
            return (1 - xi**2) / denom**1.5

        # 2-term HG
        denom1 = 1 + 2*b*cg + b**2
        denom2 = 1 - 2*b*cg + b**2
        P1 = (1 - b**2) / torch.clamp(denom1, min=1e-6)**1.5
        P2 = (1 - b**2) / torch.clamp(denom2, min=1e-6)**1.5
        return (1 - c) * P1 + c * P2

    # compute psi from from i, e, g:
    
    def compute_psi_from_g(self, i_rad, e_rad, g_rad):
        """
        Compute azimuth difference ψ from i, e, g
        using Hapke's relation:
        
        cos g = cos i cos e + sin i sin e cos ψ
        """
        ci = torch.cos(i_rad)
        ce = torch.cos(e_rad)
        si = torch.sin(i_rad)
        se = torch.sin(e_rad)


        # avoid division by zero
        eps = 1e-6
        denom = si * se
        small = denom.abs() < eps
        denom_safe = torch.where(small, torch.ones_like(denom), denom)

        # normal case
        cos_psi = (torch.cos(g_rad) - ci * ce) / denom_safe

        # clamp for numerical safety
        cos_psi = torch.where(small, torch.ones_like(cos_psi), cos_psi)  # psi=0 later
        cos_psi = torch.clamp(cos_psi, -1.0 + eps, 1.0 - eps)

        psi = torch.acos(cos_psi)

        # if i=0 or e=0, define psi = 0
        psi = torch.where(small, torch.zeros_like(psi), psi)

        return psi



    # ============================================================
    #  Main: Radiance factor I/F
    # ============================================================

    def radiance_factor(self, mu0, mu, g_rad, e_rad, i_rad):
        # Convert inputs to tensors
        if not isinstance(mu0, torch.Tensor): mu0 = torch.tensor(mu0)
        if not isinstance(mu, torch.Tensor):  mu = torch.tensor(mu)
        if not isinstance(g_rad, torch.Tensor):   g_rad = torch.tensor(g_rad)
        if not isinstance(e_rad, torch.Tensor):   e_rad = torch.tensor(e_rad)
        if not isinstance(i_rad, torch.Tensor):   i_rad = torch.tensor(i_rad)

        # Broadcast to common shape
        mu0, mu, g_rad, e_rad, i_rad = torch.broadcast_tensors(mu0, mu, g_rad, e_rad, i_rad)
        mu0 = mu0.to(g_rad.device).float()
        mu  = mu.to(g_rad.device).float()
        g_rad   = g_rad.to(g_rad.device).float()
        e_rad   = e_rad.to(g_rad.device).float()
        i_rad   = i_rad.to(g_rad.device).float()


        if self.debug:
            # --- Debug: check inputs ---
            self._check_nan(mu0, "mu0 (input)")
            self._check_nan(mu,  "mu (input)")
            self._check_nan(g_rad,   "g_rad (input)")
            self._check_nan(e_rad,   "e_rad (input)")
            self._check_nan(i_rad,   "i_rad (input)")


        # Map support
        w      = self._to_tensor(self.w, mu0)
        theta_bar_rad  = self._to_tensor(self.theta_bar_rad, mu0)
        B0  = self._to_tensor(self.B0, mu0)
        h   = self._to_tensor(self.h,  mu0)

        if self.debug:
            # Debug parameter maps
            self._check_nan(w,     "w (albedo map)")
            self._check_nan(theta_bar_rad, "theta_bar_rad (roughness map)")
            self._check_nan(B0, "B0")
            self._check_nan(h,  "h")


        # Phase params
        if self.phase_fun == "hg1":
            xi = self._to_tensor(self.xi, mu0)
            phase_params = {"xi": xi}
        else:
            b = self._to_tensor(self.b, mu0)
            c  = self._to_tensor(self.c,  mu0)
            phase_params = {"b": b, "c": c}

        # ---------------- Roughness ----------------
        psi_rad = self.compute_psi_from_g(i_rad, e_rad, g_rad)
        mu0_eff, mu_eff, S = hapke_roughness(mu0, mu, i_rad, e_rad, psi_rad, theta_bar_rad, debug=self.debug)

        if self.debug:
            # Debug efter roughness
            self._check_nan(mu0_eff, "mu0_eff (after hapke_roughness)")
            self._check_nan(mu_eff,  "mu_eff (after hapke_roughness)")
            self._check_nan(S,       "S (roughness shadowing)")


        denom = torch.clamp(mu0_eff + mu_eff, min=self.eps)

        # Phase + opposition effects
        P = self._P(g_rad, **phase_params)
        B = self._B(g_rad, B0, h)

        if self.debug:
            # Debug efter fase + opposition
            self._check_nan(P,     "P (phase function)")
            self._check_nan(B,  "B (opposition)")


        # H-functions use roughened incident/emission cosines
        H0 = self._H(mu0_eff, w)
        H  = self._H(mu_eff,  w)

        # --------- Bi-directional reflectance r(i,e,g) ---------
        r = (w / (4*math.pi)) * (mu0_eff / denom) * ((1 + B)*P + H0*H - 1)

        # Apply roughness shadowing
        r = S * r

        # Radiance factor I/F = π * r 
        R = math.pi * r
        R = torch.clamp(R, min=0.0)

        # Apply physical visibility mask
        mask = (mu0 > 0) & (mu > 0)
        
        # Smooth transition to not kill gradients at the sahdow boundary. Should only be used during training, not for image rendering.
        if self.smooth_transition:
            vis = torch.sigmoid(self.k * mu0) * torch.sigmoid(self.k * mu)
            R = vis * R
        else:
            R = torch.where(mask, R, torch.zeros_like(R))
        
        return R