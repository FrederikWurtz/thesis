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
        
        # Clamp g_rad to avoid tan_safe(π/2) = inf
        # Phase angles typically range from 0 to π, but clamp to safe range
        g_clamped = torch.clamp(g_rad, min=0, max=torch.pi - 0.01)
        
        tan_term = tan_safe(0.5 * g_clamped)
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

def cot(x, eps=1e-3, max_val=1e3):
    """
    Blackwell-safe cotangent for Hapke physics.
    Prevents NaN/Inf in both forward and backward.

    Parameters
    ----------
    x : float or torch.Tensor
        Angle(s) in radians.
    eps : float
        Safety cutoff to avoid sin(x) ~ 0.
    max_val : float
        Maximum magnitude of cot(x).

    Returns
    -------
    torch.Tensor
        Numerically safe cotangent.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Clamp angle away from singularities
    x_safe = torch.clamp(x, min=eps, max=(torch.pi / 2 - eps))

    sin_x = torch.sin(x_safe)
    cos_x = torch.cos(x_safe)

    # *** KEY FIX ***
    # Clamp denominator BEFORE division
    sin_x = torch.clamp(sin_x, min=eps)

    cot_x = cos_x / sin_x

    # Bound magnitude (prevents gradient explosion)
    cot_x = torch.clamp(cot_x, min=-max_val, max=max_val)

    # Absolute last-resort safety
    cot_x = torch.nan_to_num(cot_x, nan=0.0, posinf=max_val, neginf=-max_val)

    return cot_x


def tan_safe(x, eps=1e-3, max_val=10.0):
    """
    Blackwell-safe tangent for Hapke-style physics.
    Prevents NaN/Inf before torch.tan().
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # 1️⃣ Sanitize first (CRITICAL)
    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=(torch.pi / 2 - eps),
        neginf=-(torch.pi / 2 - eps)
    )

    # 2️⃣ Clamp away from singularities
    x = torch.clamp(x, min=-torch.pi / 2 + eps, max=torch.pi / 2 - eps)

    # 3️⃣ Safe tan
    tan_x = torch.tan(x)

    # 4️⃣ Bound magnitude (prevents gradient explosion)
    tan_x = torch.clamp(tan_x, min=-max_val, max=max_val)

    # 5️⃣ Absolute last line of defense
    tan_x = torch.nan_to_num(
        tan_x,
        nan=0.0,
        posinf=max_val,
        neginf=-max_val
    )

    return tan_x

def hapke_roughness(mu0, mu, i, e, psi, theta_bar, debug: bool = False, big_num = False, eps = False):
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
    theta_bar = torch.clamp(theta_bar, max=torch.pi * 70.0 / 180.0)

    if debug:
        _check_nan_hapke(mu0,       "mu0 (input)")
        _check_nan_hapke(mu,        "mu (input)")
        _check_nan_hapke(i,         "i (input)")
        _check_nan_hapke(e,         "e (input)")
        _check_nan_hapke(psi,       "psi (input)")
        _check_nan_hapke(theta_bar, "theta_bar (input)")


    # Roughness parameters that are constant over map
    chi = torch.sqrt(1 + torch.pi*tan_safe(theta_bar)**2)
    chi = torch.clamp(chi, min=1.0, max=3.0)
    
    if debug: # print values of roughness parameters and input angles (degrees and radians) for debugging
        # These are maps, so just print stats
        print(f"Input angles (degrees): i={torch.rad2deg(i).min().item():.2f} to {torch.rad2deg(i).max().item():.2f}, e={torch.rad2deg(e).min().item():.2f} to {torch.rad2deg(e).max().item():.2f}, psi={torch.rad2deg(psi).min().item():.2f} to {torch.rad2deg(psi).max().item():.2f}")
        print(f"Roughness parameter (constant): chi={chi.mean().item():.6f}, theta_bar={torch.rad2deg(theta_bar).mean().item():.2f} degrees")
        
    i_safe = torch.clamp(i, min=0.0, max=math.pi / 2 - eps)
    e_safe = torch.clamp(e, min=0.0, max=math.pi / 2 - eps)
    

    cot_tb = cot(theta_bar)
    cot_i  = cot(i_safe)
    cot_e  = cot(e_safe)


    exp_arg_E1_e = -2 / torch.pi * cot_tb * cot_e
    exp_arg_E1_i = -2 / torch.pi * cot_tb * cot_i
    exp_arg_E2_e = -1 / torch.pi * (cot_tb)**2 * (cot_e)**2 
    exp_arg_E2_i = -1 / torch.pi * (cot_tb)**2 * (cot_i)**2 

    exp_arg_E1_e = torch.clamp(exp_arg_E1_e, min=-20.0, max=4.0)
    exp_arg_E1_i = torch.clamp(exp_arg_E1_i, min=-20.0, max=4.0)
    exp_arg_E2_e = torch.clamp(exp_arg_E2_e, min=-20.0, max=4.0)
    exp_arg_E2_i = torch.clamp(exp_arg_E2_i, min=-20.0, max=4.0)

    E1_e_map = torch.exp(exp_arg_E1_e)
    E1_i_map = torch.exp(exp_arg_E1_i)
    
    E2_e_map = torch.exp(exp_arg_E2_e)
    E2_e_map = torch.nan_to_num(E2_e_map, nan=1.0, posinf=10.0, neginf=0.0)
    E2_e_map = E2_e_map.clamp(0.0, 10.0)

    E2_i_map = torch.exp(exp_arg_E2_i)
    E2_i_map = torch.nan_to_num(E2_i_map, nan=1.0, posinf=10.0, neginf=0.0)
    E2_i_map = E2_i_map.clamp(0.0, 10.0)


    # Roughness parameters that vary over the map
    #E1_e_map = torch.exp(-2 / torch.pi * cot(theta_bar) * cot(e_safe) )
    #E1_i_map = torch.exp(-2 / torch.pi * cot(theta_bar) * cot(i_safe) )
    #E2_e_map = torch.exp(-1 / torch.pi * (cot(theta_bar))**2 * (cot(e_safe))**2 )
    #E2_i_map = torch.exp(-1 / torch.pi * (cot(theta_bar))**2 * (cot(i_safe))**2 )


    # ------------------------------------------------------------
    # Blackwell-safe handling of psi and psi/2
    # ------------------------------------------------------------

    # 1) Sanitize psi first (must happen before any nonlinear op)
    psi_safe = torch.nan_to_num(
        psi,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # 2) Enforce physical domain: 0 <= psi < pi
    psi_safe = torch.clamp(psi_safe, min=0.0, max=torch.pi - eps)

    # 3) Half-angle, safely away from tan(pi/2)
    psi_half = psi_safe * 0.5
    psi_half = torch.clamp(psi_half, min=eps, max=torch.pi / 2 - eps)

    # ------------------------------------------------------------
    # sin^2(psi/2) (used in E2 terms)
    # ------------------------------------------------------------

    sin_psi2 = torch.sin(psi_half)
    sin_psi2_sq = sin_psi2 * sin_psi2
    

    # ------------------------------------------------------------
    # f_psi = exp(-2 * tan(psi/2)) (used in shadowing)
    # ------------------------------------------------------------

    f_psi = torch.exp(-2.0 * tan_safe(psi_half))


    if debug: # print values of roughness parameters that vary over the map for debugging
        print(f"Roughness parameters (maps): E1_e_map={E1_e_map.mean().item():.6f}, E1_i_map={E1_i_map.mean().item():.6f}, E2_e_map={E2_e_map.mean().item():.6f}, E2_i_map={E2_i_map.mean().item():.6f}, f_psi={f_psi.mean().item():.6f}")
    # ------------------------------------------------------------
    # Effective cosines (Blackwell-safe: no inline division)
    # ------------------------------------------------------------

    tan_tb = tan_safe(theta_bar)

    # ---------- shared denominators (sanitized first) ----------

    denom_ei = 2.0 - E1_e_map - (psi_safe / torch.pi) * E1_i_map
    denom_ei = torch.nan_to_num(denom_ei, nan=1.0, posinf=1.0, neginf=1.0)
    denom_ei = torch.clamp(denom_ei, min=eps)

    denom_i0 = 2.0 - E1_i_map
    denom_i0 = torch.nan_to_num(denom_i0, nan=1.0, posinf=1.0, neginf=1.0)
    denom_i0 = torch.clamp(denom_i0, min=eps)

    denom_e0 = 2.0 - E1_e_map
    denom_e0 = torch.nan_to_num(denom_e0, nan=1.0, posinf=1.0, neginf=1.0)
    denom_e0 = torch.clamp(denom_e0, min=eps)

    denom_ie = 2.0 - E1_i_map - (psi_safe / torch.pi) * E1_e_map
    denom_ie = torch.nan_to_num(denom_ie, nan=1.0, posinf=1.0, neginf=1.0)
    denom_ie = torch.clamp(denom_ie, min=eps)

    # ---------- i <= e branch ----------
    #mu0_eff calculations
    mu0_num_i_leq_e = (
        torch.sin(i) * tan_tb
        * (torch.cos(psi_safe) * E2_e_map + sin_psi2_sq * E2_i_map)
    )

    mu0_num_i_leq_e = torch.nan_to_num(mu0_num_i_leq_e, nan=0.0, posinf=10.0, neginf=-10.0)
    mu0_num_i_leq_e = mu0_num_i_leq_e.clamp(-10.0, 10.0)

    mu0_ratio_i_leq_e = mu0_num_i_leq_e / denom_ei
    mu0_ratio_i_leq_e = torch.nan_to_num(mu0_ratio_i_leq_e, nan=0.0, posinf=10.0, neginf=-10.0)
    mu0_ratio_i_leq_e = mu0_ratio_i_leq_e.clamp(-10.0, 10.0)

    #CHECKED
    mu0_eff_i_leq_e = chi * (torch.cos(i) + mu0_ratio_i_leq_e)


    #mu0_eff at 0 calculations
    num_i0 = torch.sin(i) * tan_tb * E2_i_map

    num_i0 = torch.nan_to_num(num_i0, nan=0.0, posinf=10.0, neginf=-10.0)
    num_i0 = num_i0.clamp(-10.0, 10.0)

    ratio_i0 = num_i0 / denom_i0
    ratio_i0 = torch.nan_to_num(ratio_i0, nan=0.0, posinf=10.0, neginf=-10.0)
    ratio_i0 = ratio_i0.clamp(-10.0, 10.0)

    #CHECKED
    mu0_eff_shared_at_0 = chi * (torch.cos(i) + ratio_i0)

    #mu_eff calculations
    mu_num_i_leq_e = (
        torch.sin(e) * tan_tb
        * (E2_e_map - sin_psi2_sq * E2_i_map)
    )
    mu_num_i_leq_e = torch.nan_to_num(mu_num_i_leq_e, nan=0.0, posinf=10.0, neginf=-10.0)
    mu_num_i_leq_e = mu_num_i_leq_e.clamp(-10.0, 10.0)

    mu_ratio_i_leq_e = mu_num_i_leq_e / denom_ei
    mu_ratio_i_leq_e = torch.nan_to_num(mu_ratio_i_leq_e, nan=0.0, posinf=10.0, neginf=-10.0)
    mu_ratio_i_leq_e = mu_ratio_i_leq_e.clamp(-10.0, 10.0)

    #CHECKED
    mu_eff_i_leq_e = chi * (torch.cos(e) + mu_ratio_i_leq_e)

    #mu_eff at 0 calculations
    num_e0 = torch.sin(e) * tan_tb * E2_e_map
    num_e0 = torch.nan_to_num(num_e0, nan=0.0, posinf=10.0, neginf=-10.0)
    num_e0 = num_e0.clamp(-10.0, 10.0)

    ratio_e0 = num_e0 / denom_e0

    #Checked
    mu_eff_shared_at_0 = chi * (torch.cos(e) + ratio_e0)

    # ---------- e < i branch ----------
    #mu0_eff calculations
    mu0_num_e_lt_i = (
        torch.sin(i) * tan_tb
        * (E2_i_map - sin_psi2_sq * E2_e_map)
    )
    mu0_num_e_lt_i = torch.nan_to_num(mu0_num_e_lt_i, nan=0.0, posinf=10.0, neginf=-10.0)
    mu0_num_e_lt_i = mu0_num_e_lt_i.clamp(-10.0, 10.0)

    mu0_ratio_e_lt_i = mu0_num_e_lt_i / denom_ie
    mu0_ratio_e_lt_i = torch.nan_to_num(mu0_ratio_e_lt_i, nan=0.0, posinf=10.0, neginf=-10.0)
    mu0_ratio_e_lt_i = mu0_ratio_e_lt_i.clamp(-10.0, 10.0)

    #CHECKED
    mu0_eff_e_lt_i = chi * (torch.cos(i) + mu0_ratio_e_lt_i)

    #mu0_eff_at_0 calculations
    #CHECKED
    # at 0 term is shared

    #mu_eff calculations
    mu_num_e_lt_i = (
        torch.sin(e) * tan_tb
        * (torch.cos(psi_safe) * E2_i_map + sin_psi2_sq * E2_e_map)
    )
    mu_num_e_lt_i = torch.nan_to_num(mu_num_e_lt_i, nan=0.0, posinf=10.0, neginf=-10.0)
    mu_num_e_lt_i = mu_num_e_lt_i.clamp(-10.0, 10.0)

    mu_ratio_e_lt_i = mu_num_e_lt_i / denom_ie
    mu_ratio_e_lt_i = torch.nan_to_num(mu_ratio_e_lt_i, nan=0.0, posinf=10.0, neginf=-10.0)
    mu_ratio_e_lt_i = mu_ratio_e_lt_i.clamp(-10.0, 10.0)

    #CHECKED
    mu_eff_e_lt_i = chi * (torch.cos(e) + mu_ratio_e_lt_i)


    # at zero term is shared  # same term as for i leq e

    # ------------------------------------------------------------
    # Select correct effective cosines per pixel
    # ------------------------------------------------------------

    mu0_eff = torch.where(i <= e, mu0_eff_i_leq_e, mu0_eff_e_lt_i)
    mu_eff  = torch.where(i <= e, mu_eff_i_leq_e,  mu_eff_e_lt_i)

    # ------------------------------------------------------------
    # Shadowing factor S (already safe structure)
    # ------------------------------------------------------------
    
    # ---------- shared terms ----------
    S_term_1 = mu_eff / torch.clamp(mu_eff_shared_at_0, min=eps)
    S_term_1 = torch.clamp(S_term_1, max = big_num)
    S_term_2 = mu0 / torch.clamp(mu0_eff_shared_at_0, min=eps)
    S_term_2 = torch.clamp(S_term_2, max = big_num)

    # ---------- i <= e branch ----------
    S_denom_i_leq_e = 1.0 - f_psi + f_psi * chi * S_term_2
    S_denom_i_leq_e = torch.clamp(S_denom_i_leq_e, min=eps, max=10.0)

    S_i_leq_e = S_term_1 * S_term_2 * chi / S_denom_i_leq_e

    # ---------- e < i branch ----------
    S_denom_term_e_lt_i = mu / torch.clamp(mu_eff_shared_at_0, min=eps)
    S_denom_term_e_lt_i = torch.clamp(S_denom_term_e_lt_i, max=10.0)
    S_denom_e_lt_i = 1.0 - f_psi + f_psi * chi * S_denom_term_e_lt_i

    S_e_lt_i = S_term_1 * S_term_2 * chi / S_denom_e_lt_i

    #choose which
    S = torch.where(i <= e, S_i_leq_e, S_e_lt_i)

    # ------------------------------------------------------------
    # Final safety (cheap, guarantees clean outputs)
    # ------------------------------------------------------------

    mu0_eff = torch.nan_to_num(mu0_eff, nan=0.0, posinf=1.0, neginf=0.0)
    mu_eff  = torch.nan_to_num(mu_eff,  nan=0.0, posinf=1.0, neginf=0.0)
    S       = torch.nan_to_num(S,       nan=1.0, posinf=1.0, neginf=0.0)

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
                 theta_bar_rad=math.radians(23.4),
                 #theta_bar_rad=math.radians(20),

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

        self.eps = 1e-6
        self.big_num = 1e3

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



    def _to_tensor(self, x, ref):
        """
        Convert scalar or tensor to a CUDA tensor matching ref,
        fully sanitized for Blackwell GPUs.
        """
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x, dtype=ref.dtype)

        # Move to correct device if necessary
        if t.device != ref.device:
            t = t.to(device=ref.device)

        # Ensure dtype
        if t.dtype != ref.dtype:
            t = t.to(dtype=ref.dtype)

        # ***** CRITICAL SAFETY STEP *****
        # Blackwell requires NaN/Inf be removed BEFORE any CUDA math
        t = torch.nan_to_num(
            t,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return t


    # ------------------ H-function ------------------

    def _H(self, x, w):
        # Ensure tensor safety
        x = torch.nan_to_num(
            x,
            nan=self.eps,
            posinf=1.0,
            neginf=self.eps
        )

        # Physically valid domain: mu > 0
        x = torch.clamp(x, min=self.eps, max=1.0)

        # Single-scattering albedo must be in (0,1)
        w = torch.nan_to_num(
            w,
            nan=0.5,
            posinf=1.0 - self.eps,
            neginf=self.eps
        )
        w = torch.clamp(w, min=self.eps, max=1.0 - self.eps)

        # Hapke H-function parameters

        w_safe = w.clamp(min=0.0, max=1.0 - self.eps)

        y = torch.sqrt(1.0 - w_safe)
        r0 = (1.0 - y) / (1.0 + y)

        # ---------- CRITICAL FIX ----------
        # Prevent division by zero or Inf BEFORE it happens
        log_arg = (1.0 + x) / torch.clamp(x, min=self.eps)

        # Bound log argument to keep backward stable
        log_arg = torch.clamp(log_arg, min=1.0, max=1e6)

        log_val = torch.log(log_arg)

        bracket = r0 + (1.0 - 0.5 * r0 - x * r0) * log_val

        # Compute denominator
        denom = 1.0 - (1.0 - y) * x * bracket

        # ---- CRITICAL FIX ----
        # Sanitize and clamp BEFORE division
        denom = torch.nan_to_num(
            denom,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )

        denom = torch.clamp(denom, min=1e-6)

        # Safe reciprocal
        H = 1.0 / denom

        # Final safety (cheap, guarantees cleanliness)
        H = torch.nan_to_num(
            H,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )
        H = H.clamp(0.0, 10.0)

        return H


    # ------------------ Opposition effect ------------------

    def _B(self, g, B0, h): # g in radians
        g = torch.clamp(g, 0.0, math.pi - 1e-4)
        tanh = tan_safe(0.5 * g)
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
        Blackwell-safe computation of azimuth difference ψ from i, e, g
        using Hapke's relation:

            cos g = cos i cos e + sin i sin e cos ψ
        """

        # Ensure tensors (device/dtype assumed correct upstream)
        ci = torch.cos(i_rad)
        ce = torch.cos(e_rad)
        si = torch.sin(i_rad)
        se = torch.sin(e_rad)

        denom = si * se

        # Identify degenerate pixels (i≈0 or e≈0)
        valid = denom.abs() >= self.eps

        # Allocate cos_psi safely (no garbage values)
        cos_psi = torch.zeros_like(denom)

        # --- SAFE computation only on valid pixels ---
        denom_valid = denom[valid]
        numer_valid = torch.cos(g_rad[valid]) - ci[valid] * ce[valid]

        cos_psi_valid = numer_valid / denom_valid

        # Clamp BEFORE acos (mandatory on Blackwell)
        cos_psi_valid = torch.clamp(
            cos_psi_valid,
            min=-1.0 + self.eps,
            max= 1.0 - self.eps
        )

        cos_psi[valid] = cos_psi_valid

        # --- Compute psi ---
        psi = torch.zeros_like(cos_psi)
        psi[valid] = torch.acos(cos_psi[valid])

        # Degenerate case: i=0 or e=0 ⇒ ψ = 0 by definition
        # (psi already zero there)

        # Final safety net (should be no-ops now, but essential on Blackwell)
        psi = torch.nan_to_num(
            psi,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

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

        mu0 = torch.nan_to_num(mu0, nan=0.0).clamp(0.0, 1.0)
        mu  = torch.nan_to_num(mu,  nan=0.0).clamp(0.0, 1.0)

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
        w = torch.nan_to_num(w, nan=0.5, posinf=1.0, neginf=0.0)
        w = w.clamp(0.0, 1.0 - self.eps)

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
        mu0_eff, mu_eff, S = hapke_roughness(mu0, mu, i_rad, e_rad, psi_rad, theta_bar_rad, debug=self.debug, big_num=self.big_num, eps=self.eps)

        mu0_eff = torch.nan_to_num(mu0_eff, nan=0.0).clamp(0.0, 1.0)
        mu_eff  = torch.nan_to_num(mu_eff,  nan=0.0).clamp(0.0, 1.0)

        S = torch.nan_to_num(S, nan=0.0).clamp(0.0, 1.0)


        if self.debug:
            # Debug efter roughness
            self._check_nan(mu0_eff, "mu0_eff (after hapke_roughness)")
            self._check_nan(mu_eff,  "mu_eff (after hapke_roughness)")
            self._check_nan(S,       "S (roughness shadowing)")

        denom = mu0_eff + mu_eff
        denom = torch.nan_to_num(denom, nan=1e-6)
        denom = denom.clamp(min=1e-6, max=2.0)

        # Phase + opposition effects
        P = self._P(g_rad, **phase_params)
        B = self._B(g_rad, B0, h)


        P = torch.nan_to_num(P, nan=0.0, posinf=self.big_num, neginf=0.0)
        P = P.clamp(0.0, self.big_num)

        B = torch.nan_to_num(B, nan=0.0, posinf=self.big_num, neginf=0.0)
        B = B.clamp(0.0, self.big_num)


        if self.debug:
            # Debug efter fase + opposition
            self._check_nan(P,     "P (phase function)")
            self._check_nan(B,  "B (opposition)")


        # H-functions use roughened incident/emission cosines
        H0 = self._H(mu0_eff, w)
        H  = self._H(mu_eff,  w)

        # --------- Bi-directional reflectance r(i,e,g) ---------
        denom_safe = denom.clamp(min=self.eps)
        mu0_safe = mu0_eff.clamp(min=0)

        term = (1 + B) * P + H0 * H - 1
        term = torch.nan_to_num(term, nan=0.0, posinf=self.big_num, neginf=-self.big_num)
        term = term.clamp(-5, 10.0)


        ratio = mu0_safe / denom_safe
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=self.big_num, neginf=0.0)
        ratio = ratio.clamp(0.0, 10.0)


        scale = w / (4 * math.pi)
        scale = torch.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=0.0)

        prod = ratio * term
        prod = torch.nan_to_num(prod, nan=0.0, posinf=self.big_num, neginf=-self.big_num)
        prod = prod.clamp(-self.big_num, self.big_num)

        r = scale * prod

        # r = (w / (4 * math.pi)) * (ratio) * term

        # Apply roughness shadowing
        r = S * r

        r = torch.nan_to_num(r, nan=0.0, posinf=self.big_num, neginf=0.0)
        r = r.clamp(0.0, self.big_num)


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