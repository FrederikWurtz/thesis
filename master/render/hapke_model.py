import numpy as np
import torch

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
        gamma = torch.sqrt(torch.tensor(1.0 - self.w, dtype=torch.float32))
        return (1.0 + 2.0 * mu) / (1.0 + 2.0 * gamma * mu + 1e-12)

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
        return self.B0 / (1.0 + (1.0/self.h) * torch.tan(0.5*g_rad) + 1e-12)

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
            return (1 - self.xi**2) / torch.pow(1 + 2*self.xi*cg + self.xi**2, 1.5)
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
        P = self._P_phase(g_rad)      # Phase function
        B = self._B_SH(g_rad)         # Opposition effect
        H0 = self._H_function(mu0c)   # H-function for incidence
        H = self._H_function(muc)     # H-function for emission

        # Hapke reflectance equation
        r = (self.w / (4.0*torch.pi)) * (mu0c / denom) * ((1 + B) * P + H0*H - 1.0)
        R = torch.pi * r  # Convert to radiance factor

        # Return R only where both mu0 and mu are positive, else 0
        # Use torch.maximum to clamp negative values to 0 (equivalent to np.maximum)
        return torch.where((mu0 > 0) & (mu > 0), torch.maximum(R, torch.tensor(0.0)), torch.tensor(0.0))
