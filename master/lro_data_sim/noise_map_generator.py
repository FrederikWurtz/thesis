import math
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import os
from tqdm import tqdm
import torch


from master.configs.config_utils import load_config_file
from master.lro_data_sim.lro_generator_multi_band import generate_and_return_lro_data_multi_band


# -----------------------------
# Device helper
# -----------------------------
def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# -----------------------------
# Perlin/fBm på GPU (MPS/CUDA/CPU)
# -----------------------------
def perlin_tile_torch(xx, yy, hash_seed=0):
    """
    Core Perlin-noise beregning.
    xx, yy: float32 koordinater (h, w) på samme device.
    Returnerer: torch.Tensor (h, w) på samme device.
    """
    x0 = torch.floor(xx).to(torch.int64)
    y0 = torch.floor(yy).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    xf = xx - x0
    yf = yy - y0

    def fade(t):
        # Klassisk Perlin fade-funktion
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def random_grad(ix, iy):
        # Deterministisk hash → vinkel → enhedsvektor
        h = (ix * 374761393 + iy * 668265263 + hash_seed * 911382323) & 0xFFFFFFFF
        h = h.to(torch.int64)
        theta = (h % 36000).to(torch.float32) * (np.pi / 18000.0)
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (..., 2)

    def dot(ix, iy, x, y):
        g = random_grad(ix, iy)
        return g[..., 0] * (x - ix.to(torch.float32)) + g[..., 1] * (y - iy.to(torch.float32))

    n00 = dot(x0, y0, xx, yy)
    n10 = dot(x1, y0, xx, yy)
    n01 = dot(x0, y1, xx, yy)
    n11 = dot(x1, y1, xx, yy)

    u = fade(xf)
    v = fade(yf)

    x1_ = n00 * (1 - u) + n10 * u
    x2_ = n01 * (1 - u) + n11 * u
    out = x1_ * (1 - v) + x2_ * v
    return out


def fBm_tile_gpu(shape, gx, gy, base_scale,
                 octaves=4, lacunarity=2.0, gain=0.5,
                 hash_seed=0, device=None):
    """
    fBm (fractal Brownian motion) for én tile.
    shape: (h, w)
    gx, gy: globale pixel offsets
    base_scale: basis-skala i "noise space"
    hash_seed: hver værdi giver et uafhængigt noise-felt (fx 0 for W, 1 for theta)
    device: torch.device(...)
    Returnerer: NumPy array float32 (h, w)
    """
    if device is None:
        device = get_best_device()

    h, w = shape

    # Globale basis-koordinater (uden scale)
    ys_base = (torch.arange(h, device=device, dtype=torch.float32) + gy)
    xs_base = (torch.arange(w, device=device, dtype=torch.float32) + gx)
    yy_base, xx_base = torch.meshgrid(ys_base, xs_base, indexing='ij')

    noise = torch.zeros((h, w), device=device, dtype=torch.float32)

    scale = base_scale
    amp = 1.0
    for _ in range(octaves):
        yy = yy_base * scale
        xx = xx_base * scale
        noise += amp * perlin_tile_torch(xx, yy, hash_seed=hash_seed)
        scale *= lacunarity
        amp *= gain

    return noise.detach().cpu().numpy().astype(np.float32)



def gaussian_field_tile(shape, corr_len_px, device=None):
    """
    Isotropisk Gaussian Random Field tile vha. FFT.

    shape        : (h, w)
    corr_len_px  : korrelationslængde i pixels (størrelse af 'klumper')
    device       : torch.device("mps"/"cuda"/"cpu")

    Returnerer: np.ndarray float32 (h, w) med ca. range [-1, 1].
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    h, w = shape

    # Frekvensgrid
    ky = torch.fft.fftfreq(h, d=1.0).to(device)
    kx = torch.fft.fftfreq(w, d=1.0).to(device)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')

    k2 = KX**2 + KY**2

    # Isotropisk Gaussisk power spectrum
    # corr_len_px styrer hvor hurtigt P(k) falder.
    power = torch.exp(-k2 * (corr_len_px**2))

    # Kompleks hvid støj
    noise_real = torch.randn(h, w, device=device)
    noise_imag = torch.randn(h, w, device=device)
    noise_fft = noise_real + 1j * noise_imag

    # Filtrer i k-space
    field_fft = noise_fft * power

    # Tilbage til rumdomænet
    field = torch.fft.ifft2(field_fft).real

    # Normalisér til ca. [-1, 1]
    field = field / (torch.max(torch.abs(field)) + 1e-12)

    return field.detach().cpu().numpy().astype(np.float32)


def random_blob_field_tile(shape, blob_radius_px, density=0.001, device=None, seed=None):
    """
    Generér et 'random blob field' på en tile.

    shape        : (h, w)
    blob_radius_px : ca. radius (i pixels) for hver blob (styrer hvor brede patches er)
    density      : sandsynlighed for at hver pixel er et blob-center (0..1)
    device       : torch.device("mps"/"cuda"/"cpu")
    seed         : valgfri, hvis du vil have deterministisk output for en given tile

    Returnerer: np.ndarray float32 (h, w) med ca. range [-1, 1]
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    h, w = shape

    if seed is not None:
        # simpelt: seeddet globalt; for mere avanceret kan du kombinere seed med (x0,y0)
        torch.manual_seed(seed)

    # 1) Binært 'center'-felt: hvor ligger blobs?
    centers = (torch.rand(h, w, device=device) < density).float()

    # 2) Lav en Gaussisk kernel i samme size (cirkulær) til at "splash" blobs ud
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = (blob_radius_px ** 2)
    kernel = torch.exp(-0.5 * r2 / (sigma2 + 1e-12))

    # 3) Convolution via FFT (cirkulær convolution)
    centers_fft = torch.fft.fft2(centers)
    kernel_fft = torch.fft.fft2(kernel)
    field_fft = centers_fft * kernel_fft
    field = torch.fft.ifft2(field_fft).real

    # 4) Normalisér til [-1, 1]
    field = field - field.mean()
    field = field / (field.abs().max() + 1e-12)

    return field.detach().cpu().numpy().astype(np.float32)

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import torch

# antag disse findes:
# get_best_device(), fBm_tile_gpu(), gaussian_field_tile(), random_blob_field_tile()

import torch
import torch.nn.functional as F
import numpy as np

def upscale_field(field_low: np.ndarray, H: int, W: int, device=None) -> np.ndarray:
    """
    Opskalér et lavopløsnings felt (H_low, W_low) til (H, W) med bicubic interpolation.
    field_low: np.ndarray, shape (H_low, W_low), float32
    Returnerer np.ndarray (H, W), float32
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    t = torch.from_numpy(field_low).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H_low,W_low)
    t_up = F.interpolate(t, size=(H, W), mode="bicubic", align_corners=False)
    return t_up.squeeze().cpu().numpy().astype(np.float32)

def generate_noise_bands(
    dem_path,
    new_dem_path,
    tile=2048,
    range_w=(0.7, 1.0),          # output-range for W-båndet
    range_theta=(0.0, 40.0),     # output-range for Theta-båndet (fx grader)
    noise_type="fbm",            # "fbm", "gaussian" eller "random_blob"
    fbm_params_w=None,           # {"base_scale":..., "octaves":..., "lacunarity":..., "gain":..., "hash_seed":...}
    fbm_params_theta=None,
    gauss_params_w=None,         # {"corr_len_px":...}
    gauss_params_theta=None,
    random_blob_params_w=None,   # {"blob_radius_px":..., "density":..., "seed":...}
    random_blob_params_theta=None,
    global_downsample_factor=8,  # <--- NY
    custom_H_W=None,
):
    """
    Generér W- og Theta-bånd og skriv dem sammen med DEM til ny GeoTIFF.

    noise_type:
      - "fbm":       fBm_tile_gpu (Perlin-baseret fBm, global konsistens via hash)
      - "gaussian":  gaussian_field_tile (globalt Gaussian Random Field)
      - "random_blob": random_blob_field_tile (globalt random blob field)

    fbm_params_*:
      - base_scale (float)
      - octaves (int, default 4)
      - lacunarity (float, default 2.0)
      - gain (float, default 0.5)
      - hash_seed (int)

    gauss_params_*:
      - corr_len_px (float): korrelationslængde i pixels (størrelse af klumper)

    random_blob_params_*:
      - blob_radius_px (float): radius i pixels
      - density (float, default 0.001): blob-centre pr. pixel
      - seed (int, optional): random seed for global felt
    """

    device = get_best_device()
    print("Using device:", device)

    # --- Læs DEM ---
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        crs = src.crs
        transform = src.transform

    if custom_H_W is not None:
        H, W = custom_H_W
        dem = dem[0:H, 0:W]
    else:
        H, W = dem.shape

    print(f"DEM shape: {dem.shape}")

    dtype_out = "float32"
    dem_out = dem.astype(np.float32)

    # Sanity range
    assert range_w[0] is not None and range_w[1] is not None, "range_w must be (min,max)"
    assert range_theta[0] is not None and range_theta[1] is not None, "range_theta must be (min,max)"

    # ----- Parametre pr. noise-type -----
    if noise_type == "fbm":
        if fbm_params_w is None or fbm_params_theta is None:
            raise ValueError("fbm_params_w og fbm_params_theta skal sættes når noise_type='fbm'")

        # defaults
        fbm_params_w.setdefault("octaves", 4)
        fbm_params_w.setdefault("lacunarity", 2.0)
        fbm_params_w.setdefault("gain", 0.5)
        fbm_params_w.setdefault("hash_seed", 0)

        fbm_params_theta.setdefault("octaves", 4)
        fbm_params_theta.setdefault("lacunarity", 2.0)
        fbm_params_theta.setdefault("gain", 0.5)
        fbm_params_theta.setdefault("hash_seed", 1)

    elif noise_type == "gaussian":
        if gauss_params_w is None or "corr_len_px" not in gauss_params_w:
            raise ValueError("gauss_params_w['corr_len_px'] skal sættes når noise_type='gaussian'")
        if gauss_params_theta is None or "corr_len_px" not in gauss_params_theta:
            raise ValueError("gauss_params_theta['corr_len_px'] skal sættes når noise_type='gaussian'")

    elif noise_type == "random_blob":
        if random_blob_params_w is None or "blob_radius_px" not in random_blob_params_w:
            raise ValueError("random_blob_params_w['blob_radius_px'] skal sættes når noise_type='random_blob'")
        if random_blob_params_theta is None or "blob_radius_px" not in random_blob_params_theta:
            raise ValueError("random_blob_params_theta['blob_radius_px'] skal sættes når noise_type='random_blob'")
        # defaults
        random_blob_params_w.setdefault("density", 0.001)
        random_blob_params_w.setdefault("seed", 0)
        random_blob_params_theta.setdefault("density", 0.001)
        random_blob_params_theta.setdefault("seed", 1)

    else:
        raise ValueError(f"Ukendt noise_type: {noise_type}")

    # ---------------------------------
    # CASE 1: fBm → tile-baseret generering (hash-baseret global konsistens)
    # ---------------------------------
    if noise_type == "fbm":
        # PASS 1: global min/max via tiles
        global_min_w = float('inf')
        global_max_w = float('-inf')
        global_min_theta = float('inf')
        global_max_theta = float('-inf')

        n_tiles_total = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)

        print("Pass 1 (fBm): scanner global min/max for W og Theta ...")
        with tqdm(total=n_tiles_total, desc="Pass 1 (stats)", unit="tile") as pbar:
            for y0 in range(0, H, tile):
                for x0 in range(0, W, tile):
                    h = min(tile, H - y0)
                    w_ = min(tile, W - x0)

                    nw = fBm_tile_gpu(
                        (h, w_), x0, y0,
                        base_scale=fbm_params_w["base_scale"],
                        octaves=fbm_params_w["octaves"],
                        lacunarity=fbm_params_w["lacunarity"],
                        gain=fbm_params_w["gain"],
                        hash_seed=fbm_params_w["hash_seed"],
                        device=device,
                    )
                    nt = fBm_tile_gpu(
                        (h, w_), x0, y0,
                        base_scale=fbm_params_theta["base_scale"],
                        octaves=fbm_params_theta["octaves"],
                        lacunarity=fbm_params_theta["lacunarity"],
                        gain=fbm_params_theta["gain"],
                        hash_seed=fbm_params_theta["hash_seed"],
                        device=device,
                    )

                    global_min_w = min(global_min_w, float(nw.min()))
                    global_max_w = max(global_max_w, float(nw.max()))
                    global_min_theta = min(global_min_theta, float(nt.min()))
                    global_max_theta = max(global_max_theta, float(nt.max()))

                    pbar.set_postfix({
                        "W_min": f"{global_min_w:.3f}",
                        "W_max": f"{global_max_w:.3f}",
                        "T_min": f"{global_min_theta:.3f}",
                        "T_max": f"{global_max_theta:.3f}",
                    })
                    pbar.update(1)

        print(f"Global W min/max: {global_min_w:.6f}, {global_max_w:.6f}")
        print(f"Global Theta min/max: {global_min_theta:.6f}, {global_max_theta:.6f}")

        # PASS 2: skriv til GeoTIFF
        if os.path.exists(new_dem_path):
            print(f"{new_dem_path} findes allerede. Sletter og laver ny.")
            os.remove(new_dem_path)

        with rasterio.open(
            new_dem_path,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=3,
            dtype=dtype_out,
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=tile,
            blockysize=tile,
            compress='deflate',
            BIGTIFF='YES',
        ) as dst:
            print(f"Creating new GeoTIFF with 3 bands at {new_dem_path}...")
            dst.write(dem_out[0:H, 0:W], 1)
            print("First band (DEM) written.")

            with tqdm(total=n_tiles_total, desc="Pass 2 (write fBm)", unit="tile") as pbar:
                for y0 in range(0, H, tile):
                    for x0 in range(0, W, tile):
                        h = min(tile, H - y0)
                        w_ = min(tile, W - x0)

                        pbar.set_postfix({"tile": f"y:{y0}-{y0+h}, x:{x0}-{x0+w_}"})

                        nw = fBm_tile_gpu(
                            (h, w_), x0, y0,
                            base_scale=fbm_params_w["base_scale"],
                            octaves=fbm_params_w["octaves"],
                            lacunarity=fbm_params_w["lacunarity"],
                            gain=fbm_params_w["gain"],
                            hash_seed=fbm_params_w["hash_seed"],
                            device=device,
                        )
                        nt = fBm_tile_gpu(
                            (h, w_), x0, y0,
                            base_scale=fbm_params_theta["base_scale"],
                            octaves=fbm_params_theta["octaves"],
                            lacunarity=fbm_params_theta["lacunarity"],
                            gain=fbm_params_theta["gain"],
                            hash_seed=fbm_params_theta["hash_seed"],
                            device=device,
                        )

                        # global scaling
                        nw_norm = (nw - global_min_w) / (global_max_w - global_min_w + 1e-12)
                        nt_norm = (nt - global_min_theta) / (global_max_theta - global_min_theta + 1e-12)

                        nw_scaled = nw_norm * (range_w[1] - range_w[0]) + range_w[0]
                        nt_scaled = nt_norm * (range_theta[1] - range_theta[0]) + range_theta[0]

                        window = Window(x0, y0, w_, h)
                        dst.write(nw_scaled.astype(dtype_out), 2, window=window)
                        dst.write(nt_scaled.astype(dtype_out), 3, window=window)

                        pbar.update(1)

        print(f"✓ New DEM with W + Theta (fBm) saved to {new_dem_path}")
        return

    # CASE 2: Gaussian / random_blob → global felter i nedskaleret grid
    if noise_type in ("gaussian", "random_blob"):
        print(f"Generating global {noise_type} fields on downsampled grid (factor={global_downsample_factor})...")

        # Nedskaleret grid
        H_low = int(np.ceil(H / global_downsample_factor))
        W_low = int(np.ceil(W / global_downsample_factor))
        print(f"Low-res grid for global field: {H_low} x {W_low}")

        if noise_type == "gaussian":
            # juster korrelationslængde til lavere opløsning
            corr_w_low = gauss_params_w["corr_len_px"] / global_downsample_factor
            corr_t_low = gauss_params_theta["corr_len_px"] / global_downsample_factor

            global_w_low = gaussian_field_tile(
                (H_low, W_low),
                corr_len_px=corr_w_low,
                device=device,
            )
            global_theta_low = gaussian_field_tile(
                (H_low, W_low),
                corr_len_px=corr_t_low,
                device=device,
            )

        elif noise_type == "random_blob":
            # juster blob radius til lavere opløsning
            rad_w_low = random_blob_params_w["blob_radius_px"] / global_downsample_factor
            rad_t_low = random_blob_params_theta["blob_radius_px"] / global_downsample_factor

            global_w_low = random_blob_field_tile(
                (H_low, W_low),
                blob_radius_px=rad_w_low,
                density=random_blob_params_w["density"],
                device=device,
                seed=random_blob_params_w["seed"],
            )
            global_theta_low = random_blob_field_tile(
                (H_low, W_low),
                blob_radius_px=rad_t_low,
                density=random_blob_params_theta["density"],
                device=device,
                seed=random_blob_params_theta["seed"],
            )

        # Opskalér til fuld størrelse
        global_w = upscale_field(global_w_low, H, W, device=device)
        global_theta = upscale_field(global_theta_low, H, W, device=device)

        # global stats + scaling (som du allerede gør)
        global_min_w, global_max_w = float(global_w.min()), float(global_w.max())
        global_min_theta, global_max_theta = float(global_theta.min()), float(global_theta.max())

        print(f"Global W min/max: {global_min_w:.6f}, {global_max_w:.6f}")
        print(f"Global Theta min/max: {global_min_theta:.6f}, {global_max_theta:.6f}")

        global_w_norm = (global_w - global_min_w) / (global_max_w - global_min_w + 1e-12)
        global_theta_norm = (global_theta - global_min_theta) / (global_max_theta - global_min_theta + 1e-12)

        global_w_scaled = global_w_norm * (range_w[1] - range_w[0]) + range_w[0]
        global_theta_scaled = global_theta_norm * (range_theta[1] - range_theta[0]) + range_theta[0]

        # Skriv i tiles (uændret logik)
        n_tiles_total = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)

        if os.path.exists(new_dem_path):
            print(f"{new_dem_path} findes allerede. Sletter og laver ny.")
            os.remove(new_dem_path)

        with rasterio.open(
            new_dem_path,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=3,
            dtype=dtype_out,
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=tile,
            blockysize=tile,
            compress='deflate',
            BIGTIFF='YES',
        ) as dst:
            print(f"Creating new GeoTIFF with 3 bands at {new_dem_path}...")
            dst.write(dem_out[0:H, 0:W], 1)
            print("First band (DEM) written.")

            from rasterio.windows import Window
            with tqdm(total=n_tiles_total, desc=f"Write {noise_type} fields", unit="tile") as pbar:
                for y0 in range(0, H, tile):
                    for x0 in range(0, W, tile):
                        h = min(tile, H - y0)
                        w_ = min(tile, W - x0)

                        pbar.set_postfix({"tile": f"y:{y0}-{y0+h}, x:{x0}-{x0+w_}"})

                        nw_tile = global_w_scaled[y0:y0+h, x0:x0+w_]
                        nt_tile = global_theta_scaled[y0:y0+h, x0:x0+w_]

                        window = Window(x0, y0, w_, h)
                        dst.write(nw_tile.astype(dtype_out), 2, window=window)
                        dst.write(nt_tile.astype(dtype_out), 3, window=window)

                        pbar.update(1)

        print(f"✓ New DEM with W + Theta ({noise_type}) saved to {new_dem_path}")
        return


# doesnt work - tomorrow use this tile-wise upscaling function

import torch
import torch.nn.functional as F
import numpy as np

def upsample_lowres_tile_to_highres(
    field_low,       # np.ndarray (H_lr, W_lr)
    factor,          # int, fx 8
    y0, x0, h, w,    # tile-koordinater i high-res
    device=None,
    mode="bicubic"
):
    """
    Sampl en (h, w)-tile i high-res fra et globalt low-res field vha. grid_sample.
    field_low: globalt low-res field (H_lr, W_lr)
    factor: downsample faktor (H ≈ H_lr*factor, W ≈ W_lr*factor)
    return: np.ndarray (h, w) float32
    """
    H_lr, W_lr = field_low.shape

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Low-res tensor på device
    t_lr = torch.from_numpy(field_low).to(device=device, dtype=torch.float32)[None, None, ...]  # (1,1,H_lr,W_lr)

    # High-res pixel-koordinater for denne tile
    ys_hr = np.linspace(y0, y0 + h - 1, h, dtype=np.float32)
    xs_hr = np.linspace(x0, x0 + w - 1, w, dtype=np.float32)

    # Map til low-res "continuous" coords
    ys_lr = ys_hr / float(factor)
    xs_lr = xs_hr / float(factor)

    # Normaliser til [-1,1] i grid_sample-koordinater
    ys_norm = ys_lr / max(H_lr - 1, 1) * 2 - 1
    xs_norm = xs_lr / max(W_lr - 1, 1) * 2 - 1

    yy, xx = np.meshgrid(ys_norm, xs_norm, indexing='ij')
    grid = torch.from_numpy(
        np.stack([xx, yy], axis=-1)  # (h, w, 2), note (x,y) rækkefølge
    ).to(device=device, dtype=torch.float32)[None, ...]  # (1,h,w,2)

    t_up = F.grid_sample(
        t_lr, grid,
        mode=mode,
        align_corners=True
    )  # (1,1,h,w)

    tile_hr = t_up[0, 0].detach().cpu().numpy().astype(np.float32)
    return tile_hr



# -----------------------------
# Eksempel på brug
# -----------------------------
if __name__ == "__main__":
    dem_path = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    new_dem_path = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014_with_bands_random_blob.tif"

      # Brug mindre størrelse til test/visualisering
    custom_H_W = (10000, 10000)

    # remove existing file for testing
    if os.path.exists(new_dem_path):
        print(f"Removing existing file {new_dem_path} for testing...")
        os.remove(new_dem_path)

    if not os.path.exists(new_dem_path):
        # generate_noise_bands(
        #     dem_path=dem_path,
        #     new_dem_path=new_dem_path,
        #     tile=2048,
        #     noise_type="gaussian",
        #     range_w=(0.1, 0.3),
        #     range_theta=(math.radians(0.0), math.radians(40.0)),
        #     gauss_params_w={
        #         "corr_len_px": 850,   # ~100 km hvis 118 m/px
        #     },
        #     gauss_params_theta={
        #         "corr_len_px": 170,   # ~20 km
        #     },
        #     custom_H_W=custom_H_W
        # )
        # generate_noise_bands(
        #     dem_path=".../Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
        #     new_dem_path=".../Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014_fbm.tif",
        #     tile=2048,
        #     noise_type="fbm",
        #     range_w=(0.1, 0.3),
        #     range_theta=(math.radians(0.0), math.radians(40.0)),
        #     fbm_params_w={
        #         "base_scale": 1.0 / 8000.0,  # store albedo-strukturer
        #         "octaves": 4,
        #         "lacunarity": 2.0,
        #         "gain": 0.5,
        #         "hash_seed": 0,
        #     },
        #     fbm_params_theta={
        #         "base_scale": 1.0 / 3000.0,  # lidt finere roughness
        #         "octaves": 4,
        #         "lacunarity": 2.0,
        #         "gain": 0.5,
        #         "hash_seed": 1,
        #     },
        # )
        generate_noise_bands(
            dem_path=dem_path,
            new_dem_path=new_dem_path,
            tile=2048,
            noise_type="random_blob",
            range_w=(0.1, 0.3),
            range_theta=(math.radians(0.0), math.radians(40.0)),
            random_blob_params_w={
                "blob_radius_px": 20,   # ~60 km hvis 118 m/px
                "density": 0.01,
                "seed": 0,
            },
            random_blob_params_theta={
                "blob_radius_px": 20,   # ~12 km
                "density": 0.01,
                "seed": 1,
            },
            global_downsample_factor=8         
            )
    else:
        print(f"File {new_dem_path} already exists. Skipping creation.")


    # #make sure figures directory exists
    # os.makedirs("master/lro_data_sim/figures", exist_ok=True)

    # # read back the generated bands for visualization
    # with rasterio.open(new_dem_path) as src:
    #     H, W = src.height, src.width
    #     band_w = src.read(2, window=rasterio.windows.Window(0, 0, W, H))
    #     band_theta = src.read(3, window=rasterio.windows.Window(0, 0, W, H))
    
    # # generate and return lro multi band data for testing
    # # config = load_config_file()
    # # images, reflectance_maps, dem_tensor, metas, w_tensor, theta_bar_tensor, lro_meta = generate_and_return_lro_data_multi_band(config, device=device)

    # # save different subsets of the w band
    # for size in (100, 1000, 5000, 10000):
    #     # print different statistics
    #     print(f"W band subset {size}x{size} stats: min={band_w[0:size, 0:size].min()}, max={band_w[0:size, 0:size].max()}, mean={band_w[0:size, 0:size].mean()}")
    #     subset = band_w[0:size, 0:size]
    #     subset_path = f"master/lro_data_sim/figures/w_band_subset_{size}x{size}.png"
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(subset, cmap='gray')
    #     plt.title(f"W Band Subset {size}x{size}")
    #     plt.colorbar()
    #     plt.savefig(subset_path)
    #     plt.close()
    #     print(f"W band subset saved to {subset_path}")
    
    # for size in (100, 1000, 5000, 10000):
    #     print(f"Theta band subset {size}x{size} stats: min={band_theta[0:size, 0:size].min()}, max={band_theta[0:size, 0:size].max()}, mean={band_theta[0:size, 0:size].mean()}")
    #     subset = band_theta[0:size, 0:size]
    #     subset_path = f"master/lro_data_sim/figures/theta_band_subset_{size}x{size}.png"
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(subset, cmap='gray')
    #     plt.title(f"Theta Band Subset {size}x{size}")
    #     plt.colorbar()
    #     plt.savefig(subset_path)
    #     plt.close()
    #     print(f"Theta band subset saved to {subset_path}")

    # splatter_map = create_splatter_band(10000, 10000, n_dots=100000, dot_radius=20)
    # plt.imshow(splatter_map, cmap='gray')
    # plt.title("Splatter Map")
    # plt.colorbar()
    # save_dir = "master/lro_data_sim/figures/"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "splatter_map.png")
    # plt.savefig(save_path)
    # plt.close()
    # print(f"Splatter map saved to {save_path}")

    # # also plot zommed in version of this
    # zoomed_in = splatter_map[4500:5500, 4500:5500]
    # plt.imshow(zoomed_in, cmap='gray')
    # plt.title("Zoomed-in Splatter Map")
    # plt.colorbar()
    # zoomed_save_path = os.path.join(save_dir, "splatter_map_zoomed.png")
    # plt.savefig(zoomed_save_path)
    # plt.close()
    # print(f"Zoomed-in splatter map saved to {zoomed_save_path}")

