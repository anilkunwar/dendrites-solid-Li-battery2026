"""
Li dendrite growth in lithium-ion batteries with polymer solid electrolytes
============================================================================
Spectral phase-field simulator with Butler-Volmer electrochemistry,
Hart effective diffusion, and elastoplastic mechanics.

Adapted from the original Na-ion simulator of Wu et al., J. Energy
Storage 183 (2027) 125085. All material data, variable names, dict
keys, plot labels, and comments in this codebase now describe
Li-metal electrodes paired with polymer solid electrolytes
(PEO/LiTFSI, PAN, PPC, PVDF); no Na-ion material data is used by the
Li solver code path.

Domain (left -> right):
    |<--- Li metal electrode --->|<--- polymer solid electrolyte (Voronoi grains) --->|
    eta = 1                          eta = 0, phi_g in [0,1] (grain indicator)

Physics (Wu et al., adapted for Li-polymer systems):
    1. Allen-Cahn evolution of Li/SE order parameter eta (Eq. 7)
       - Double-well + grain-boundary coupling
       - Butler-Volmer source term from electrodeposition
    2. Li+ reaction-diffusion (Eq. 8)
       - Hart effective diffusivity: D_eff = h(eta) D_Li + (1 - h(eta)) phi_g D_SE
       - Nernst-Planck flux with electrostatic drift
    3. Charge conservation (Eq. 9)
       - nabla . (sigma_eff nabla phi) = -c_Li * F * deta/dt
    4. Spectral elastoplasticity (plane-strain, J2 flow, KAM-GND density)
       - Provides sigma_h for stress-migration coupling
    5. Effective diffusion anisotropy
       - D_bulk (intragranular) vs D_gb (grain-boundary) -> preferential
         dendrite propagation along GBs.

Geometry options for SE:
    - random_voronoi : random seeds in electrolyte region (default)
    - columnar       : vertically elongated grains (moulded polymer)
    - spherical      : packed circular particles with internal GBs + voids

Initial-dendrite options:
    - flat_noise    : flat Li/SE interface with small random perturbation
    - preseed_tips  : pre-place Li protrusions at GB-electrode intersections
    - single_notch  : one rectangular notch centered on the interface

Material options (Li-ion polymer electrolytes):
    - PEO_Li   : PEO/LiTFSI (requires T > 60 C, soft, GB-dependent)
    - PAN      : Polyacrylonitrile (better stability, moderate stiffness)
    - PPC      : Polypropylene carbonate (amorphous, RT mobility, weak GB)
    - PVDF     : Polyvinylidene fluoride (very stiff, mechanically suppresses)
    - custom   : user-configurable D_bulk and D_gb/D_bulk ratio

Operating temperature is configurable in the sidebar (default 60 C). This is
critical for PEO/LiTFSI, whose ionic conductivity only becomes adequate
above ~60 C due to polymer chain segmental motion.

Author: AI assistant
"""

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import (Colormap, LinearSegmentedColormap,
                                ListedColormap, LogNorm)
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import plotly.graph_objects as go
from datetime import datetime
import json, zipfile, pickle, hashlib, traceback, warnings
from io import BytesIO, StringIO
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SENTINEL + ERROR HANDLING
# ============================================================================
class _Sentinel:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reason = None
        return cls._instance
    def __bool__(self): return False
    def __repr__(self): return f"<FAILED: {self.reason}>"


FAILED = _Sentinel()


def handle_errors(func=None, *, re_raise=False):
    def deco(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                FAILED.reason = f"{f.__name__}: {e}"
                logger.error(f"{e}\n{traceback.format_exc()}")
                try:
                    st.error(f"Error in {f.__name__}: {e}")
                except Exception:
                    pass
                if re_raise:
                    raise
                return FAILED
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper
    if func is None:
        return deco
    return deco(func)


def safe_call(result, default=None):
    """Return result if not FAILED, else default."""
    if result is FAILED or result is None:
        return default
    return result


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
F_FARADAY = 96485.33212       # C/mol
R_GAS      = 8.314462618      # J/(mol K)
T_ROOM     = 300.0            # K
NA_AVOGADRO = 6.02214076e23


# ============================================================================
# SPECTRAL OPERATORS (rectangular grid)
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    """Build wavenumber vectors for 2D FFT on a rectangular grid."""
    if dy is None:
        dy = dx
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx).reshape(1, -1)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy).reshape(-1, 1)
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0e-12  # avoid divide-by-zero in DC component
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    """Return (df/dx, df/dy) computed via FFT."""
    fh = fft2(field)
    return (np.real(ifft2(1j * kx * fh)),
            np.real(ifft2(1j * ky * fh)))


def spectral_laplacian(field, k2):
    return np.real(ifft2(-k2 * fft2(field)))


def spectral_divergence(fx, fy, kx, ky):
    return np.real(ifft2(1j * kx * fft2(fx) + 1j * ky * fft2(fy)))


# ============================================================================
# COLORMAPS
# ============================================================================
def _build_custom_colormaps():
    """Custom colormaps for Li-dendrite visualization."""
    li_conc = LinearSegmentedColormap.from_list('li_conc', [
        (0.00, '#0d1b2a'), (0.15, '#1b4f72'), (0.35, '#7d3c98'),
        (0.60, '#c39bd3'), (0.85, '#f5eef8'), (1.00, '#ffffff')])

    grain_map = ListedColormap([
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
        '#cedb9c', '#9c9ede', '#e7969c', '#e7ba52', '#e7cb94',
        '#de9ed6', '#adc6dc', '#bd9e39', '#c49c94', '#f7b6d2',
    ], name='grain_map')

    stress_pub = LinearSegmentedColormap.from_list('stress_pub', [
        (0.0, '#2c7bb6'), (0.25, '#abd9e9'), (0.5, '#ffffbf'),
        (0.75, '#fdae61'), (1.0, '#d7191c')])

    gb_hot = LinearSegmentedColormap.from_list('gb_hot', [
        (0.0, '#000004'), (0.3, '#420a68'), (0.6, '#932667'),
        (0.85, '#f8850f'), (1.0, '#fcffa4')])

    potential_map = LinearSegmentedColormap.from_list('potential', [
        (0.0, '#053061'), (0.3, '#2166ac'), (0.5, '#f7f7f7'),
        (0.7, '#b2182b'), (1.0, '#67001f')])

    return {'li_conc': li_conc, 'grain_map': grain_map,
            'stress_pub': stress_pub, 'gb_hot': gb_hot,
            'potential': potential_map}


_CUSTOM_COLORMAPS = _build_custom_colormaps()
for _n, _c in _CUSTOM_COLORMAPS.items():
    try:
        if _n not in mpl.colormaps:
            mpl.colormaps.register(_c, name=_n, force=True)
    except Exception as _e:
        logger.warning(f"Could not register colormap '{_n}': {_e}")


class ColormapRegistry:
    BASE = [
        'viridis', 'viridis_r', 'plasma', 'plasma_r',
        'inferno', 'inferno_r', 'magma', 'magma_r',
        'cividis', 'cividis_r', 'turbo', 'turbo_r',
        'Greys', 'Greys_r', 'Blues', 'Blues_r',
        'Greens', 'Greens_r', 'Reds', 'Reds_r',
        'Oranges', 'Oranges_r', 'Purples', 'Purples_r',
        'YlOrRd', 'YlOrRd_r', 'YlOrBr', 'YlOrBr_r',
        'YlGnBu', 'YlGnBu_r', 'YlGn', 'YlGn_r',
        'BuPu', 'BuPu_r', 'PuRd', 'PuRd_r',
        'BuGn', 'BuGn_r', 'PuBu', 'PuBu_r',
        'PuBuGn', 'PuBuGn_r', 'GnBu', 'GnBu_r',
        'OrRd', 'OrRd_r',
        'RdBu', 'RdBu_r', 'coolwarm', 'coolwarm_r',
        'bwr', 'bwr_r', 'seismic', 'seismic_r',
        'Spectral', 'Spectral_r',
        'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
        'BrBG', 'BrBG_r', 'PiYG', 'PiYG_r',
        'PRGn', 'PRGn_r', 'PuOr', 'PuOr_r',
        'RdGy', 'RdGy_r',
        'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
        'hsv', 'jet', 'jet_r', 'rainbow', 'rainbow_r',
        'nipy_spectral', 'nipy_spectral_r', 'gist_rainbow', 'gist_rainbow_r',
        'binary', 'binary_r', 'gray', 'gray_r',
        'bone', 'bone_r', 'copper', 'copper_r',
        'pink', 'pink_r', 'hot', 'hot_r',
        'afmhot', 'afmhot_r', 'gist_heat', 'gist_heat_r',
        'gist_gray', 'gist_gray_r', 'gist_yarg', 'gist_yarg_r',
        'flag', 'prism', 'ocean', 'ocean_r',
        'terrain', 'terrain_r', 'gist_earth', 'gist_earth_r',
        'gist_ncar', 'gist_ncar_r', 'gist_stern', 'gist_stern_r',
        'CMRmap', 'CMRmap_r', 'cubehelix', 'cubehelix_r',
        'brg', 'brg_r', 'cool', 'cool_r', 'warm', 'warm_r',
        'Wistia', 'Wistia_r',
        'spring', 'spring_r', 'summer', 'summer_r',
        'autumn', 'autumn_r', 'winter', 'winter_r',
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c',
    ]

    @classmethod
    def list_all(cls):
        names = list(cls.BASE) + list(_CUSTOM_COLORMAPS.keys())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    @classmethod
    def resolve(cls, name, invert=False):
        if name is None:
            cmap = mpl.colormaps['viridis']
        elif isinstance(name, Colormap):
            cmap = name
        else:
            key = str(name)
            if key in mpl.colormaps:
                cmap = mpl.colormaps[key]
            elif key in _CUSTOM_COLORMAPS:
                cmap = _CUSTOM_COLORMAPS[key]
            else:
                logger.warning(f"Unknown colormap '{name}', using viridis")
                cmap = mpl.colormaps['viridis']
        if invert:
            cmap = cmap.reversed()
        return cmap

    @classmethod
    def to_plotly(cls, name, n=64, invert=False):
        cmap = cls.resolve(name, invert=invert)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f'rgb({r},{g},{b})']
                for ti, (r, g, b) in zip(t, rgb)]


# ============================================================================
# FIELD CONFIGURATION
# ============================================================================
FIELD_DEFAULT_CMAPS = {
    'eta':         'li_conc',
    'c_li':       'plasma',
    'c_salt':     'cividis',
    'c_e':        'YlOrBr',
    'phi_g':      'gray',
    'grain_map':  'grain_map',
    'phi_gb':     'gb_hot',
    'phi_pot':    'potential',
    'D_eff':      'magma',
    'sigma_eq':   'hot',
    'sigma_h':    'RdBu_r',
    'h':          'cividis',
    'eps_p_mag':  'YlOrRd',
    'sigma_y':    'viridis',
    'rho_gnd':    'inferno',
    'phi_flux':   'autumn',
    'sei_damage': 'inferno',
}

FIELD_LABELS = {
    'eta':         'eta (Li metal order parameter)',
    'c_li':       'c_Li+ (Li+ ion concentration)',
    'c_salt':     'c_salt (LiTFSI salt concentration)',
    'c_e':        'c_e- (electron concentration)',
    'phi_g':      'phi_g (SE grain indicator)',
    'grain_map':  'Grain ID map (Voronoi)',
    'phi_gb':     'phi_gb (GB indicator)',
    'phi_pot':    'phi (electric potential, V)',
    'D_eff':      'D_eff (Hart effective, m^2/s)',
    'sigma_eq':   'sigma_eq (Von Mises, GPa)',
    'sigma_h':    'sigma_h (Hydrostatic, GPa)',
    'h':          'Local grain size (um)',
    'eps_p_mag':  '|eps_p| (plastic strain)',
    'sigma_y':    'sigma_y (yield stress, MPa)',
    'rho_gnd':    'rho_GND (GND density)',
    'phi_flux':   '|J_Li+| (Li+ flux magnitude)',
    'sei_damage': 'SEI damage (0=intact, 1=fractured)',
}

FIELD_UNITS = {
    'eta': '', 'c_li': '', 'c_salt': '', 'c_e': '', 'phi_g': '',
    'grain_map': '', 'phi_gb': '', 'phi_pot': 'V', 'D_eff': 'm^2/s',
    'sigma_eq': 'GPa', 'sigma_h': 'GPa', 'h': 'um',
    'eps_p_mag': '', 'sigma_y': 'MPa', 'rho_gnd': '',
    'phi_flux': '', 'sei_damage': '',
}

FIELD_DEFAULT_RNG = {
    'eta':         (0.0, 1.0),
    'c_li':       (0.0, 1.0),
    'c_salt':     (0.0, 1.0),
    'c_e':        (0.0, 1.0),
    'phi_g':      (0.0, 1.0),
    'phi_gb':     (0.0, 1.0),
    'phi_pot':    (-0.25, 0.05),
    'rho_gnd':    (0.0, 1.0),
    'h':          (0.0, 50.0),
    'sei_damage': (0.0, 1.0),
}

DISCRETE_FIELDS = {'grain_map'}
VECTOR_FIELD_SOURCES = ['eta', 'c_li', 'phi_pot', 'D_eff', 'phi_gb']


# ============================================================================
# MATERIAL PROPERTY SETS
# ============================================================================
class MaterialProperties:
    """Material parameters for Li metal and Li-polymer solid electrolytes.

    The material sets below are for Li-metal electrodes paired with
    Li-polymer solid electrolytes (PEO/LiTFSI, PAN, PPC, PVDF).  No
    Na-ion material data is included in this codebase; see the module
    docstring for the historical Na-ion reference.

    All values are in SI units:
        D_*       : m^2/s
        sigma_*   : S/m
        c0        : mol/m^3 (Li site concentration)
        c_surf    : mol/m^3 (surface electron concentration, approximated)
        E, nu     : Pa, dimensionless
        beta      : dimensionless (Li insertion misfit)
    """

    # All material sets below are for Li-metal electrodes paired
    # with Li-polymer solid electrolytes. The original Na-ion
    # material data from Wu et al. (J. Energy Storage 183 (2027)
    # 125085) has been removed; this codebase now contains only
    # Li-ion battery materials.
    # ====================================================================
    # LITHIUM METAL + LITHIUM-POLYMER ELECTROLYTES
    # --------------------------------------------------------------------
    # Li metal self-diffusion at room T (~1e-12 to 1e-14 m^2/s) is
    # actually FASTER than Na's (~1e-16 m^2/s).  The fast Li self-
    # diffusion tends to flatten the Li/electrolyte interface, so Li
    # dendrites are NOT caused by slow self-diffusion.  They are
    # caused by (a) low transference number t+ (~0.2-0.4 in PEO/LiTFSI)
    # which causes concentration polarization at the plating front,
    # and (b) the mechanically robust SEI on Li that cracks under
    # deposition pressure, forcing Li to extrude from the root in a
    # whisker / root-growth mode (rather than Na's particle mode
    # enabled by its weak SEI and fast surface diffusion).  Misfit
    # strains are smaller for Li because Li+ is smaller than Na+.
    # Polymer electrolytes differ widely in stiffness, GB dependence,
    # and ionic conductivity:
    #   - PEO   : semi-crystalline, requires >60 C, low stiffness
    #   - PAN   : better electrochemical stability, moderate stiffness
    #   - PPC   : highly amorphous, excellent RT mobility, weak GB dependence
    #   - PVDF  : very stiff, mechanically suppresses dendrites
    # ====================================================================
    
        # --- LITHIUM METAL ---
    LI_METAL = {
        'name': 'Li',
        'D_Li': 1.0e-13,            # m^2/s.  Realistic Li metal self-diffusion
                          # at room T is in the range ~1e-12 to 1e-14,
                          # which is FASTER than Na's ~1e-16 m^2/s (not
                          # slower, as a previous comment claimed).  The
                          # fast self-diffusion tends to flatten the
                          # interface, so Li dendrites are NOT produced
                          # by fudging this value; they are produced by
                          # (a) low transference number t+ which causes
                          # concentration polarization, and (b) the SEI
                          # mechanical layer cracking under stress,
                          # which forces Li to extrude from the root.
                          # See the SEI damage field and the t+ scaling
                          # in _evolve_c_li() for the actual physics.
        'sigma_Li': 1.1e7,          # S/m (electronic conductivity)
        'c0': 7.69e4,               # mol/m^3 (density 534 kg/m3 / 0.00694 kg/mol)
        'elastic': {'C11': 7.86e9, 'C12': 4.24e9, 'C44': 1.81e9,
                    'source': 'Li BCC, room T (E~4.9 GPa, nu~0.35)'},
        'plasticity': {'mu': 1.9e9, 'nu': 0.35, 'b': 0.371e-9,  # Li Burgers vector
                       'sigma0': 1.0e6, 'gamma0_dot': 1e-3, 'm': 20},
        'misfit_beta': 0.15,       # Li insertion misfit
    }

    # --- POLYMER ELECTROLYTES (Li-ion) ---
    PEO_LI = {
        'name': 'PEO/LiTFSI (60C)',
        'D_SE': 5.0e-13,            # m^2/s (requires >60 C for good mobility)
        'D_gb_ratio': 200.0,        # GB enhancement
        'sigma_SE': 1.0e-4,         # S/m (typical at 60 C)
        'c_surf': 5.0e4,            # mol/m^3
        'elastic': {'C11': 0.3e9, 'C12': 0.2e9, 'C44': 0.05e9,  # Soft polymer
                    'source': 'PEO at 60 C (E~0.1 GPa)'},
        'plasticity': {'mu': 0.05e9, 'nu': 0.45, 'b': 0.371e-9,
                       'sigma0': 2e6, 'gamma0_dot': 1e-3, 'm': 10},
        'misfit_beta': 0.10,
    }

    PAN_LI = {
        'name': 'PAN (Polyacrylonitrile)',
        'D_SE': 2.0e-12,            # m^2/s (good polarity, decent RT mobility)
        'D_gb_ratio': 300.0,
        'sigma_SE': 5.0e-4,         # S/m (higher electrochemical stability)
        'c_surf': 5.0e4,
        'elastic': {'C11': 1.5e9, 'C12': 0.8e9, 'C44': 0.35e9,  # Stiffer than PEO
                    'source': 'PAN (E~1.0 GPa)'},
        'plasticity': {'mu': 0.4e9, 'nu': 0.38, 'b': 0.371e-9,
                       'sigma0': 10e6, 'gamma0_dot': 1e-3, 'm': 15},
        'misfit_beta': 0.12,
    }

    PPC_LI = {
        'name': 'PPC (Polypropylene carbonate)',
        'D_SE': 1.0e-11,            # m^2/s (highly amorphous, excellent RT mobility)
        'D_gb_ratio': 100.0,        # Less GB dependence due to amorphous nature
        'sigma_SE': 1.5e-4,         # S/m
        'c_surf': 5.0e4,
        'elastic': {'C11': 0.8e9, 'C12': 0.4e9, 'C44': 0.2e9,
                    'source': 'PPC (E~0.5 GPa)'},
        'plasticity': {'mu': 0.2e9, 'nu': 0.40, 'b': 0.371e-9,
                       'sigma0': 5e6, 'gamma0_dot': 1e-3, 'm': 12},
        'misfit_beta': 0.10,
    }

    PVDF_LI = {
        'name': 'PVDF (Polyvinylidene fluoride)',
        'D_SE': 8.0e-13,            # m^2/s
        'D_gb_ratio': 400.0,
        'sigma_SE': 2.0e-4,         # S/m
        'c_surf': 5.0e4,
        'elastic': {'C11': 3.5e9, 'C12': 1.8e9, 'C44': 0.85e9,  # Very stiff
                    'source': 'PVDF (E~2.0 - 2.5 GPa)'},
        'plasticity': {'mu': 0.9e9, 'nu': 0.35, 'b': 0.371e-9,
                       'sigma0': 30e6, 'gamma0_dot': 1e-3, 'm': 20},
        'misfit_beta': 0.12,
    }

    @classmethod
    def get_se_li(cls, name):
        """Lithium-ion polymer SE lookup.

        Mapping keys are the names shown in the Streamlit UI sidebar.
        Falls back to PEO/LiTFSI when the requested name is unknown.
        """
        mapping = {
            'PEO_Li': cls.PEO_LI,
            'PAN':    cls.PAN_LI,
            'PPC':    cls.PPC_LI,
            'PVDF':   cls.PVDF_LI,
        }
        return mapping.get(name, cls.PEO_LI)

    @classmethod
    def list_se_li_options(cls):
        """Lithium-ion polymer SE option list (shown in the sidebar)."""
        return ['PEO_Li', 'PAN', 'PPC', 'PVDF', 'custom']

    @classmethod
    def get_li(cls):
        """Returns Li metal property set."""
        return cls.LI_METAL

    @staticmethod
    def validate_parameters(params):
        errors, warns = [], []
        if params.get('dt', 0) <= 0:
            errors.append("dt must be positive")
        if params.get('dx', 0) <= 0:
            errors.append("dx must be positive")
        if params.get('Nx', 0) < 32 or params.get('Ny', 0) < 32:
            warns.append("N < 32 per axis may be inaccurate")

        D_gb = float(params.get('D_gb', 1.0e-12))
        D_lat = float(params.get('D_bulk', 1.0e-14))
        delta_D_max = max(D_gb - D_lat, 0.0)
        dx_min = min(params.get('dx', 1.0), params.get('dy', params.get('dx', 1.0)))
        dt_expl_max = 0.20 * dx_min * dx_min / max(delta_D_max + D_lat, 1.0e-30)
        if params.get('dt', 0) > dt_expl_max:
            warns.append(
                f"dt = {params.get('dt'):.2e} > CFL for explicit GB term "
                f"({dt_expl_max:.2e}); solver will auto-substep.")
        return errors, warns


# ============================================================================
# TIME-UNIT HANDLING
# ============================================================================
TIME_UNITS = {
    's':  {'label': 's',  'long': 'seconds'},
    'ms': {'label': 'ms', 'long': 'milliseconds'},
    'us': {'label': 'μs', 'long': 'microseconds'},
    'ns': {'label': 'ns', 'long': 'nanoseconds'},
}

DEFAULT_TIME_UNIT = 's'


def get_time_unit_label(params):
    if params is None:
        return DEFAULT_TIME_UNIT
    return params.get('time_unit', DEFAULT_TIME_UNIT)


def fmt_time(t_code, params, fmt='%.4f'):
    unit = get_time_unit_label(params)
    return f"{fmt % float(t_code)} {unit}"


def fmt_time_range(t_code_arr, params):
    unit = get_time_unit_label(params)
    return np.asarray(t_code_arr, dtype=float), unit


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================
def default_field_settings():
    return {
        f: {'auto': True, 'vmin': None, 'vmax': None,
            'log': False, 'invert': False,
            'cmap': FIELD_DEFAULT_CMAPS.get(f, 'viridis')}
        for f in FIELD_DEFAULT_CMAPS.keys()
    }


def default_figure_customization():
    return {
        'multi_field_width':    13.5,
        'multi_field_height':   13.0,
        'overlay_width':        11.0,
        'overlay_height':        5.5,
        'vector_width':         11.0,
        'vector_height':         5.5,
        'diagnostics_width':    13.0,
        'diagnostics_height':   11.0,
        'line_profile_width':   11.0,
        'line_profile_height':   4.5,
        'line_width':            2.0,
        'line_width_minor':      1.5,
        'marker_size':           6,
        'marker_width':          1.0,
        'contour_line_width':    1.0,
        'quiver_width':          0.0025,
        'quiver_scale':          30.0,
        'line_style':            '-',
        'line_style_minor':      '--',
        'marker_style':          'none',
        'marker_style_minor':    'none',
        'color_series_1':        '#1f77b4',
        'color_series_2':        '#d62728',
        'color_series_3':        '#2ca02c',
        'color_series_4':        '#9467bd',
        'color_series_5':        '#000000',
        'title_override':       '',
        'xlabel_override':      '',
        'ylabel_override':      '',
        'title_font_size':      None,
        'label_font_size':      None,
        'tick_font_size':       None,
        'legend_font_size':     None,
        'show_grid':            True,
        'grid_alpha':           0.3,
        'grid_linestyle':       '-',
        'grid_color':           '#cccccc',
        'xlim':                 (None, None),
        'ylim':                 (None, None),
        'show_colorbar':        True,
        'colorbar_orientation': 'vertical',
        'spine_width':          1.0,
        'tick_direction':       'out',
        'tick_width':           1.0,
        'tick_length':          4,
        'spine_top':            True,
        'spine_right':          True,
        'spine_left':           True,
        'spine_bottom':         True,
        'show_legend':          True,
        'legend_position':      'best',
        'legend_frameon':       True,
        'legend_framealpha':    1.0,
        'axis_equal':           True,
        'tight_layout':         True,
    }


def default_viz_config():
    return {
        'theme': 'default',
        'title_font_size': 11,
        'label_font_size': 9,
        'tick_font_size': 8,
        'legend_font_size': 8,
        'dpi': 150,
        'colorbar_fraction': 0.046,
        'colorbar_pad': 0.04,
        'field_settings': default_field_settings(),
        'figure_customization': default_figure_customization(),
        'overlay': {
            'enabled': False,
            'base': 'eta',
            'overlay': 'phi_gb',
            'alpha': 0.5,
            'mode': 'contour',
            'n_contours': 10,
            'show_grain_ghost': False,
        },
        'vector': {
            'enabled': False,
            'source': 'c_li',
            'mode': 'flux',
            'density': 24,
            'color': 'white',
            'scale': 1.0,
        },
        'animation': {
            'format': 'gif',
            'fps': 5,
            'dpi': 120,
            'watermark': False,
            'watermark_text': '',
            'watermark_pos': 'lower right',
        },
    }


def get_viz_config():
    if 'viz_cfg' not in st.session_state:
        st.session_state.viz_cfg = default_viz_config()
    cfg = st.session_state.viz_cfg
    defaults = default_viz_config()
    if 'figure_customization' not in cfg:
        cfg['figure_customization'] = defaults['figure_customization']
    else:
        for k, v in defaults['figure_customization'].items():
            cfg['figure_customization'].setdefault(k, v)
    for fname, fset in cfg['field_settings'].items():
        fset.setdefault('cmap', FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
    return cfg


# ============================================================================
# AXIS STYLING HELPERS
# ============================================================================
def apply_axis_styling(ax, fig_cfg, *, title=None, xlabel=None, ylabel=None,
                       title_fs=None, label_fs=None, grid_axes=None,
                       force_aspect=None):
    if fig_cfg is None or ax is None:
        return
    if title_fs is None:
        title_fs = fig_cfg.get('title_font_size')
    if label_fs is None:
        label_fs = fig_cfg.get('label_font_size')
    tick_fs = fig_cfg.get('tick_font_size')
    if title is not None:
        override = fig_cfg.get('title_override', '').strip()
        ax.set_title(override if override else title, fontsize=title_fs)
    if xlabel is not None:
        override = fig_cfg.get('xlabel_override', '').strip()
        ax.set_xlabel(override if override else xlabel, fontsize=label_fs)
    if ylabel is not None:
        override = fig_cfg.get('ylabel_override', '').strip()
        ax.set_ylabel(override if override else ylabel, fontsize=label_fs)
    if tick_fs is not None:
        for la in ax.get_xticklabels() + ax.get_yticklabels():
            la.set_fontsize(float(tick_fs))
    target = grid_axes if grid_axes is not None else ax
    if fig_cfg.get('show_grid', True):
        target.grid(True,
                    alpha=float(fig_cfg.get('grid_alpha', 0.3)),
                    linestyle=fig_cfg.get('grid_linestyle', '-'),
                    color=fig_cfg.get('grid_color', '#cccccc'))
    else:
        target.grid(False)
    xlim = fig_cfg.get('xlim', (None, None))
    ylim = fig_cfg.get('ylim', (None, None))
    if xlim and (xlim[0] is not None or xlim[1] is not None):
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim and (ylim[0] is not None or ylim[1] is not None):
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    sw = float(fig_cfg.get('spine_width', 1.0))
    spine_visible = {
        'top':    fig_cfg.get('spine_top', True),
        'right':  fig_cfg.get('spine_right', True),
        'left':   fig_cfg.get('spine_left', True),
        'bottom': fig_cfg.get('spine_bottom', True),
    }
    for sn, sp in ax.spines.items():
        sp.set_linewidth(sw)
        sp.set_visible(spine_visible.get(sn, True))
    ax.tick_params(width=float(fig_cfg.get('tick_width', 1.0)),
                   length=float(fig_cfg.get('tick_length', 4)),
                   direction=fig_cfg.get('tick_direction', 'out'))
    if force_aspect is not None:
        aspect_mode = force_aspect
    elif fig_cfg.get('axis_equal', True):
        aspect_mode = 'equal'
    else:
        aspect_mode = 'auto'
    try:
        if aspect_mode == 'equal':
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('auto')
    except Exception:
        pass


def apply_legend_styling(ax, fig_cfg):
    if fig_cfg is None or ax is None:
        return
    if not fig_cfg.get('show_legend', True):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return
    leg_pos = fig_cfg.get('legend_position', 'best')
    legend_fs = fig_cfg.get('legend_font_size')
    frameon = bool(fig_cfg.get('legend_frameon', True))
    framealpha = float(fig_cfg.get('legend_framealpha', 1.0))
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    return ax.legend(handles, labels, loc=leg_pos, frameon=frameon,
                     framealpha=framealpha, fontsize=legend_fs)


# ============================================================================
# VORONOI UTILITIES
# ============================================================================
def voronoi_assign(Nx, Ny, extent, seeds, mask=None):
    """Assign each pixel to its nearest Voronoi seed, restricted to `mask`."""
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if len(seeds) == 0:
        return np.full((Ny, Nx), -1, dtype=int)
    tree = cKDTree(seeds)
    _, idx = tree.query(pts, k=1)
    gm = idx.reshape(Ny, Nx).astype(int)
    if mask is not None:
        gm = np.where(mask, gm, -1)
    return gm


def diffuse_field(binary, sigma_px):
    """Smooth a binary mask by sigma_px (used for soft GB profile)."""
    if sigma_px <= 0:
        return binary.astype(float)
    return gaussian_filter(binary.astype(float), sigma=sigma_px, mode='nearest')


# ============================================================================
# GEOMETRY BUILDER
# ============================================================================
class LiSSBGeometry:
    """Build the 2D simulation domain.

    Left fraction : Li metal electrode (eta = 1)
    Right fraction: polymer solid electrolyte (Voronoi grains, eta = 0)

    Three SE grain configurations are supported:
        - random_voronoi : N random seeds in electrolyte region
        - columnar       : elongated grains perpendicular to interface
        - spherical      : packed circular particles with internal GBs + voids
    """

    def __init__(self, Nx, Ny, dx, dy=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.extent = [0.0, Nx * self.dx, 0.0, Ny * self.dy]
        self.x = np.linspace(self.extent[0], self.extent[1], Nx)
        self.y = np.linspace(self.extent[2], self.extent[3], Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @handle_errors
    def build(self,
              li_frac=0.10,
              se_geometry='random_voronoi',
              n_grains=25,
              grain_size_um=20.0,
              gb_width_um=1.0,
              n_particles=4,
              particle_radius_um=30.0,
              seed=1234,
              initial_dendrite='flat_noise',
              notch_depth_um=10.0,
              notch_width_um=10.0,
              n_preseeds=3,
              noise_amp_um=0.5):
        """Build the geometry and return (eta0, c_li0, grain_map, phi_g, meta).

        Returns
        -------
        eta0      : (Ny, Nx) initial Li order parameter (1=Li metal, 0=SE)
        c_li0    : (Ny, Nx) initial Li+ concentration (0 at Li side, 1 at SE side)
        grain_map: (Ny, Nx) integer Voronoi grain ID (0 = Li region, >=1 = SE grains)
        phi_g    : (Ny, Nx) smoothed grain indicator (1 interior, 0 at GBs)
        meta     : dict with extra geometry info
        """
        rng = np.random.default_rng(seed)
        Nx, Ny = self.Nx, self.Ny
        x_right = self.extent[1]
        y_top = self.extent[3]

        x_li = x_right * li_frac   # x-coordinate of Li/SE interface
        Lx_se = x_right - x_li
        Ly = self.extent[3] - self.extent[2]

        # Binary masks
        mask_li = self.X < x_li
        mask_se = ~mask_li

        # ---- SE grain map ----
        if se_geometry == 'random_voronoi':
            gm_se, n_grains_actual = self._build_random_voronoi(
                mask_se, n_grains, Lx_se, Ly, x_li, rng)
        elif se_geometry == 'columnar':
            gm_se, n_grains_actual = self._build_columnar(
                mask_se, n_grains, Lx_se, Ly, x_li, rng)
        elif se_geometry == 'spherical':
            gm_se, n_grains_actual = self._build_spherical_particles(
                mask_se, n_particles, particle_radius_um,
                grain_size_um, Lx_se, Ly, x_li, rng)
        else:
            gm_se, n_grains_actual = self._build_random_voronoi(
                mask_se, n_grains, Lx_se, Ly, x_li, rng)

        # Composite grain map (0 = Li region, 1..n = SE grains)
        grain_map = np.zeros((Ny, Nx), dtype=int)
        grain_map[mask_se] = gm_se[mask_se] + 1
        n_total_grains = n_grains_actual

        # Smooth grain indicator phi_g (1 interior, 0 at GBs)
        sigma_gb = max(2.0 * gb_width_um / self.dx, 1.0)
        phi_g = self._compute_phi_g(grain_map, mask_se, sigma_gb)
        phi_gb = np.clip(1.0 - phi_g, 0.0, 1.0)  # GB indicator

        # ---- Initial eta (Li metal order parameter) ----
        eta0 = np.zeros((Ny, Nx), dtype=float)
        eta0[mask_li] = 1.0

        # Smooth interface over a few grid cells
        sigma_iface = max(1.5, gb_width_um / self.dx)
        eta0 = np.clip(gaussian_filter(eta0, sigma=sigma_iface, mode='nearest'),
                      0.0, 1.0)

        # ---- Initial-dendrite perturbation ----
        eta0 = self._apply_initial_dendrite(
            eta0, x_li, mask_li, initial_dendrite,
            notch_depth_um, notch_width_um, n_preseeds,
            noise_amp_um, rng, gb_width_um)

        # ---- Initial Li+ concentration ----
        # c_li = 0 inside Li metal, increases to 1 inside SE bulk
        c_li0 = np.zeros((Ny, Nx), dtype=float)
        c_li0[mask_se] = 1.0
        # Smooth transition at interface
        c_li0 = np.clip(gaussian_filter(c_li0, sigma=sigma_iface,
                                         mode='nearest'), 0.0, 1.0)
        # Inside Li metal, set c_li = 0 (all Li atoms, no free Li+)
        c_li0 = np.where(eta0 > 0.5, 0.0, c_li0)

        meta = {
            'x_li_interface': x_li,
            'Lx_se': Lx_se,
            'Ly': Ly,
            'mask_li': mask_li,
            'mask_se': mask_se,
            'n_grains_actual': n_grains_actual,
            'n_total_grains': n_total_grains,
            'se_geometry': se_geometry,
            'gb_width_um': gb_width_um,
            'initial_dendrite': initial_dendrite,
            'sigma_iface_px': sigma_iface,
        }
        return eta0, c_li0, grain_map, phi_g, phi_gb, meta

    # ------------------------------------------------------------------
    def _build_random_voronoi(self, mask_se, n_grains, Lx_se, Ly, x_li, rng):
        """Place N random seeds in SE region and Voronoi-tessellate."""
        ys, xs = np.where(mask_se)
        if len(ys) == 0:
            return np.full((self.Ny, self.Nx), 0, dtype=int), 0
        n_pick = min(int(n_grains), len(ys))
        picks = rng.choice(len(ys), size=n_pick, replace=False)
        seeds = np.column_stack([self.X[ys[picks], xs[picks]],
                                  self.Y[ys[picks], xs[picks]]])
        gm = voronoi_assign(self.Nx, self.Ny, self.extent, seeds, mask=mask_se)
        return gm, n_pick

    # ------------------------------------------------------------------
    def _build_columnar(self, mask_se, n_grains, Lx_se, Ly, x_li, rng):
        """Columnar grains elongated in x (perpendicular to interface)."""
        # Seeds placed on a vertical line; small y-jitter gives columnar shape
        n = max(2, int(n_grains))
        y_seeds = np.linspace(self.extent[2] + 0.05 * Ly,
                              self.extent[3] - 0.05 * Ly, n)
        y_seeds += rng.uniform(-0.5 * Ly / n, 0.5 * Ly / n, size=n)
        # Place each seed at a different x-position to break symmetry slightly
        x_seeds = x_li + 0.5 * Lx_se + rng.uniform(-0.1, 0.1, size=n) * Lx_se
        seeds = np.column_stack([x_seeds, y_seeds])
        gm = voronoi_assign(self.Nx, self.Ny, self.extent, seeds, mask=mask_se)
        return gm, n

    # ------------------------------------------------------------------
    def _build_spherical_particles(self, mask_se, n_particles,
                                     particle_radius_um, grain_size_um,
                                     Lx_se, Ly, x_li, rng):
        """Packed circular SE particles with internal fine grains + voids."""
        Ny, Nx = self.Ny, self.Nx
        gm = np.full((Ny, Nx), -1, dtype=int)
        # Place particle centers on a quasi-random grid in the SE region
        n_part = max(1, int(n_particles))
        centers = []
        attempts = 0
        min_dist = 2.0 * particle_radius_um * 0.95
        while len(centers) < n_part and attempts < 500:
            cx = x_li + particle_radius_um + rng.uniform(
                0, max(0.0, Lx_se - 2 * particle_radius_um))
            cy = self.extent[2] + particle_radius_um + rng.uniform(
                0, max(0.0, Ly - 2 * particle_radius_um))
            ok = all(((cx - c[0]) ** 2 + (cy - c[1]) ** 2) ** 0.5 > min_dist
                     for c in centers)
            if ok:
                centers.append((cx, cy))
            attempts += 1
        centers = np.array(centers) if centers else np.zeros((0, 2))

        # Distance-to-nearest-center map
        if len(centers) > 0:
            tree = cKDTree(centers)
            dist_to_center = tree.query(
                np.column_stack([self.X.ravel(), self.Y.ravel()]), k=1)[0]
            dist_to_center = dist_to_center.reshape(Ny, Nx)
        else:
            dist_to_center = np.full((Ny, Nx), 1.0e9)

        inside_particle = (dist_to_center < particle_radius_um) & mask_se
        # Voids are mask_se pixels outside any particle
        # Each particle gets a few internal Voronoi grains
        n_internal_grains = max(2, int(round(
            np.pi * particle_radius_um ** 2 / max(grain_size_um ** 2, 1.0))))
        all_seeds = []
        for cx, cy in centers:
            for _ in range(n_internal_grains):
                r = particle_radius_um * np.sqrt(rng.uniform(0, 0.85))
                theta = rng.uniform(0, 2 * np.pi)
                all_seeds.append([cx + r * np.cos(theta),
                                  cy + r * np.sin(theta)])
        if all_seeds:
            seeds_arr = np.array(all_seeds)
            gm_internal = voronoi_assign(self.Nx, self.Ny, self.extent,
                                           seeds_arr, mask=inside_particle)
            # Reindex sequentially
            valid = gm_internal >= 0
            gm[valid] = gm_internal[valid]
            n_grains_actual = len(seeds_arr)
        else:
            n_grains_actual = 0
        return gm, n_grains_actual

    # ------------------------------------------------------------------
    def _compute_phi_g(self, grain_map, mask_se, sigma_px):
        """Smooth grain indicator: 1 inside grain, 0 at GBs.

        For each grain g, compute eta_g = Gauss(mask(grain==g), sigma).
        Then phi_g = 1 - sum(eta_g^2) clamped to [0,1].
        """
        Ny, Nx = self.Ny, self.Nx
        ssum = np.zeros((Ny, Nx), dtype=float)
        n_grains = int(grain_map.max() + 1) if grain_map.size else 0
        for g in range(1, n_grains):  # skip 0 (Li region)
            m = (grain_map == g).astype(float)
            if m.sum() == 0:
                continue
            eta = np.clip(gaussian_filter(m, sigma=sigma_px, mode='nearest'),
                          0.0, 1.0)
            ssum += eta * eta
        phi_g = np.clip(1.0 - ssum, 0.0, 1.0)
        # Force phi_g = 1 inside Li region (no grains there)
        phi_g[~mask_se] = 1.0
        return phi_g

    # ------------------------------------------------------------------
    def _apply_initial_dendrite(self, eta0, x_li, mask_li, mode,
                                  notch_depth_um, notch_width_um,
                                  n_preseeds, noise_amp_um, rng,
                                  gb_width_um):
        """Apply the chosen initial-dendrite perturbation to eta0."""
        Ny, Nx = self.Ny, self.Nx
        dx = self.dx
        if mode == 'flat_noise':
            # Small random perturbation at the interface
            sigma_iface = max(1.5, gb_width_um / dx)
            # Distance to interface
            dist_iface = (self.X - x_li)
            in_band = np.abs(dist_iface) < 4.0 * gb_width_um
            noise = rng.uniform(-noise_amp_um, noise_amp_um, size=(Ny, Nx))
            # Apply only in the band around the interface, scaled by distance
            band_profile = np.exp(-(dist_iface / (2.0 * gb_width_um)) ** 2)
            shift = noise * band_profile
            # Translate amplitude to grid cells
            shift_px = shift / dx
            # Recompute eta0 around the interface using a sigmoid
            eta_new = 0.5 * (1.0 - np.tanh(
                (self.X - x_li - shift) / (2.0 * gb_width_um)))
            # Keep Li region inside Li electrode (x < x_li - 4*gb_w)
            deep_na = self.X < (x_li - 4.0 * gb_width_um)
            eta_new[deep_na] = 1.0
            deep_se = self.X > (x_li + 4.0 * gb_width_um)
            eta_new[deep_se] = 0.0
            return np.clip(eta_new, 0.0, 1.0)

        elif mode == 'preseed_tips':
            # Place a few small Li protrusions at GB-electrode intersections
            eta_new = eta0.copy()
            # Find GBs that meet the interface
            sigma_iface = max(1.5, gb_width_um / dx)
            eta_smooth = 0.5 * (1.0 - np.tanh(
                (self.X - x_li) / (2.0 * gb_width_um)))
            eta_new = eta_smooth
            # Find rows where phi_gb is high near the interface
            interface_band = np.abs(self.X - x_li) < 1.5 * gb_width_um
            # Pick top n_preseeds rows with highest GB density
            gb_density_per_row = np.sum(
                self._compute_local_gb_density(eta0.shape), axis=1) \
                if False else np.zeros(Ny)
            # Simpler: pick equally spaced rows
            n_preseeds = max(1, int(n_preseeds))
            tip_rows = np.linspace(int(0.1 * Ny), int(0.9 * Ny),
                                    n_preseeds).astype(int)
            tip_depth_px = max(int(notch_depth_um / dx), 2)
            tip_width_px = max(int(notch_width_um / dx), 2)
            for r in tip_rows:
                r0 = max(0, r - tip_width_px // 2)
                r1 = min(Ny, r + tip_width_px // 2 + 1)
                c0 = max(0, int(x_li / dx) - 0)
                c1 = min(Nx, c0 + tip_depth_px)
                eta_new[r0:r1, c0:c1] = 1.0
            # Smooth the protrusions
            eta_new = np.clip(
                gaussian_filter(eta_new, sigma=sigma_iface, mode='nearest'),
                0.0, 1.0)
            return eta_new

        elif mode == 'single_notch':
            # Single rectangular notch at the center
            eta_new = eta0.copy()
            sigma_iface = max(1.5, gb_width_um / dx)
            # Start from a flat interface
            eta_new = 0.5 * (1.0 - np.tanh(
                (self.X - x_li) / (2.0 * gb_width_um)))
            notch_depth_px = max(int(notch_depth_um / dx), 2)
            notch_width_px = max(int(notch_width_um / dx), 2)
            r_center = Ny // 2
            r0 = max(0, r_center - notch_width_px // 2)
            r1 = min(Ny, r_center + notch_width_px // 2 + 1)
            c0 = max(0, int(x_li / dx) - 0)
            c1 = min(Nx, c0 + notch_depth_px)
            eta_new[r0:r1, c0:c1] = 1.0
            eta_new = np.clip(
                gaussian_filter(eta_new, sigma=sigma_iface, mode='nearest'),
                0.0, 1.0)
            return eta_new

        else:
            return eta0

    # ------------------------------------------------------------------
    def _compute_local_gb_density(self, shape):
        """Placeholder for GB density along rows (not currently used)."""
        return np.zeros(shape)


# Forward-compatibility alias so new code can use `LiSSBGeometry`.
# (Forward-compatibility alias removed: LiSSBGeometry is now the
# primary class name. Legacy code using LiSSBGeometry should be
# updated to use LiSSBGeometry directly.)


# ============================================================================
# SPECTRAL ELASTICITY (plane-strain)
# ============================================================================
class SpectralElasticSolver:
    """Spectral (FFT) elasticity solver in plane-strain.

    Uses an isotropic elastic stiffness tensor expressed in terms of
    the (3D) C11, C12, C44 — converted to 2D plane-strain moduli
    (C11_2d, C12_2d, C44_2d) for the spectral Green's function.
    """

    def __init__(self, Nx, Ny, dx, dy, elastic_params,
                 kx=None, ky=None, k2=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        if kx is None:
            self.kx, self.ky, self.k2 = make_k_vectors(Nx, Ny, dx, dy)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2

        C11 = float(elastic_params['C11'])
        C12 = float(elastic_params['C12'])
        C44 = float(elastic_params['C44'])
        # Plane-strain moduli (2D)
        self.C11_2d = (C11 + C12 + 2.0 * C44) / 2.0
        self.C12_2d = (C11 + C12 - 2.0 * C44) / 2.0
        self.C44_2d = C44
        self.lam = self.C12_2d
        self.mu = (self.C11_2d - self.C12_2d) / 2.0

        k2 = self.k2
        inv_mu_k2 = 1.0 / (self.mu * k2 + 1.0e-30)
        ratio = (self.lam + self.mu) / (self.lam + 2.0 * self.mu)
        self.G11 = inv_mu_k2 * (1.0 - ratio * (self.kx ** 2) / k2)
        self.G12 = inv_mu_k2 * (-ratio * self.kx * self.ky / k2)
        self.G22 = inv_mu_k2 * (1.0 - ratio * (self.ky ** 2) / k2)
        self.G11[0, 0] = self.G12[0, 0] = self.G22[0, 0] = 0.0

    @handle_errors
    def solve(self, exx_star, eyy_star, exy_star,
              app_xx=0.0, app_yy=0.0, app_xy=0.0):
        """Solve mechanical equilibrium with eigenstrain (exx_star, etc.).

        Returns: sigma_eq, sxx, syy, sxy, sigma_h, exx, eyy, exy.
        """
        exx_h = fft2(exx_star)
        eyy_h = fft2(eyy_star)
        exy_h = fft2(exy_star)

        eps_kk_h = exx_h + eyy_h
        fx_h = self.lam * self.kx * eps_kk_h + 2.0 * self.mu * (
            self.kx * exx_h + self.ky * exy_h)
        fy_h = self.lam * self.ky * eps_kk_h + 2.0 * self.mu * (
            self.kx * exy_h + self.ky * eyy_h)

        ux_h = self.G11 * fx_h + self.G12 * fy_h
        uy_h = self.G12 * fx_h + self.G22 * fy_h

        exx_el = np.real(ifft2(1j * self.kx * ux_h))
        eyy_el = np.real(ifft2(1j * self.ky * uy_h))
        exy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_h + self.ky * ux_h)))

        exx = exx_el + exx_star
        eyy = eyy_el + eyy_star
        exy = exy_el + exy_star

        sxx = self.C11_2d * exx_el + self.C12_2d * eyy_el + app_xx
        syy = self.C12_2d * exx_el + self.C11_2d * eyy_el + app_yy
        sxy = 2.0 * self.C44_2d * exy_el + app_xy

        sigma_eq = np.sqrt(0.5 * ((sxx - syy) ** 2 + syy ** 2 + sxx ** 2
                                    + 6.0 * sxy ** 2))
        sigma_eq = np.clip(sigma_eq, 0.0, 5.0e9)
        sigma_h = 0.5 * (sxx + syy)
        return sigma_eq, sxx, syy, sxy, sigma_h, exx, eyy, exy


# ============================================================================
# PLASTICITY (J2 flow + KAM-GND density)
# ============================================================================
def compute_plasticity_step(eps_p_xx, eps_p_yy, eps_p_xy,
                              sigma_eq, h, params, mat_plasticity, dt):
    """Update plastic strain increments using a fixed-direction J2 flow rule.

    Returns (eps_p_xx_new, eps_p_yy_new, eps_p_xy_new, sigma_y, eps_p_mag).
    """
    mu_p = float(mat_plasticity['mu'])
    nu = float(mat_plasticity['nu'])
    b = float(mat_plasticity['b'])
    sigma0 = float(params.get('sigma0', mat_plasticity['sigma0']))
    gamma0_dot = float(mat_plasticity['gamma0_dot'])
    m = int(mat_plasticity['m'])

    h_safe = np.maximum(h, 2.001 * b)
    sigma_y = sigma0 + (mu_p * b / (2.0 * np.pi * h_safe * (1.0 - nu))) * \
              np.log(h_safe / b)
    sigma_y = np.where(h > 2.0 * b, sigma_y,
                       sigma0 + mu_p / (2.0 * np.pi * (1.0 - nu)))

    overstress = np.clip(
        (sigma_eq - sigma_y) / np.maximum(sigma_y, 1.0e-9), 0.0, 1.0)
    gamma_dot = gamma0_dot * overstress ** m
    d_inc = np.minimum(2.0 / 3.0 * gamma_dot * dt, 1.0e-3)

    eps_p_xx_new = np.clip(eps_p_xx + d_inc, -0.1, 0.1)
    eps_p_yy_new = np.clip(eps_p_yy - 0.5 * d_inc, -0.1, 0.1)
    eps_p_xy_new = np.clip(eps_p_xy + 0.5 * d_inc, -0.1, 0.1)

    eps_p_mag = np.sqrt(2.0 / 3.0 * (
        eps_p_xx_new ** 2 + eps_p_yy_new ** 2
        + 2.0 * eps_p_xy_new ** 2 + 1.0e-15))
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new, sigma_y, eps_p_mag


def compute_kam_density(exy_total, gb_indicator):
    """Geometrically-necessary dislocation (GND) density proxy.

    Uses the gradient magnitude of the total shear strain, weighted by
    the GB indicator (GNDs accumulate near GBs).
    """
    gy, gx = np.gradient(exy_total)
    geom_factor = np.sqrt(gx ** 2 + gy ** 2)
    rho = geom_factor * (0.5 + 0.5 * gb_indicator)
    rho_max = float(np.max(rho)) if np.max(rho) > 1.0e-15 else 1.0
    return np.clip(rho / rho_max, 0.0, 1.0)


def compute_local_grain_size(phi_g, kx, ky):
    """Local grain size h(x,y) from the gradient of phi_g (spectral).

    h ~ 2 / |grad phi_g|, clipped to [0, 300] um.
    """
    gx, gy = spectral_gradients(phi_g, kx, ky)
    gm = np.sqrt(gx ** 2 + gy ** 2 + 1.0e-15)
    return np.clip(2.0 / gm, 0.0, 300.0)


# ============================================================================
# HELPER: interpolation + double-well functions
# ============================================================================
def h_interp(eta):
    """Smooth interpolation h(eta) = 6*eta^5 - 15*eta^4 + 10*eta^3.
    h(0)=0, h(1)=1, monotone, smooth derivatives at 0 and 1.
    """
    return 6.0 * eta ** 5 - 15.0 * eta ** 4 + 10.0 * eta ** 3


def h_interp_prime(eta):
    """Derivative of h(eta): 30*eta^4 - 60*eta^3 + 30*eta^2 = 30*eta^2*(1-eta)^2."""
    return 30.0 * eta ** 2 * (1.0 - eta) ** 2


def g_double_well(eta):
    """Double-well potential g(eta) = eta^2 * (1-eta)^2."""
    return eta * eta * (1.0 - eta) ** 2


def g_double_well_prime(eta):
    """Derivative: 2*eta*(1-eta)*(1-2*eta)."""
    return 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta)


# ============================================================================
# MAIN PHASE-FIELD SOLVER
# ============================================================================
class LiSSBSolver:
    """Lithium dendrite growth solver for all-solid-state Li batteries with
    polymer solid electrolytes (PEO, PAN, PPC, PVDF).

    Coupled physics per Wu et al. (2027), adapted for Li metal:
        1. Allen-Cahn for eta (Li/SE order parameter) with Butler-Volmer source
        2. Li+ reaction-diffusion with Nernst-Planck flux and Hart D_eff
        3. Charge conservation for electric potential phi
        4. Spectral elastoplasticity for sigma_h stress-migration coupling

    Geometry: Li electrode on left (eta=1), polymer SE on right (eta=0, Voronoi).

    Temperature is taken from params['temperature_K'] (default 333.15 K = 60 C),
    which is critical for PEO/LiTFSI where chain segmental motion only
    enables adequate ionic mobility above ~60 C.
    """

    def __init__(self, params):
        self.params = params
        self.Nx = int(params['Nx'])
        self.Ny = int(params['Ny'])
        self.dx = float(params['dx'])
        self.dy = float(params.get('dy', params['dx']))
        self.dt = float(params['dt'])

        # Operating temperature (K) — defaults to 60 C for PEO compatibility.
        # PEO/LiTFSI relies on polymer chain segmental motion to transport
        # Li+, so it requires T > 60 C. PAN/PPC/PVDF can run at lower T.
        self.temperature_K = float(params.get('temperature_K', 333.15))

        # Material property sets — Li metal + Li-polymer SE.
        # `self.mat_metal` is the canonical name; `self.mat_metal` is kept as a
        # backward-compatible alias (the solver body still references it).
        self.mat_metal = MaterialProperties.get_li()
        se_name = params.get('se_material', 'PEO_Li')
        if se_name == 'custom':
            # Build a custom SE dict from user-supplied D_bulk, D_gb_ratio.
            # Defaults inherit PEO_LI elastic/plastic constants so the user
            # only needs to override transport properties.
            base = MaterialProperties.PEO_LI.copy()
            base['name'] = 'custom'
            base['D_SE'] = float(params.get('custom_D_bulk', 5.0e-13))
            base['D_gb_ratio'] = float(params.get('custom_D_gb_ratio', 200.0))
            base['sigma_SE'] = float(params.get('custom_sigma_SE', 1.0e-4))
            self.mat_se = base
        else:
            # Look up the named material, then let the user override the
            # transport properties (D_SE, D_gb_ratio, sigma_SE) via the
            # always-on UI sliders.  This makes PEO/PAN/PPC/PVDF tunable
            # without picking 'custom'.
            self.mat_se = MaterialProperties.get_se_li(se_name).copy()
            if 'custom_D_bulk' in params:
                self.mat_se['D_SE'] = float(params['custom_D_bulk'])
            if 'custom_D_gb_ratio' in params:
                self.mat_se['D_gb_ratio'] = float(params['custom_D_gb_ratio'])
            if 'custom_sigma_SE' in params:
                self.mat_se['sigma_SE'] = float(params['custom_sigma_SE'])

        # Wavenumber vectors
        self.kx, self.ky, self.k2 = make_k_vectors(
            self.Nx, self.Ny, self.dx, self.dy)

        # Build geometry
        self.geom = LiSSBGeometry(self.Nx, self.Ny, self.dx, self.dy)
        result = self.geom.build(
            li_frac=params.get('li_frac', 0.10),
            se_geometry=params.get('se_geometry', 'random_voronoi'),
            n_grains=params.get('n_grains', 25),
            grain_size_um=params.get('grain_size_um', 20.0),
            gb_width_um=params.get('gb_width_um', 1.0),
            n_particles=params.get('n_particles', 4),
            particle_radius_um=params.get('particle_radius_um', 30.0),
            seed=params.get('seed', 1234),
            initial_dendrite=params.get('initial_dendrite', 'flat_noise'),
            notch_depth_um=params.get('notch_depth_um', 10.0),
            notch_width_um=params.get('notch_width_um', 10.0),
            n_preseeds=params.get('n_preseeds', 3),
            noise_amp_um=params.get('noise_amp_um', 0.5))
        if result is FAILED or result is None:
            raise RuntimeError(f"Geometry build failed: {FAILED.reason}")
        (eta_init, self.c_li, self.grain_map, _phi_g_static, _phi_gb_static,
         self.geom_meta) = result

        # Cache geometry grid for use in solver methods
        self.X = self.geom.X
        self.Y = self.geom.Y

        # ---- UNIFIED MPF STATE: eta_list ----
        # Build unified eta_list: eta_0 = Li metal, eta_1..N = SE grains.
        # All fields share the constraint sum(eta_i) = 1 (Lagrange projection).
        # GB indicator phi_gb = 1 - sum(eta_i^2) is now DYNAMIC, recomputed
        # at every step from the current eta_list (no longer static Voronoi).
        self.eta_list = self._build_eta_list(eta_init, self.grain_map)
        self.eta = self.eta_list[0]   # alias for backward compatibility

        # Phase-field parameters
        self.L_eta = float(params.get('L_eta', 1.0e-6))    # interface mobility
        self.L_phi = float(params.get('L_phi', 5.0e-3))  # reaction prefactor (raised from 1e-3)
        self.kappa = float(params.get('kappa', 2.0e-15))  # gradient energy (Li surface energy)
        self.w_barrier = float(params.get('w_barrier', 2.0e6))  # barrier height (Li surface energy)
        self.alpha_bv = float(params.get('alpha_bv', 0.5))  # BV symmetry
        self.aniso_strength = float(params.get('aniso_strength', 0.0))
        self.Omega_stress = float(params.get('Omega_stress', 0.05))
        self.beta_misfit = float(self.mat_se.get('misfit_beta', 0.10))
        # MPF anti-overlap parameter (controls grain-grain repulsion)
        self.B_anti_overlap = float(params.get('B_anti_overlap', 2.0 * 1.0e6))
        # Li-SE coupling factor: scales the anti-overlap term for eta_0 (Li).
        # Smaller value -> weaker repulsion between Li and SE grains ->
        # dendrites can wet GBs and propagate. Larger value -> Li beads up
        # at the interface instead of penetrating. Default 0.05 = weak coupling.
        self.li_se_coupling_factor = float(
            params.get('li_se_coupling_factor', 0.05))
        # Mobility reduction factor for SE grains (keeps SE grains static-ish
        # so dendrites propagate into the existing GB network rather than
        # coarsening the grains themselves).
        self.se_grain_mobility_factor = float(
            params.get('se_grain_mobility_factor', 1.0e-3))
        # Wetting enhancement: boosts BV source at Li/SE interface where
        # dynamic GB indicator is high (drives preferential GB propagation).
        #self.wetting_factor = float(params.get('wetting_factor', 1.0))
        self.wetting_factor = float(params.get('wetting_factor', 1.0))

        # --- NEW PHYSICS: transference number, salt diffusion, SEI damage ---
        # Transference number t+ of Li+ in the polymer electrolyte.
        # PEO/LiTFSI: t+ ~ 0.2-0.4 (low — anions carry most of the
        # current).  PAN/PPC/PVDF: t+ ~ 0.5-0.7.  Low t+ causes severe
        # concentration polarization at the plating front, which is the
        # primary driver of Li dendrite growth in real cells.  In the
        # Nernst-Planck solver, the cation migration term is scaled by
        # t+ (the cation only carries t+ of the ionic current).
        self.transference_number = float(
            params.get('transference_number', 0.30))

        # SEI (solid-electrolyte interphase) mechanical damage.
        # The SEI on Li is mechanically robust; under deposition pressure
        # it accumulates damage until it cracks, at which point Li
        # extrudes through the crack in a whisker / root-growth mode
        # (the dominant Li dendrite morphology).  Damage accumulates
        # when the local hydrostatic stress exceeds the SEI yield
        # stress, and relaxes (heals) exponentially.
        self.sei_yield_stress_Pa = float(
            params.get('sei_yield_stress_MPa', 50.0)) * 1.0e6  # MPa -> Pa
        self.sei_healing_rate = float(
            params.get('sei_healing_rate', 1.0e-4))  # 1/s (slow)

        # Salt (LiTFSI) concentration field c_salt: diffuses with its
        # own D_salt.  Initially uniform = 1 (normalized), depleted near
        # the plating front by Li+ consumption.
        self.D_salt = float(self.mat_se.get('D_SE', 5.0e-13))
        self.c_salt = np.ones((self.Ny, self.Nx), dtype=float)

        # SEI damage field: 0 = intact SEI, 1 = fully fractured.
        # Initialized to 0 everywhere (no pre-existing cracks).
        self.sei_damage = np.zeros((self.Ny, self.Nx), dtype=float)

        # Applied voltage (single plating direction) — must be set before
        # _init_potential() is called.
        self.applied_voltage = float(params.get('applied_voltage', 0.2))  # V

        # Effective diffusivity (Hart)
        self.D_Li = float(self.mat_metal['D_Li'])
        self.D_SE_bulk = float(self.mat_se['D_SE'])
        self.D_gb_ratio = float(self.mat_se.get('D_gb_ratio', 500.0))
        self.D_gb = self.D_SE_bulk * self.D_gb_ratio
        self.sigma_Li = float(self.mat_metal['sigma_Li'])
        self.sigma_SE = float(self.mat_se['sigma_SE'])
        self.c0_ref = float(self.mat_metal['c0'])
        self.c_surf = float(self.mat_se['c_surf'])

        # Compute DYNAMIC phi_g, phi_gb from eta_list (replaces static versions)
        self._update_dynamic_grain_fields()

        # Electric potential (initialized to applied voltage profile)
        self.phi_pot = self._init_potential()

        # Precompute D_eff and sigma_eff based on initial fields
        self.D_eff = self._compute_D_eff()
        self.sigma_eff = self._compute_sigma_eff()

        # Plastic strain accumulators
        self.eps_p_xx = np.zeros((self.Ny, self.Nx))
        self.eps_p_yy = np.zeros((self.Ny, self.Nx))
        self.eps_p_xy = np.zeros((self.Ny, self.Nx))

        # Spectral elasticity solver (use SE elastic moduli; Li region uses
        # the same solver with eigenstrain-driven deformation)
        self.elastic = SpectralElasticSolver(
            self.Nx, self.Ny, self.dx, self.dy,
            self.mat_se['elastic'], self.kx, self.ky, self.k2)

        # History tracking
        self.history = {
            'time': [], 'eta_gb_mean': [], 'eta_bulk_mean': [],
            'c_li_gb_mean': [], 'c_li_bulk_mean': [],
            'c_salt_gb_mean': [], 'c_salt_bulk_mean': [],
            'max_dendrite_depth_um': [], 'li_volume_frac': [],
            'phi_min': [], 'phi_max': [],
            'avg_stress_GPa': [], 'max_stress_GPa': [],
            'plastic_work': [], 'energy': [],
            'avg_sei_damage': [], 'max_sei_damage': [],
        }

    # ------------------------------------------------------------------
    def _build_eta_list(self, eta_init, grain_map):
        """Build the unified eta_list [eta_0=Li, eta_1, ..., eta_N=SE grains].

        Each eta_i is smoothed with sigma = gb_width / dx so the GB indicator
        1 - sum(eta_i^2) has a finite width matching the physical GB width.
        After construction, fields are renormalized so sum(eta_i) = 1.
        """
        Ny, Nx = self.Ny, self.Nx
        sigma_gb = max(2.0 * self.geom_meta['gb_width_um'] / self.dx, 1.0)
        eta_list = [eta_init.copy()]   # eta_0 = Li metal

        n_grains_max = int(grain_map.max()) if grain_map.size else 0
        for g in range(1, n_grains_max + 1):
            mask = (grain_map == g).astype(float)
            if mask.sum() == 0:
                continue
            eta = np.clip(
                gaussian_filter(mask, sigma=sigma_gb, mode='nearest'),
                0.0, 1.0)
            eta_list.append(eta)

        # Renormalize so sum(eta_i) = 1 (Lagrange projection onto the
        # hyperplane sum(eta_i) = 1).
        ssum = sum(eta_list)
        ssum_safe = np.where(ssum > 1.0e-12, ssum, 1.0)
        eta_list = [e / ssum_safe for e in eta_list]
        return eta_list

    # ------------------------------------------------------------------
    def _update_dynamic_grain_fields(self):
        """Recompute phi_g and phi_gb from current eta_list (DYNAMIC).

        phi_g  = sum(eta_i^2)               (1 inside grain, 0 at GB)
        phi_gb = 1 - sum(eta_i^2)           (0 inside grain, 1 at GB)

        With eta_0 = Li included in the sum, the Li/SE interface automatically
        appears as a high-phi_gb region (because eta_0 and eta_k overlap there,
        so sum(eta_i^2) drops below 1). This is the key fix: GBs are no longer
        static Voronoi artifacts — they evolve with the moving Li front.
        """
        s = np.zeros((self.Ny, self.Nx))
        for e in self.eta_list:
            s = s + e * e
        self.phi_g = np.clip(s, 0.0, 1.0)
        self.phi_gb = np.clip(1.0 - s, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _init_potential(self):
        """Initialize electric potential linearly from V_app (left) to 0 (right)."""
        x_li = self.geom_meta['x_li_interface']
        x_right = self.geom.extent[1]
        # Linear gradient: phi = V_app at x=0, phi=0 at x=x_right
        phi = self.applied_voltage * (1.0 - self.X / x_right)
        # Inside Li region, phi is roughly uniform (high conductivity)
        mask_li = self.geom_meta['mask_li']
        phi[mask_li] = self.applied_voltage
        return phi

    # ------------------------------------------------------------------
    def _compute_D_eff(self):
        """Hart effective diffusivity using DYNAMIC GB indicator.

        D_eff = h(eta_0) * D_Li + (1 - h(eta_0)) * [phi_g_eff * D_bulk + (1 - phi_g_eff) * D_gb]

        where phi_g_eff includes a "wetting" enhancement: the Li/SE interface
        itself (where eta_0 crosses 0.5) is treated as an additional GB for
        diffusion, so dendrites automatically experience high D_eff as they
        propagate along GBs and through the moving interface.
        """
        h_eta = h_interp(self.eta_list[0])
        # Dynamic GB indicator from current eta_list (includes Li/SE interface)
        phi_g = self.phi_g
        # Wetting enhancement: eta_interface peaks at eta=0.5 (the Li/SE
        # interface) and forces phi_g_eff -> 0 there, which means D_eff
        # picks up the high D_gb value at the moving front. This is the
        # physical mechanism that makes dendrites propagate along GBs.
        eta_interface = 4.0 * self.eta_list[0] * (1.0 - self.eta_list[0])
        phi_g_eff = phi_g * (1.0 - eta_interface) + 0.0 * eta_interface
        D_se_local = phi_g_eff * self.D_SE_bulk + (1.0 - phi_g_eff) * self.D_gb
        return h_eta * self.D_Li + (1.0 - h_eta) * D_se_local

    # ------------------------------------------------------------------
    def _compute_sigma_eff(self):
        """Effective conductivity using dynamic phi_g (no wetting boost)."""
        h_eta = h_interp(self.eta_list[0])
        sigma_se_local = self.phi_g * self.sigma_SE
        return h_eta * self.sigma_Li + (1.0 - h_eta) * sigma_se_local

    # ------------------------------------------------------------------
    def _compute_electron_concentration(self):
        """Electron concentration: c_e = c_surf * eta + c_surf_at_GB * (1-eta)*(1-phi_g)."""
        # Following paper: c_e = c_surf * eta + (c_surf/c0) * (1-eta) * (1-phi_g)
        c_e_norm = self.eta + (1.0 - self.eta) * (1.0 - self.phi_g) * 0.5
        return np.clip(c_e_norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _compute_overpotential(self):
        """Local overpotential eta = phi - phi_eq (reference).

        For plating, eta < 0 at the Li electrode side drives Li+ + e- -> Li.
        """
        # Reference potential: equilibrium potential (take as 0)
        # eta = -phi (so negative phi at anode drives plating)
        # Use the deviation from the right boundary (counter-electrode)
        return -self.phi_pot

    # ------------------------------------------------------------------
    def _butler_volmer_source(self):
        """Butler-Volmer source for eta_0 (Li metal) ONLY.

        R = L_phi * h'(eta_0) * [exp(-alpha*F*eta/RT) - c_Li+ * c_e * exp((1-alpha)*F*eta/RT)]
            + nucleation_term

        Key change vs. old code: h'(eta_0) restricts the reaction to the
        Li/SE interface region (where eta_0 is between 0 and 1), and the
        dynamic phi_gb in c_e now naturally enhances BV at GBs because
        c_e is higher there.

        A small nucleation term is added so eta_0 can grow from 0 in the
        SE bulk (where h'(0) = 0). The nucleation rate is proportional to
        phi_gb * c_li * (positive overpotential) — i.e., Li nucleates at
        GBs where there is sufficient Li+ supply and a driving force.
        Without this, dendrites can't propagate past the initial interface
        because h'(eta_0=0) = 0 kills the BV source.
        """
        eta_0 = self.eta_list[0]
        eta_local = self._compute_overpotential()
        c_e = self._compute_electron_concentration()
        c_li_safe = np.clip(self.c_li, 0.0, 1.0)

        F_over_RT = F_FARADAY / (R_GAS * self.temperature_K)
        alpha = self.alpha_bv

        # Plating rate (bwd - fwd): positive means Li deposits (eta_0 grows)
        fwd = c_li_safe * c_e * np.exp((1.0 - alpha) * F_over_RT * eta_local)
        bwd = np.exp(-alpha * F_over_RT * eta_local)
        bv_rate = bwd - fwd

        # h'(eta_0) = 30 * eta_0^2 * (1-eta_0)^2 — restricts reaction to interface
        hp = h_interp_prime(eta_0)
        # Apply wetting factor — boosts BV at GB-rich interface regions
        wetting = 1.0 + self.wetting_factor * self.phi_gb
        bv_interface = self.L_phi * hp * bv_rate * wetting

        # Nucleation term: allows eta_0 to grow from 0 in the SE bulk.
        # Active ONLY where eta_0 < 0.3 (deep SE), and weighted by:
        #   - phi_gb (GBs are preferred nucleation sites)
        #   - c_li (need Li+ supply)
        #   - max(0, bv_rate) (only plate, don't dissolve)
        # Without this, dendrites cannot propagate past the initial
        # interface because h'(eta_0=0) = 0 kills the BV source.
        nucleation_mask = (eta_0 < 0.3).astype(float)
        nucleation_rate = (self.L_phi * 0.5 *
                           self.phi_gb *
                           c_li_safe *
                           np.maximum(bv_rate, 0.0) *
                           nucleation_mask *
                           self.wetting_factor)
        # --- SEI damage enhancement (root-growth / whisker mode) ---
        # Where the SEI has cracked (sei_damage -> 1), the BV source is
        # boosted by a factor (1 + sei_damage * sei_enhancement), so Li
        # preferentially plates through the crack.  This is the
        # physical mechanism that makes Li dendrites grow via root
        # extrusion rather than Na's surface-particle mode.
        sei_enhancement = 5.0  # max 5x BV at full damage
        bv_interface = bv_interface * (1.0 + self.sei_damage * sei_enhancement)
        return bv_interface + nucleation_rate

    # ------------------------------------------------------------------
    def _evolve_eta(self, sigma_h):
        """UNIFIED MPF evolution of all eta_i fields.

        Replaces the old decoupled Allen-Cahn update of eta with a full MPF
        step that evolves eta_0 (Li) AND eta_1..N (SE grains) together,
        enforcing sum(eta_i) = 1 via Lagrange projection.

        For each eta_i:
            d(eta_i)/dt = -L_i * mu_i + BV_source (eta_0 only) + stress_coupling (eta_0 only)

        where  mu_i = w * g'(eta_i) + 2 * B * eta_i * sum_{j!=i}(eta_j^2)
                       - kappa * nabla^2 eta_i

        Mobility L_i is full for eta_0 (Li metal — moves freely) and reduced
        by `se_grain_mobility_factor` for SE grains (so they don't coarsen
        away the GB network that the dendrite is supposed to follow).

        After the explicit step, all eta_i are renormalized so sum = 1.
        """
        A = self.w_barrier
        B = self.B_anti_overlap
        kappa = self.kappa
        L_base = self.L_eta

        n_fields = len(self.eta_list)
        sq = [e * e for e in self.eta_list]
        total_sq = sum(sq)
        # Precompute Laplacian for each eta
        laps = [spectral_laplacian(e, self.k2) for e in self.eta_list]

        # Stress-migration coupling (only for eta_0 = Li)
        sigma_h_GPa = (sigma_h / 1.0e9) if sigma_h is not None else 0.0
        stress_coupling = -self.Omega_stress * sigma_h_GPa * 1.0e6

        # Butler-Volmer source for eta_0 only
        bv_src_eta0 = self._butler_volmer_source()
        # Per-step stability limiter: d(eta_0) per step can't exceed 0.1
        # (10% of a phase-field unit per step). This is a numerical safety
        # only — the physical rate is set by L_phi and the BV exponentials.
        max_rate = 0.1 / max(self.dt, 1.0e-30)
        bv_src_eta0 = np.clip(bv_src_eta0, -max_rate, max_rate)

        new_list = []
        for i, e in enumerate(self.eta_list):
            others_sq = total_sq - sq[i]
            # MPF chemical potential: double-well + anti-overlap + gradient.
            # For eta_0 (Li metal), use a much smaller anti-overlap coefficient
            # so the BV source can drive dendrite propagation into the SE.
            # Without this, the SE grains' anti-overlap term (2*B*eta_0*sum_others^2)
            # overwhelms the BV source and pushes eta_0 back to 0.
            if i == 0:
                # Li-SE coupling: small B so dendrites can wet GBs
                B_eff = B * self.li_se_coupling_factor
            else:
                # SE-SE coupling: full B to keep grains distinct
                B_eff = B
            df_de = (2.0 * A * e * (1.0 - e) * (1.0 - 2.0 * e)
                     + 2.0 * B_eff * e * others_sq)
            mu = df_de - kappa * laps[i]

            # Mobility: full for Li, reduced for SE grains (static grains)
            if i == 0:
                L_eff = L_base
            else:
                L_eff = L_base * self.se_grain_mobility_factor

            rhs = -L_eff * mu

            if i == 0:
                # Add Butler-Volmer source (drives plating at interface)
                rhs = rhs + bv_src_eta0
                # Add stress-migration coupling
                rhs = rhs - L_eff * stress_coupling

            # Stability clip on rhs (per-step change can't exceed 1.0 in eta)
            rhs_clip = 1.0 / max(self.dt, 1.0e-30)
            rhs = np.clip(rhs, -rhs_clip, rhs_clip)
            e_new = e + self.dt * rhs
            new_list.append(np.clip(e_new, 0.0, 1.0))

        # Renormalize: sum(eta_i) = 1 (Lagrange projection)
        ssum = sum(new_list)
        ssum_safe = np.where(ssum > 1.0e-12, ssum, 1.0)
        new_list = [e / ssum_safe for e in new_list]

        # Enforce Dirichlet BC in deep Li region: eta_0 = 1, all others = 0
        x_li = self.geom_meta['x_li_interface']
        gb_w = self.geom_meta['gb_width_um']
        deep_na = self.X < (x_li - 4.0 * gb_w)
        for i, e in enumerate(new_list):
            if i == 0:
                e[deep_na] = 1.0
            else:
                e[deep_na] = 0.0

        return new_list

    # ------------------------------------------------------------------
    def _evolve_c_li(self, deta_dt):
        """Li+ reaction-diffusion step (Nernst-Planck with transference number).

        Eq. 8 (modified for transference number t+):
            d(c_li)/dt = nabla . [D_eff * (nabla c_li + t+ * (F/RT) * c_li * nabla phi)]
                                - (c_Li/c0) * deta/dt

        The migration term is scaled by t+ (the cation carries only t+
        of the ionic current).  With low t+ (e.g. 0.3 for PEO/LiTFSI),
        the cation flux is mostly diffusive, and concentration
        polarization builds up at the plating front — the primary
        driver of Li dendrite growth in real cells.
        """
        D_eff = self.D_eff
        F_over_RT = F_FARADAY / (R_GAS * self.temperature_K)
        t_plus = self.transference_number

        # Substep for stability (explicit diffusion)
        dx_min = min(self.dx, self.dy)
        D_max = float(np.max(D_eff)) if D_eff.size else 1.0e-12
        D_max = max(D_max, 1.0e-30)
        dt_expl = 0.20 * dx_min * dx_min / D_max
        n_sub = max(1, int(np.ceil(self.dt / dt_expl)))
        dt_sub = self.dt / n_sub

        c_li_new = self.c_li.copy()
        for _ in range(n_sub):
            # Gradient of c_li and phi
            c_gx, c_gy = spectral_gradients(c_li_new, self.kx, self.ky)
            phi_gx, phi_gy = spectral_gradients(self.phi_pot, self.kx, self.ky)
            # Nernst-Planck flux with transference number t+:
            #   J = -D_eff * (grad c_li + t+ * (F/RT) * c_li * grad phi)
            # The diffusion term is full D_eff; the migration term is
            # scaled by t+ (cation carries only t+ of the ionic current).
            c_safe = np.clip(c_li_new, 0.0, 1.0)
            Jx = -D_eff * (c_gx + t_plus * F_over_RT * c_safe * phi_gx)
            Jy = -D_eff * (c_gy + t_plus * F_over_RT * c_safe * phi_gy)
            div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)
            # Reaction source: -(c_Li/c0) * deta/dt (consumption of Li+ when eta grows)
            c_total_norm = c_safe
            source = -c_total_norm * deta_dt
            c_li_new = c_li_new + dt_sub * (div_J + source)
            # Enforce BCs: c_li = 0 inside Li metal, c_li = 1 at right boundary
            c_li_new = np.where(self.eta > 0.5, 0.0, c_li_new)
            # Right boundary: Dirichlet c_li = 1
            c_li_new[:, -1] = 1.0
            c_li_new = np.clip(c_li_new, 0.0, 1.0)

        return c_li_new

    # ------------------------------------------------------------------
    def _evolve_c_salt(self, deta_dt):
        """Salt (LiTFSI) concentration step (diffusion only).

        The salt concentration c_salt represents the local LiTFSI
        molarity (normalized to 1 in the bulk).  It diffuses with D_salt
        and is depleted at the plating front by the BV sink (Li+ is
        consumed, so the salt is consumed too).  Visualizing c_salt
        shows the salt-depletion zone near the dendrite, which is a
        key signature of low-transference-number cells.

            d(c_salt)/dt = nabla . (D_salt * nabla c_salt) - c_salt * deta/dt
        """
        D_salt = self.D_salt
        dx_min = min(self.dx, self.dy)
        D_max = max(D_salt, 1.0e-30)
        dt_expl = 0.20 * dx_min * dx_min / D_max
        n_sub = max(1, int(np.ceil(self.dt / dt_expl)))
        dt_sub = self.dt / n_sub

        c_salt_new = self.c_salt.copy()
        for _ in range(n_sub):
            c_gx, c_gy = spectral_gradients(c_salt_new, self.kx, self.ky)
            Jx = -D_salt * c_gx
            Jy = -D_salt * c_gy
            div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)
            # Depletion at the plating front (Li+ consumed -> salt consumed)
            source = -np.clip(c_salt_new, 0.0, 1.0) * deta_dt
            c_salt_new = c_salt_new + dt_sub * (div_J + source)
            # BCs: c_salt = 1 in the SE bulk (right boundary); arbitrary in Li metal.
            c_salt_new = np.where(self.eta > 0.5, 0.0, c_salt_new)
            c_salt_new[:, -1] = 1.0
            c_salt_new = np.clip(c_salt_new, 0.0, 1.0)

        return c_salt_new

    # ------------------------------------------------------------------
    def _update_sei_damage(self, sigma_h):
        """Update the SEI damage field.

        Damage accumulates when the local hydrostatic stress exceeds the
        SEI yield stress (compressive or tensile — we use |sigma_h|).
        Damage relaxes (heals) exponentially with rate sei_healing_rate.
        High sei_damage locally enhances the Butler-Volmer source in
        _butler_volmer_source(), producing the root-growth / whisker
        morphology characteristic of Li dendrites.

            d(damage)/dt = max(0, |sigma_h| - sigma_y_sei)/sigma_y_sei
                            - sei_healing_rate * damage
        """
        sigma_h_safe = np.abs(sigma_h) if sigma_h is not None else np.zeros_like(self.eta)
        # Damage rate: 0 if |sigma_h| < sigma_y_sei, grows linearly above.
        overstress = np.maximum(
            0.0, (sigma_h_safe - self.sei_yield_stress_Pa) /
            max(self.sei_yield_stress_Pa, 1.0))
        # Cap accumulation rate to 1/s to avoid runaway.
        accum_rate = np.clip(overstress, 0.0, 1.0)
        # Only accumulate in the SE region (the SEI lives at the Li/SE
        # interface, which is the high-phi_gb band).
        accum_rate = accum_rate * (0.5 + 0.5 * self.phi_gb)
        d_damage = accum_rate - self.sei_healing_rate * self.sei_damage
        self.sei_damage = np.clip(
            self.sei_damage + self.dt * d_damage, 0.0, 1.0)
        return self.sei_damage

    # ------------------------------------------------------------------
    def _evolve_phi_pot(self, deta_dt):
        """Electric potential update (charge conservation, Eq. 9).

            nabla . (sigma_eff * nabla phi) = -c_Li * F * deta/dt

        For numerical stability we use a relaxed update:
            1. Start from the linear baseline (V_app at left, 0 at right).
            2. Add a small bounded perturbation driven by the local
               reaction source, scaled inversely by sigma_eff.
            3. Clip the perturbation to avoid runaway values.

        This is qualitatively correct (potential drops faster where
        current is being consumed) without the numerical blow-up of
        an unregularized spectral Poisson solve.
        """
        x_right = self.geom.extent[1]
        # Linear baseline from V_app at left to 0 at right
        baseline = self.applied_voltage * (1.0 - self.X / x_right)
        # Force baseline inside Li electrode (high conductivity -> equipotential)
        mask_li = self.geom_meta['mask_li']

        # Local source-driven perturbation:
        # delta_phi ~ -source * dx^2 / sigma_eff (discrete Poisson)
        # but clipped to [-0.5, 0.5] V for stability.
        sigma_eff_safe = np.maximum(self.sigma_eff, 1.0e-10)
        # Only the SE region gets a perturbation (Li region is equipotential)
        mask_se = self.geom_meta['mask_se']
        dx_min = min(self.dx, self.dy)
        # Source: -c0 * F * deta_dt (units: C/(m^3 s))
        src = -self.c0_ref * F_FARADAY * deta_dt
        # Local potential drop: src * dx^2 / sigma_eff (units: V)
        delta_phi = src * dx_min * dx_min / sigma_eff_safe
        # Clip to physical range
        delta_phi = np.clip(delta_phi, -0.5, 0.5)
        # Only apply in SE region (Li region is equipotential)
        delta_phi = np.where(mask_se, delta_phi, 0.0)

        # Blend: smooth perturbation over a few grid cells
        sigma_blend = max(1.0, 2.0 * self.geom_meta['gb_width_um'] / dx_min)
        delta_phi_smooth = gaussian_filter(delta_phi, sigma=sigma_blend,
                                             mode='nearest')

        phi_new = baseline + delta_phi_smooth
        # Force Dirichlet BCs
        phi_new[mask_li] = self.applied_voltage
        phi_new[:, -1] = 0.0  # right boundary
        # Final clip to applied voltage range for safety
        phi_new = np.clip(phi_new, -2.0 * abs(self.applied_voltage),
                          2.0 * abs(self.applied_voltage))
        return phi_new

    # ------------------------------------------------------------------
    def _compute_eigenstrain(self):
        """Eigenstrain from Li insertion: eps* = beta * eta."""
        beta = self.beta_misfit
        return beta * self.eta, beta * self.eta, np.zeros_like(self.eta)

    # ------------------------------------------------------------------
    def step(self):
        """Advance one time step.

        Sequence:
            1. Solve elastoplasticity (with current eta -> eigenstrain)
            2. Compute deta/dt (Allen-Cahn + BV), update eta
            3. Update Li+ concentration
            4. Update electric potential
            5. Recompute D_eff, sigma_eff
        """
        # 1. Mechanics: eigenstrain + applied stress
        sig = float(self.params.get('applied_stress', 0.0))
        theta = np.deg2rad(float(self.params.get('applied_stress_angle', 0.0)))
        app_xx = sig * np.cos(theta) ** 2
        app_yy = sig * np.sin(theta) ** 2
        app_xy = sig * np.sin(theta) * np.cos(theta)

        exx_s, eyy_s, exy_s = self._compute_eigenstrain()
        exx_t = exx_s + self.eps_p_xx
        eyy_t = eyy_s + self.eps_p_yy
        exy_t = exy_s + self.eps_p_xy

        el = self.elastic.solve(exx_t, eyy_t, exy_t,
                                  app_xx, app_yy, app_xy)
        if el is FAILED or el is None:
            raise RuntimeError("Elastic solve failed")
        (sigma_eq, sxx, syy, sxy, sigma_h,
         exx_tot, eyy_tot, exy_tot) = el

        # Plasticity update
        h_grain = compute_local_grain_size(self.phi_g, self.kx, self.ky)
        (self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
         sigma_y, eps_p_mag) = compute_plasticity_step(
            self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
            sigma_eq, h_grain, self.params,
            self.mat_se['plasticity'], self.dt)

        rho_gnd = compute_kam_density(exy_tot, self.phi_gb)

        # 2. Unified MPF evolution of eta_list (returns new list)
        eta_old = self.eta_list[0].copy()
        new_eta_list = self._evolve_eta(sigma_h)  # name kept for compat
        self.eta_list = new_eta_list
        self.eta = self.eta_list[0]   # update alias
        # Recompute DYNAMIC phi_g, phi_gb from new eta_list (this is the
        # key fix — GBs evolve with the moving Li front)
        self._update_dynamic_grain_fields()
        deta_dt = (self.eta - eta_old) / self.dt

        # 3. Update Li+ concentration (Nernst-Planck with transference number t+)
        self.c_li = self._evolve_c_li(deta_dt)

        # 3b. Update salt (LiTFSI) concentration
        self.c_salt = self._evolve_c_salt(deta_dt)

        # 4. Update electric potential
        self.phi_pot = self._evolve_phi_pot(deta_dt)

        # 4b. Update SEI damage (root-growth / whisker mode driver)
        self._update_sei_damage(sigma_h)

        # 5. Recompute effective transport properties (uses DYNAMIC phi_g)
        self.D_eff = self._compute_D_eff()
        self.sigma_eff = self._compute_sigma_eff()

        # Compute c_e and flux magnitude for visualization
        c_e = self._compute_electron_concentration()
        F_over_RT = F_FARADAY / (R_GAS * self.temperature_K)
        c_gx, c_gy = spectral_gradients(self.c_li, self.kx, self.ky)
        phi_gx, phi_gy = spectral_gradients(self.phi_pot, self.kx, self.ky)
        c_safe = np.clip(self.c_li, 0.0, 1.0)
        Jx = -self.D_eff * (c_gx + F_over_RT * c_safe * phi_gx)
        Jy = -self.D_eff * (c_gy + F_over_RT * c_safe * phi_gy)
        phi_flux_mag = np.sqrt(Jx ** 2 + Jy ** 2)
        # Normalize flux for display
        flux_max = float(np.max(phi_flux_mag)) if np.max(phi_flux_mag) > 0 else 1.0
        phi_flux_norm = phi_flux_mag / flux_max

        # ---- Diagnostics ----
        x_li = self.geom_meta['x_li_interface']
        # Dendrite depth: max x where eta > 0.5, measured from x_li
        dendrite_mask = (self.eta > 0.5) & (self.X > x_li)
        if np.any(dendrite_mask):
            max_x_dendrite = float(np.max(self.X[dendrite_mask]))
            dendrite_depth = max_x_dendrite - x_li
        else:
            dendrite_depth = 0.0

        # Mean eta in GB vs bulk (within SE region only)
        mask_se = self.geom_meta['mask_se']
        gb_in_se = (self.phi_gb > 0.5) & mask_se
        bulk_in_se = (self.phi_gb < 0.5) & mask_se
        eta_gb_mean = float(np.mean(self.eta[gb_in_se])) if gb_in_se.any() else 0.0
        eta_bulk_mean = float(np.mean(self.eta[bulk_in_se])) if bulk_in_se.any() else 0.0
        c_li_gb_mean = float(np.mean(self.c_li[gb_in_se])) if gb_in_se.any() else 0.0
        c_li_bulk_mean = float(np.mean(self.c_li[bulk_in_se])) if bulk_in_se.any() else 0.0
        c_salt_gb_mean = float(np.mean(self.c_salt[gb_in_se])) if gb_in_se.any() else 0.0
        c_salt_bulk_mean = float(np.mean(self.c_salt[bulk_in_se])) if bulk_in_se.any() else 0.0
        avg_sei_damage = float(np.mean(self.sei_damage))
        max_sei_damage = float(np.max(self.sei_damage))

        # Li volume fraction
        li_vol_frac = float(np.mean(self.eta > 0.5))

        # Energy (gradient + bulk)
        gx, gy = spectral_gradients(self.eta, self.kx, self.ky)
        grad_energy = 0.5 * self.kappa * float(np.sum(gx ** 2 + gy ** 2)) \
                       * self.dx * self.dy
        bulk_energy = self.w_barrier * float(np.sum(g_double_well(self.eta))) \
                       * self.dx * self.dy
        total_energy = grad_energy + bulk_energy

        # Append to history
        t_now = (len(self.history['time']) + 1) * self.dt
        self.history['time'].append(t_now)
        self.history['eta_gb_mean'].append(eta_gb_mean)
        self.history['eta_bulk_mean'].append(eta_bulk_mean)
        self.history['c_li_gb_mean'].append(c_li_gb_mean)
        self.history['c_li_bulk_mean'].append(c_li_bulk_mean)
        self.history['c_salt_gb_mean'].append(c_salt_gb_mean)
        self.history['c_salt_bulk_mean'].append(c_salt_bulk_mean)
        self.history['max_dendrite_depth_um'].append(dendrite_depth)
        self.history['li_volume_frac'].append(li_vol_frac)
        self.history['phi_min'].append(float(np.min(self.phi_pot)))
        self.history['phi_max'].append(float(np.max(self.phi_pot)))
        self.history['avg_stress_GPa'].append(float(np.mean(sigma_eq) / 1.0e9))
        self.history['max_stress_GPa'].append(float(np.max(sigma_eq) / 1.0e9))
        self.history['plastic_work'].append(
            float(np.sum(eps_p_mag) * self.dx * self.dy))
        self.history['energy'].append(total_energy)
        self.history['avg_sei_damage'].append(avg_sei_damage)
        self.history['max_sei_damage'].append(max_sei_damage)

        # Return snapshot for visualization
        return {
            'eta':         self.eta.copy(),
            'c_li':       self.c_li.copy(),
            'c_salt':     self.c_salt.copy(),
            'c_e':        c_e,
            'phi_g':      self.phi_g.copy(),
            'grain_map':  self.grain_map.astype(float),
            'phi_gb':     self.phi_gb.copy(),
            'phi_pot':    self.phi_pot.copy(),
            'D_eff':      self.D_eff.copy(),
            'sigma_eq':   sigma_eq.copy(),
            'sigma_h':    sigma_h.copy(),
            'h':          h_grain.copy(),
            'eps_p_mag':  eps_p_mag.copy(),
            'sigma_y':    sigma_y.copy(),
            'rho_gnd':    rho_gnd,
            'phi_flux':   phi_flux_norm,
            'sei_damage': self.sei_damage.copy(),
            'sxx':        sxx.copy(),
            'syy':        syy.copy(),
            'sxy':        sxy.copy(),
            'convergence': {
                'eta_gb_mean': eta_gb_mean,
                'eta_bulk_mean': eta_bulk_mean,
                'max_dendrite_depth_um': dendrite_depth,
                'li_volume_frac': li_vol_frac,
                'avg_stress_GPa': float(np.mean(sigma_eq) / 1.0e9),
                'energy': total_energy,
            },
        }


# Backward-compatibility alias so existing call sites that use the old
# `LiSSBSolver` name keep working. New code should use `LiSSBSolver`.
# (Backward-compatibility alias removed: LiSSBSolver is now the only
# solver name. Legacy code using LiSSBSolver should be updated to use
# LiSSBSolver directly.)


# ============================================================================
# LINE PROFILER
# ============================================================================
class EnhancedLineProfiler:
    """Extract 1D profiles from 2D fields along various directions."""

    def __init__(self, Nx, Ny, dx, dy):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.extent = [0.0, Nx * dx, 0.0, Ny * dy]

    @handle_errors
    def extract_profile(self, data, profile_type='horizontal',
                        position_ratio=0.5, angle_deg=45):
        ptype = str(profile_type).lower().replace('-', '_')
        ny, nx = data.shape
        cx, cy = nx // 2, ny // 2
        off_x = int(nx * 0.35 * position_ratio)
        off_y = int(ny * 0.35 * position_ratio)

        if ptype == 'horizontal':
            r = min(max(cy + off_y, 0), ny - 1)
            return (np.linspace(self.extent[0], self.extent[1], nx),
                    data[r, :],
                    (self.extent[0], r * self.dy + self.extent[2],
                     self.extent[1], r * self.dy + self.extent[2]))
        if ptype == 'vertical':
            cidx = min(max(cx + off_x, 0), nx - 1)
            return (np.linspace(self.extent[2], self.extent[3], ny),
                    data[:, cidx],
                    (cidx * self.dx + self.extent[0], self.extent[2],
                     cidx * self.dx + self.extent[0], self.extent[3]))
        if ptype == 'diagonal':
            L = int(min(nx, ny) * 0.8)
            s = (cx - L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] + i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        if ptype == 'anti_diagonal':
            L = int(min(nx, ny) * 0.8)
            s = (cx + L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] - i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        return None, None, None


# ============================================================================
# FIELD RESCALING HELPERS
# ============================================================================
def _rescale_field(field_name, data):
    """Convert field to display units and return (data, unit_str)."""
    d = data.copy()
    if field_name in ('sigma_eq', 'sigma_h', 'sxx', 'syy', 'sxy'):
        d = d / 1.0e9   # Pa -> GPa
    elif field_name == 'sigma_y':
        d = d / 1.0e6   # Pa -> MPa
    elif field_name == 'D_eff':
        # Log10 for display
        d = np.log10(np.maximum(d, 1.0e-30))
    return d, FIELD_UNITS.get(field_name, '')


def compute_field_norm(field_name, data, fs, default_rng=None):
    """Return (norm, vmin, vmax) for colorbar scaling."""
    if fs is None:
        fs = {}
    auto = fs.get('auto', True)
    use_log = fs.get('log', False)

    d = np.asarray(data)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return None, None, None

    if auto:
        if default_rng is not None:
            vmin, vmax = float(default_rng[0]), float(default_rng[1])
        else:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if field_name in ('sigma_h', 'phi_pot'):
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
    else:
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None
                    else float(np.min(finite)))
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None
                    else float(np.max(finite)))

    if vmax - vmin < 1.0e-12:
        vmax = vmin + 1.0e-9

    norm = None
    if use_log:
        pos = finite[finite > 0]
        if pos.size > 0:
            if vmin <= 0:
                vmin = float(np.percentile(pos, 2))
            if vmax <= 0:
                vmax = float(np.percentile(pos, 98))
            if vmin >= vmax:
                vmin = vmax / 10.0
            try:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            except Exception:
                norm = None
    return norm, vmin, vmax


def get_global_bounds(history, field_name, fs, default_rng=None):
    """Return (vmin, vmax) fixed across all history frames."""
    if fs is None:
        fs = {}
    if not fs.get('auto', True):
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None else 0.0)
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None else 1.0)
        return float(vmin), float(vmax)
    if default_rng is not None:
        return float(default_rng[0]), float(default_rng[1])
    vmin_all, vmax_all = [], []
    for h in history:
        if field_name not in h:
            continue
        dd, _ = _rescale_field(field_name, h[field_name])
        finite = dd[np.isfinite(dd)]
        if finite.size:
            vmin_all.append(np.percentile(finite, 2))
            vmax_all.append(np.percentile(finite, 98))
    if not vmin_all:
        return None, None
    vmin = float(np.min(vmin_all))
    vmax = float(np.max(vmax_all))
    if field_name in ('sigma_h', 'phi_pot'):
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax


def compute_vector_field(results_dict, source, mode='gradient'):
    """Compute (vx, vy) for vector overlay."""
    if mode == 'principal_stress':
        sxx = results_dict.get('sxx')
        syy = results_dict.get('syy')
        sxy = results_dict.get('sxy')
        if sxx is None or syy is None or sxy is None:
            return None, None
        theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
        return np.cos(theta), np.sin(theta)
    f = results_dict.get(source)
    if f is None:
        return None, None
    gy, gx = np.gradient(f)
    if mode == 'flux':
        D = results_dict.get('D_eff')
        if D is None:
            D = np.ones_like(f)
        vx = -D * gx
        vy = -D * gy
    else:
        vx, vy = gx, gy
    return vx, vy


# ============================================================================
# VISUALIZER
# ============================================================================
class LiSSBVisualizer:
    """Matplotlib + Plotly visualization for Li dendrite simulation results."""

    def __init__(self, Nx, Ny, dx, dy=None, dt=1.0):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dt = dt
        self.extent = [0.0, Nx * self.dx, 0.0, Ny * self.dy]
        self.line_profiler = EnhancedLineProfiler(Nx, Ny, dx, self.dy)

    def get_colormap(self, name, invert=False):
        return ColormapRegistry.resolve(name, invert=invert)

    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 10))
        label_fs = float(style_params.get('label_font_size', 8))
        cbar_fraction = float(style_params.get('colorbar_fraction', 0.046))
        cbar_pad = float(style_params.get('colorbar_pad', 0.04))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        # Fields to display (in order)
        fields = [
            ('eta',        'eta (Li metal order parameter)', [0, 1]),
            ('c_li',      'c_Li+ (Li+ concentration)',     [0, 1]),
            ('c_salt',    'c_salt (LiTFSI salt)',          [0, 1]),
            ('c_e',       'c_e- (electron concentration)', [0, 1]),
            ('grain_map', 'Voronoi grain map',              None),
            ('phi_g',     'phi_g (SE grain indicator)',     [0, 1]),
            ('phi_gb',    'phi_gb (GB indicator)',          [0, 1]),
            ('phi_pot',   'phi (electric potential, V)',    None),
            ('D_eff',     'log10(D_eff) (m^2/s)',           None),
            ('sigma_eq',  'sigma_eq (GPa)',                 None),
            ('sigma_h',   'sigma_h (GPa)',                  None),
            ('h',         'Local grain size (um)',          None),
            ('eps_p_mag', 'Plastic strain |eps_p|',          None),
            ('sigma_y',   'Yield stress (MPa)',             None),
            ('rho_gnd',   'rho_GND (GND density)',          [0, 1]),
            ('phi_flux',  '|J_Li+| (flux magnitude)',       [0, 1]),
            ('sei_damage','SEI damage (0-1)',               [0, 1]),
        ]
        avail = [(f, t, r) for (f, t, r) in fields if f in results_dict]
        if not avail:
            return None

        n = len(avail)
        cols = 3 if n >= 3 else n
        rows = (n + cols - 1) // cols
        fig_w = float(fig_cfg.get('multi_field_width', 4.5 * cols))
        fig_h = float(fig_cfg.get('multi_field_height', 3.8 * rows))
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')

        for idx, (fname, title, rng) in enumerate(avail):
            ax = axes[idx]
            d, unit = _rescale_field(fname, results_dict[fname])

            fs = field_settings.get(fname, {})
            cmap_name = (fs.get('cmap') or
                         FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
            cmap = self.get_colormap(cmap_name, invert=fs.get('invert', False))
            interp = 'nearest' if fname in DISCRETE_FIELDS else 'bilinear'

            if fname == 'grain_map':
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               origin='lower', aspect='equal',
                               interpolation='nearest')
                t_ov = fig_cfg.get('title_override', '').strip()
                ax.set_title(t_ov if t_ov else title, fontsize=title_fs)
                xl = fig_cfg.get('xlabel_override', '').strip() or 'x (um)'
                yl = fig_cfg.get('ylabel_override', '').strip() or 'y (um)'
                ax.set_xlabel(xl, fontsize=label_fs)
                ax.set_ylabel(yl, fontsize=label_fs)
                apply_axis_styling(ax, fig_cfg, grid_axes=None)
                continue

            default_rng = FIELD_DEFAULT_RNG.get(fname, rng)
            norm, vmin, vmax = compute_field_norm(fname, d, fs,
                                                   default_rng=default_rng)
            if norm is not None:
                im = ax.imshow(d, extent=self.extent, cmap=cmap, norm=norm,
                               origin='lower', aspect='equal',
                               interpolation=interp)
            else:
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal',
                               interpolation=interp)
            t_ov = fig_cfg.get('title_override', '').strip()
            ax.set_title(t_ov if t_ov else title, fontsize=title_fs)
            xl = fig_cfg.get('xlabel_override', '').strip() or 'x (um)'
            yl = fig_cfg.get('ylabel_override', '').strip() or 'y (um)'
            ax.set_xlabel(xl, fontsize=label_fs)
            ax.set_ylabel(yl, fontsize=label_fs)
            if show_cb:
                cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction,
                                     pad=cbar_pad, orientation=cb_orient)
                if unit:
                    cbar.set_label(unit)
            apply_axis_styling(ax, fig_cfg, grid_axes=None)

        for k in range(n, len(axes)):
            axes[k].axis('off')
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_overlay(self, results_dict, base_field, overlay_field,
                       alpha=0.5, mode='contour', n_contours=10,
                       show_grain_ghost=False, grain_map=None,
                       style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        if base_field not in results_dict or overlay_field not in results_dict:
            return None
        base, base_unit = _rescale_field(base_field, results_dict[base_field])
        ov, ov_unit = _rescale_field(overlay_field, results_dict[overlay_field])

        base_fs = field_settings.get(base_field, {})
        ov_fs = field_settings.get(overlay_field, {})
        base_cmap = self.get_colormap(
            base_fs.get('cmap') or FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'),
            invert=base_fs.get('invert', False))

        fig_w = float(fig_cfg.get('overlay_width', 11.0))
        fig_h = float(fig_cfg.get('overlay_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field in DISCRETE_FIELDS:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                origin='lower', aspect='equal',
                                interpolation='nearest')
        else:
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    norm=norm, origin='lower', aspect='equal',
                                    interpolation='bilinear')
            else:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    vmin=vmin, vmax=vmax,
                                    origin='lower', aspect='equal',
                                    interpolation='bilinear')

        contour_lw = float(fig_cfg.get('contour_line_width', 1.0))
        if mode == 'contour':
            ov_cmap = self.get_colormap(
                ov_fs.get('cmap') or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'),
                invert=ov_fs.get('invert', False))
            x = np.linspace(self.extent[0], self.extent[1], self.Nx)
            y = np.linspace(self.extent[2], self.extent[3], self.Ny)
            X, Y = np.meshgrid(x, y)
            ov_for_contour = ov.copy()
            ov_for_contour[~np.isfinite(ov_for_contour)] = np.nan
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            try:
                if ov_vmin is None or ov_vmax is None or ov_vmax <= ov_vmin:
                    levels = n_contours
                else:
                    levels = np.linspace(ov_vmin, ov_vmax, n_contours)
                cs = ax.contour(X, Y, ov_for_contour, levels=levels,
                                cmap=ov_cmap, linewidths=contour_lw, alpha=alpha)
                ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
            except Exception as e:
                logger.warning(f"Overlay contour failed: {e}")
        else:
            ov_cmap = self.get_colormap(
                ov_fs.get('cmap') or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'),
                invert=ov_fs.get('invert', False))
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            ax.imshow(ov, extent=self.extent, cmap=ov_cmap,
                      vmin=ov_vmin, vmax=ov_vmax,
                      origin='lower', aspect='equal',
                      interpolation='bilinear', alpha=alpha)

        if fig_cfg.get('show_colorbar', True):
            cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
            cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04,
                                 orientation=cb_orient)
            if base_unit:
                cbar.set_label(f"{base_field} ({base_unit})")

        default_title = (
            f"Overlay: {FIELD_LABELS.get(base_field, base_field)} + "
            f"{FIELD_LABELS.get(overlay_field, overlay_field)} "
            f"(mode={mode}, alpha={alpha:.2f})")
        apply_axis_styling(ax, fig_cfg,
                            title=default_title, title_fs=title_fs,
                            xlabel='x (um)', ylabel='y (um)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_vector_overlay(self, results_dict, source, mode='gradient',
                              density=24, color='white', scale=1.0,
                              base_field=None, style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        vx, vy = compute_vector_field(results_dict, source, mode=mode)
        if vx is None:
            return None

        fig_w = float(fig_cfg.get('vector_width', 11.0))
        fig_h = float(fig_cfg.get('vector_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field and base_field in results_dict:
            base, base_unit = _rescale_field(base_field,
                                              results_dict[base_field])
            base_fs = field_settings.get(base_field, {})
            base_cmap = self.get_colormap(
                base_fs.get('cmap') or
                FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'),
                invert=base_fs.get('invert', False))
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               norm=norm, origin='lower', aspect='equal',
                               interpolation='bilinear')
            else:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal',
                               interpolation='bilinear')
            if fig_cfg.get('show_colorbar', True):
                cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             orientation=cb_orient,
                             label=f"{base_field} ({base_unit})")
        else:
            ax.set_facecolor('#111111')

        step = max(1, self.Nx // max(density, 1))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)[::step]
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)[::step]
        X, Y = np.meshgrid(x, y)
        U = vx[::step, ::step] * scale
        V = vy[::step, ::step] * scale
        mag = np.sqrt(U ** 2 + V ** 2) + 1.0e-12
        U_n = U / mag
        V_n = V / mag

        q_width = float(fig_cfg.get('quiver_width', 0.0025))
        q_scale = float(fig_cfg.get('quiver_scale', 30.0))
        ax.quiver(X, Y, U_n, V_n, mag, cmap='autumn', alpha=0.9,
                  scale=q_scale, width=q_width, pivot='mid')

        apply_axis_styling(ax, fig_cfg,
                            title=f"Vector field: {mode} of {source}",
                            title_fs=title_fs,
                            xlabel='x (um)', ylabel='y (um)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, zmid=None, viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = (cmap_name or fs.get('cmap') or
                  FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        z = d
        if fs.get('log', False):
            pos = d > 0
            z = np.where(pos, np.log10(np.maximum(d, 1.0e-30)), np.nan)
        if zmid is None and field_name in ('sigma_h', 'phi_pot'):
            zmid = 0

        fig_cfg = viz_cfg.get('figure_customization', {})
        p_width = int(fig_cfg.get('overlay_width', 11.0) * 60)
        p_height = int(fig_cfg.get('overlay_height', 5.5) * 60)
        xl = fig_cfg.get('xlabel_override', '').strip() or 'x (um)'
        yl = fig_cfg.get('ylabel_override', '').strip() or 'y (um)'
        title_ov = fig_cfg.get('title_override', '').strip()
        title_str = (title_ov if title_ov
                     else f"{field_name} - Frame {frame_idx} ({chosen})")

        fig = go.Figure()
        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_dict = dict(title=f"{field_name} {unit}") if show_cb else None
        fig.add_trace(go.Heatmap(
            z=z,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid, zmin=vmin, zmax=vmax,
            colorbar=cb_dict,
            hovertemplate='x: %{x:.1f} um<br>y: %{y:.1f} um<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title=title_str,
                          xaxis_title=xl, yaxis_title=yl,
                          yaxis=dict(scaleanchor="x",
                                     scaleratio=self.dy / self.dx),
                          width=max(400, p_width), height=max(300, p_height),
                          template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_animation(self, history, field_name, dt,
                                cmap_name=None, viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})
        chosen = (cmap_name or fs.get('cmap') or
                  FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        zmin, zmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))
        if field_name == 'grain_map':
            zmin, zmax = None, None

        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        frames = []
        for i, h in enumerate(history):
            d, _ = _rescale_field(field_name, h[field_name])
            frames.append(go.Frame(
                data=[go.Heatmap(z=d, x=x, y=y, colorscale=colorscale,
                                 zmin=zmin, zmax=zmax,
                                 colorbar=dict(title=f"{field_name}"))],
                name=f"t={i*dt:.4f}"))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"{field_name} - interactive time scrubbing",
            xaxis_title="x (um)", yaxis_title="y (um)",
            yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
            width=850, height=600, template="plotly_white",
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                          fromcurrent=True,
                                          transition=dict(duration=0))]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ])],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f.name],
                                  dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))],
                            label=f.name) for f in frames],
                x=0.05, len=0.9, xanchor="left", y=0, yanchor="top",
                currentvalue=dict(prefix="Frame: ", visible=True,
                                  xanchor="right"))])
        return fig

    @handle_errors
    def create_animation_buffer(self, history, field_name, dt,
                                output_format='gif', fps=5, dpi=120,
                                cmap_name=None, watermark=False,
                                watermark_text='', watermark_pos='lower right',
                                viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        cmap = ColormapRegistry.resolve(
            cmap_name or fs.get('cmap') or
            FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'),
            invert=fs.get('invert', False))

        d0, unit = _rescale_field(field_name, history[0][field_name])
        vmin, vmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))
        if field_name == 'grain_map':
            vmin, vmax = None, None

        interp = 'nearest' if field_name in DISCRETE_FIELDS else 'bilinear'
        im = ax.imshow(d0, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', interpolation=interp, aspect='auto')
        if field_name != 'grain_map':
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f"{field_name} {unit}")
        title = ax.set_title(f"{field_name} - t = 0 s")

        wm_artist = None
        if watermark and watermark_text:
            xmap = {'lower right': (0.98, 0.02, 'right', 'bottom'),
                    'lower left':  (0.02, 0.02, 'left',  'bottom'),
                    'upper right': (0.98, 0.98, 'right', 'top'),
                    'upper left':  (0.02, 0.98, 'left',  'top')}
            x, y, ha, va = xmap.get(watermark_pos, (0.98, 0.02, 'right', 'bottom'))
            wm_artist = ax.text(x, y, watermark_text, transform=ax.transAxes,
                                fontsize=8, color='white', alpha=0.75,
                                ha=ha, va=va,
                                bbox=dict(boxstyle='round,pad=0.25',
                                          facecolor='black', alpha=0.35,
                                          edgecolor='none'))

        def update(i):
            d, _ = _rescale_field(field_name, history[i][field_name])
            im.set_array(d)
            title.set_text(f"{field_name} - t = {i * dt:.4f} s")
            return ([im, title] if wm_artist is None
                    else [im, title, wm_artist])

        ani = animation.FuncAnimation(fig, update, frames=len(history),
                                       interval=1000 / max(fps, 1), blit=True)
        buf = BytesIO()
        try:
            if output_format == 'gif':
                ani.save(buf, writer='pillow', fps=fps, dpi=dpi)
            else:
                ani.save(buf, writer='ffmpeg', fps=fps, dpi=dpi)
        except Exception as e:
            logger.error(f"Animation export failed: {e}")
            plt.close(fig)
            return None
        plt.close(fig)
        buf.seek(0)
        return buf

    @handle_errors
    def create_multi_time_profiles(self, history, field_name, ptype='horizontal',
                                   frames=None, position_ratio=0.5,
                                   use_plotly=False, viz_cfg=None):
        if frames is None:
            frames = list(range(len(history)))
        frames = [f for f in frames if 0 <= f < len(history)]
        if not frames:
            return None
        viz_cfg = viz_cfg or {}
        fig_cfg = viz_cfg.get('figure_customization', {}) or {}

        if use_plotly:
            fig = go.Figure()
            lw = float(fig_cfg.get('line_width', 2.0))
            for f in frames:
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                         name=f"Frame {f}",
                                         line=dict(width=lw)))
            p_width = int(float(fig_cfg.get('line_profile_width', 11.0)) * 60)
            p_height = int(float(fig_cfg.get('line_profile_height', 4.5)) * 60)
            fig.update_layout(title=f"{field_name} - multi-time {ptype}",
                              xaxis_title="Position (um)",
                              yaxis_title=f"{field_name}",
                              hovermode='x unified', template="plotly_white",
                              width=max(400, p_width), height=max(300, p_height))
            return fig
        else:
            fig_w = float(fig_cfg.get('line_profile_width', 11.0))
            fig_h = float(fig_cfg.get('line_profile_height', 4.5))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cmap = plt.cm.viridis
            n = len(frames)
            lw = float(fig_cfg.get('line_width', 2.0))
            for k, f in enumerate(frames):
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                color = cmap(k / max(n - 1, 1))
                ax.plot(dist, prof, color=color, lw=lw, label=f"Frame {f}")
            apply_axis_styling(ax, fig_cfg,
                                title=f'{field_name} - multi-time {ptype}',
                                xlabel='Position (um)',
                                ylabel=f'{field_name}')
            ax.legend(loc='best', fontsize=8)
            if fig_cfg.get('tight_layout', True):
                plt.tight_layout()
            return fig


# Forward-compatibility alias so new code can use `LiSSBVisualizer`.
# (Forward-compatibility alias removed: LiSSBVisualizer is now the
# primary class name. Legacy code using LiSSBVisualizer should be
# updated to use LiSSBVisualizer directly.)


# ============================================================================
# DATA EXPORTERS
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, name):
        buf = BytesIO()
        pickle.dump({'params': params, 'history': history,
                     'metadata': sim_data.get('metadata', {})}, buf)
        buf.seek(0)
        return buf, f"{name}.pkl"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, name):
        def conv(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(i) for i in o]
            return o
        s = json.dumps({'params': params,
                        'metadata': sim_data.get('metadata', {}),
                        'history': conv(history)},
                       indent=2, cls=NumpyEncoder)
        return BytesIO(s.encode()), f"{name}.json"

    @staticmethod
    @handle_errors
    def export_csv(history, name, extent, Nx, Ny, dx, dy):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            keys = ('eta', 'c_li', 'c_salt', 'c_e', 'phi_g', 'phi_gb',
                    'grain_map', 'phi_pot', 'sigma_eq', 'sigma_h', 'sigma_y',
                    'h', 'eps_p_mag', 'D_eff', 'rho_gnd', 'phi_flux',
                    'sei_damage')
            for i, frame in enumerate(history):
                d = {'x': X.flatten(), 'y': Y.flatten()}
                for k in keys:
                    if k in frame:
                        v = frame[k]
                        if k in ('sigma_eq', 'sigma_h'):
                            v = v / 1.0e9
                        elif k == 'sigma_y':
                            v = v / 1.0e6
                        d[k] = np.asarray(v).flatten()
                zf.writestr(f"{name}_frame_{i:04d}.csv",
                            pd.DataFrame(d).to_csv(index=False))
        buf.seek(0)
        return buf, f"{name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, name, Nx, Ny, dx, dy):
        if not H5PY_AVAILABLE:
            return None, None
        buf = BytesIO()
        with h5py.File(buf, 'w') as f:
            pg = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    pg.attrs[k] = v
            f.create_dataset('x', data=np.linspace(0, Nx * dx, Nx))
            f.create_dataset('y', data=np.linspace(0, Ny * dy, Ny))
            fg = f.create_group('frames')
            for i, frame in enumerate(history):
                g = fg.create_group(f'frame_{i:04d}')
                for k, v in frame.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v)
        buf.seek(0)
        return buf, f"{name}.h5"


# ============================================================================
# SIMULATION DATABASE (in-session storage)
# ============================================================================
class SimulationDatabase:
    MAX_SIMS = 32

    @staticmethod
    def _hash(params):
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ('history', 'results', 'n_steps',
                                     'save_frequency')},
                       sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
    def save(params, history, metadata=None, run_time=0.0):
        if 'sims' not in st.session_state:
            st.session_state.sims = {}
        sid = SimulationDatabase._hash(params)
        st.session_state.sims[sid] = {
            'id': sid, 'params': params, 'results_history': history,
            'metadata': metadata or {'created_at': datetime.now().isoformat(),
                                       'run_time': run_time},
            'created_at': datetime.now().isoformat(),
        }
        if len(st.session_state.sims) > SimulationDatabase.MAX_SIMS:
            oldest = min(st.session_state.sims.items(),
                         key=lambda kv: kv[1].get('created_at', ''))
            del st.session_state.sims[oldest[0]]
        return sid

    @staticmethod
    def get(sid):
        return st.session_state.get('sims', {}).get(sid)

    @staticmethod
    def delete(sid):
        if sid in st.session_state.get('sims', {}):
            del st.session_state.sims[sid]
            return True
        return False

    @staticmethod
    def clear_all():
        st.session_state.sims = {}

    @staticmethod
    def all():
        return st.session_state.get('sims', {})

    @staticmethod
    def list_for_ui():
        out = []
        for sid, sim in SimulationDatabase.all().items():
            p = sim.get('params', {})
            se_name = p.get('se_material', '?')
            n_g = p.get('n_grains', '?')
            v = p.get('applied_voltage', 0)
            out.append({
                'id': sid,
                'name': (f"Li-SSB | SE={se_name} | grains={n_g} | "
                         f"V={v}V | seed={p.get('seed', '?')}"),
                'params': p,
            })
        return out


# ============================================================================
# STREAMLIT UI
# ============================================================================
def _clamped_number_input(label, value, min_value, max_value, step=0.5,
                            key=None, help=None, format=None):
    """st.number_input wrapper that clamps value to [min, max] before passing."""
    if value is None:
        value = min_value
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(min_value)
    if value < min_value:
        value = float(min_value)
    elif value > max_value:
        value = float(max_value)
    kwargs = dict(value=value, min_value=min_value, max_value=max_value,
                  step=step)
    if key is not None:
        kwargs['key'] = key
    if help is not None:
        kwargs['help'] = help
    if format is not None:
        kwargs['format'] = format
    return st.number_input(label, **kwargs)


def render_inline_figure_customization(viz_cfg, *, size_keys=None,
                                         key_prefix='inl',
                                         show_grid=True, show_colorbar=True,
                                         show_lines=False,
                                         label='Figure customization'):
    """Compact inline customization panel."""
    fc = viz_cfg.setdefault('figure_customization',
                              default_figure_customization())
    with st.expander(f"Fig: {label}", expanded=False):
        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            c1, c2 = st.columns(2)
            with c1:
                new_w = _clamped_number_input(
                    "Width (in)", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w")
            with c2:
                new_h = _clamped_number_input(
                    "Height (in)", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            fc[w_key] = float(new_w)
            fc[h_key] = float(new_h)
        if show_grid:
            new_grid = st.checkbox("Show grid",
                                     value=bool(fc.get('show_grid', True)),
                                     key=f"{key_prefix}_grid")
            fc['show_grid'] = new_grid
        if show_colorbar:
            new_cb = st.checkbox("Show colorbar",
                                   value=bool(fc.get('show_colorbar', True)),
                                   key=f"{key_prefix}_cb")
            fc['show_colorbar'] = new_cb
        if show_lines:
            new_lw = st.number_input(
                "Line width", value=float(fc.get('line_width', 2.0)),
                min_value=0.2, max_value=10.0, step=0.1,
                key=f"{key_prefix}_lw")
            fc['line_width'] = new_lw
    return True


def render_field_settings_panel(viz_cfg, key_prefix='fs'):
    """Per-field dynamic range + colormap panel."""
    field_keys = list(FIELD_DEFAULT_CMAPS.keys())
    field_choice = st.selectbox("Field to configure", field_keys,
                                  key=f"{key_prefix}_choice")
    fs = viz_cfg['field_settings'][field_choice]
    auto = st.checkbox("Auto range (2-98 percentile)",
                        value=fs.get('auto', True),
                        key=f"{key_prefix}_auto_{field_choice}")
    vmin_in, vmax_in = None, None
    if not auto:
        c1, c2 = st.columns(2)
        drng = FIELD_DEFAULT_RNG.get(field_choice, (0.0, 1.0))
        with c1:
            vmin_in = st.number_input(
                "Min (vmin)",
                value=float(fs.get('vmin') if fs.get('vmin') is not None
                             else drng[0]),
                key=f"{key_prefix}_vmin_{field_choice}", format="%.6g")
        with c2:
            vmax_in = st.number_input(
                "Max (vmax)",
                value=float(fs.get('vmax') if fs.get('vmax') is not None
                             else drng[1]),
                key=f"{key_prefix}_vmax_{field_choice}", format="%.6g")
    log_scale = st.checkbox("Log scale", value=fs.get('log', False),
                              key=f"{key_prefix}_log_{field_choice}")
    invert = st.checkbox("Invert colormap", value=fs.get('invert', False),
                           key=f"{key_prefix}_inv_{field_choice}")
    cmap_options = ColormapRegistry.list_all()
    current_cmap = fs.get('cmap',
                            FIELD_DEFAULT_CMAPS.get(field_choice, 'viridis'))
    cmap_idx = (cmap_options.index(current_cmap)
                if current_cmap in cmap_options else 0)
    cmap_choice = st.selectbox(
        f"Colormap for '{field_choice}'", cmap_options, index=cmap_idx,
        key=f"{key_prefix}_cmap_{field_choice}")
    if st.button("Save field settings", key=f"{key_prefix}_save"):
        viz_cfg['field_settings'][field_choice] = {
            'auto': auto, 'vmin': vmin_in, 'vmax': vmax_in,
            'log': log_scale, 'invert': invert, 'cmap': cmap_choice,
        }
        st.success(f"Saved settings for '{field_choice}'.")
        st.rerun()


def render_diagnostics_figure(solver, params, viz_cfg):
    """Render the 6-panel diagnostics time-series figure."""
    fc = viz_cfg.get('figure_customization', {})
    fig_w = float(fc.get('diagnostics_width', 13.0))
    fig_h = float(fc.get('diagnostics_height', 11.0))
    t_unit = get_time_unit_label(params)
    t = np.array(solver.history['time'])

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1],
                  hspace=0.45, wspace=0.30)
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ])

    c1 = fc.get('color_series_1', '#1f77b4')
    c2 = fc.get('color_series_2', '#d62728')
    c3 = fc.get('color_series_3', '#2ca02c')
    c4 = fc.get('color_series_4', '#9467bd')
    c5 = fc.get('color_series_5', '#000000')
    lw_main = float(fc.get('line_width', 2.0))
    lw_minor = float(fc.get('line_width_minor', 1.5))
    ls_main = fc.get('line_style', '-')
    ls_minor = fc.get('line_style_minor', '--')
    t_xlabel = fc.get('xlabel_override', '').strip() or f't ({t_unit})'

    # Panel 1: eta in GB vs bulk
    axes[0, 0].plot(t, solver.history['eta_gb_mean'], color=c1,
                     linestyle=ls_main, lw=lw_main, label='eta at GBs')
    axes[0, 0].plot(t, solver.history['eta_bulk_mean'], color=c2,
                     linestyle=ls_main, lw=lw_main, label='eta in bulk')
    apply_axis_styling(axes[0, 0], fc,
                        title='Li propagation: GBs vs bulk',
                        xlabel=t_xlabel, ylabel='mean eta',
                        force_aspect='auto')
    apply_legend_styling(axes[0, 0], fc)

    # Panel 2: c_li in GB vs bulk
    axes[0, 1].plot(t, solver.history['c_li_gb_mean'], color=c1,
                     linestyle=ls_main, lw=lw_main, label='c_Li+ at GBs')
    axes[0, 1].plot(t, solver.history['c_li_bulk_mean'], color=c2,
                     linestyle=ls_main, lw=lw_main, label='c_Li+ in bulk')
    apply_axis_styling(axes[0, 1], fc,
                        title='Li+ concentration: GBs vs bulk',
                        xlabel=t_xlabel, ylabel='mean c_Li+',
                        force_aspect='auto')
    apply_legend_styling(axes[0, 1], fc)

    # Panel 3: dendrite depth + Li volume fraction
    axes[1, 0].plot(t, solver.history['max_dendrite_depth_um'], color=c3,
                     linestyle=ls_main, lw=lw_main, label='Max depth (um)')
    axes[1, 0].plot(t, np.array(solver.history['li_volume_frac']) * 100.0,
                     color=c4, linestyle=ls_minor, lw=lw_minor,
                     label='Li vol %')
    apply_axis_styling(axes[1, 0], fc,
                        title='Dendrite depth + Li volume fraction',
                        xlabel=t_xlabel, ylabel='depth (um) / vol %',
                        force_aspect='auto')
    apply_legend_styling(axes[1, 0], fc)

    # Panel 4: stress evolution
    axes[1, 1].plot(t, solver.history['avg_stress_GPa'], color=c5,
                     linestyle=ls_main, lw=lw_main, label='Avg stress')
    axes[1, 1].plot(t, solver.history['max_stress_GPa'], color=c2,
                     linestyle=ls_minor, lw=lw_minor, label='Max stress')
    apply_axis_styling(axes[1, 1], fc,
                        title='Stress evolution',
                        xlabel=t_xlabel, ylabel='sigma_eq (GPa)',
                        force_aspect='auto')
    apply_legend_styling(axes[1, 1], fc)

    # Panel 5: SEI damage (root-growth / whisker mode driver)
    axes[2, 0].plot(t, solver.history['avg_sei_damage'], color=c1,
                     linestyle=ls_main, lw=lw_main, label='Avg SEI damage')
    axes[2, 0].plot(t, solver.history['max_sei_damage'], color=c2,
                     linestyle=ls_minor, lw=lw_minor, label='Max SEI damage')
    apply_axis_styling(axes[2, 0], fc,
                        title='SEI damage (root-growth driver)',
                        xlabel=t_xlabel, ylabel='SEI damage (0-1)',
                        force_aspect='auto')
    apply_legend_styling(axes[2, 0], fc)

    # Panel 6: salt concentration polarization (low t+ signature)
    axes[2, 1].plot(t, solver.history['c_salt_gb_mean'], color=c1,
                     linestyle=ls_main, lw=lw_main, label='c_salt at GBs')
    axes[2, 1].plot(t, solver.history['c_salt_bulk_mean'], color=c2,
                     linestyle=ls_main, lw=lw_main, label='c_salt in bulk')
    apply_axis_styling(axes[2, 1], fc,
                        title='Salt concentration polarization (low t+)',
                        xlabel=t_xlabel, ylabel='mean c_salt',
                        force_aspect='auto')
    apply_legend_styling(axes[2, 1], fc)

    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Li Dendrite SSB Simulator",
        layout="wide", initial_sidebar_state="expanded")

    viz_cfg = get_viz_config()

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem; text-align: center; margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #1b4f72, #7d3c98);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="main-header">Li Dendrite Growth in Polymer Solid-State '
        'Batteries</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#F0F8FF; padding:1rem; border-radius:8px;
                border-left:5px solid #1b4f72;">
    <b>Physics:</b> Phase-field (Allen-Cahn + Butler-Volmer) + Nernst-Planck
    Li+ diffusion + electric potential + spectral elastoplasticity.<br>
    &bull; Left: <b>Li metal electrode</b> (eta=1).
    &bull; Right: <b>polymer solid electrolyte</b> with Voronoi grains
    (eta=0, phi_g in [0,1]).<br>
    &bull; <b>Hart effective diffusivity</b>:
    D_eff = h(eta) D_Li + (1-h(eta)) [phi_g D_bulk + (1-phi_g) D_gb].<br>
    &bull; Dendrites preferentially propagate along GBs because
    D_gb >> D_bulk.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================
    # SIDEBAR
    # ==============================================================
    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Mode",
                         ["Run New Simulation", "View Saved"],
                         index=0, key="mode")

        if mode == "Run New Simulation":
            st.subheader("Geometry")
            li_frac = st.slider("Li electrode fraction", 0.05, 0.30, 0.10,
                                  0.01,
                                  help="Fraction of domain width occupied "
                                       "by Li electrode (left side).")

            se_geometry = st.selectbox(
                "SE grain geometry",
                ['random_voronoi', 'columnar', 'spherical'],
                index=0, key="se_geometry",
                help="random_voronoi: random seeds (default). "
                     "columnar: vertically elongated grains. "
                     "spherical: packed particles with internal GBs + voids.")

            if se_geometry == 'random_voronoi':
                n_grains = st.slider("Number of grains", 5, 80, 25, 1)
                grain_size_um = st.slider("Target grain size (um)",
                                            5.0, 80.0, 20.0, 1.0)
            elif se_geometry == 'columnar':
                n_grains = st.slider("Number of columns", 3, 30, 10, 1)
                grain_size_um = st.slider("Column width (um)",
                                            5.0, 80.0, 20.0, 1.0)
            else:  # spherical
                n_grains = 0  # not used for spherical
                grain_size_um = st.slider("Internal grain size (um)",
                                            2.0, 30.0, 8.0, 0.5)
                n_particles = st.slider("Number of particles", 1, 12, 4, 1,
                                          key="n_particles")
                particle_radius_um = st.slider("Particle radius (um)",
                                                 10.0, 80.0, 30.0, 1.0,
                                                 key="particle_radius_um")

            gb_width_um = st.slider("GB width (um)", 0.5, 5.0, 1.0, 0.1)
            seed_in = st.number_input("Random seed", value=1234, step=1)

            st.subheader("Initial dendrite")
            initial_dendrite = st.selectbox(
                "Initial dendrite mode",
                ['flat_noise', 'preseed_tips', 'single_notch'],
                index=0, key="initial_dendrite",
                help="flat_noise: small random perturbation (default). "
                     "preseed_tips: pre-place Li protrusions. "
                     "single_notch: one rectangular notch.")
            if initial_dendrite == 'flat_noise':
                noise_amp_um = st.slider("Noise amplitude (um)",
                                           0.0, 15.0, 3.0, 0.1,
                                           help="Perturbation amplitude at "
                                                "the Li/SE interface. Should "
                                                "be >= 2*dx for dendrites to "
                                                "resolve above the critical "
                                                "wavelength.")
                notch_depth_um, notch_width_um, n_preseeds = 5.0, 5.0, 3
            elif initial_dendrite == 'preseed_tips':
                n_preseeds = st.slider("Number of pre-seeds", 1, 8, 3, 1)
                notch_depth_um = st.slider("Tip depth (um)",
                                             2.0, 30.0, 10.0, 1.0)
                notch_width_um = st.slider("Tip width (um)",
                                             2.0, 30.0, 10.0, 1.0)
                noise_amp_um = 0.5
            else:  # single_notch
                notch_depth_um = st.slider("Notch depth (um)",
                                             2.0, 50.0, 10.0, 1.0)
                notch_width_um = st.slider("Notch width (um)",
                                             2.0, 50.0, 10.0, 1.0)
                n_preseeds, noise_amp_um = 3, 0.5

            st.subheader("Solid electrolyte material")
            se_material = st.selectbox(
                "Polymer SE material",
                MaterialProperties.list_se_li_options(),
                index=0, key="se_material",
                help="PEO_Li: standard PEO/LiTFSI (requires >60 C). "
                     "PAN: higher electrochemical stability. "
                     "PPC: amorphous, excellent RT mobility. "
                     "PVDF: high mechanical stiffness suppresses dendrites. "
                     "custom: user-configurable D_bulk and D_gb ratio.")

            # Look up the default transport properties for the chosen
            # material so the sliders below can default to the right
            # values when the user switches between PEO/PAN/PPC/PVDF.
            se_defaults_map = {
                'PEO_Li': {'D_bulk': 5.0e-13,  'D_gb_ratio': 200.0, 'sigma_SE': 1.0e-4},
                'PAN':    {'D_bulk': 2.0e-12,  'D_gb_ratio': 300.0, 'sigma_SE': 5.0e-4},
                'PPC':    {'D_bulk': 1.0e-11,  'D_gb_ratio': 100.0, 'sigma_SE': 1.5e-4},
                'PVDF':   {'D_bulk': 8.0e-13,  'D_gb_ratio': 400.0, 'sigma_SE': 2.0e-4},
                'custom': {'D_bulk': 5.0e-13,  'D_gb_ratio': 200.0, 'sigma_SE': 1.0e-4},
            }
            se_def = se_defaults_map.get(se_material, se_defaults_map['PEO_Li'])

            # --- Always-on SE transport sliders (defaults from chosen material) ---
            # The user explicitly requested that D_bulk and D_gb_ratio be
            # adjustable for ALL SE materials (not just 'custom').  The
            # slider defaults update when the user switches materials.
            custom_D_bulk = st.slider(
                "D_bulk (intrgrain, m^2/s)", 1.0e-14, 1.0e-10,
                float(se_def['D_bulk']),
                format="%.2e", key="custom_D_bulk",
                help="Intragranular Li+ diffusivity in the polymer SE. "
                     "Defaults: PEO_Li=5e-13, PAN=2e-12, PPC=1e-11, "
                     "PVDF=8e-13 m^2/s.")
            custom_D_gb_ratio = st.slider(
                "D_gb / D_bulk ratio", 10.0, 1.0e4,
                float(se_def['D_gb_ratio']),
                format="%.0f", key="custom_D_gb_ratio",
                help="Grain-boundary enhancement factor.  Higher = more "
                     "preferential dendrite propagation along GBs. "
                     "Defaults: PEO_Li=200, PAN=300, PPC=100, PVDF=400.")
            custom_sigma_SE = st.slider(
                "sigma_SE (S/m)", 1.0e-6, 1.0e-1,
                float(se_def['sigma_SE']),
                format="%.2e", key="custom_sigma_SE",
                help="Ionic conductivity of the polymer SE.  Defaults: "
                     "PEO_Li=1e-4, PAN=5e-4, PPC=1.5e-4, PVDF=2e-4 S/m.")

            # --- NEW PHYSICS: transference number, SEI damage ---
            st.subheader("Li-specific transport & SEI")
            transference_number = st.slider(
                "Transference number t+ (Li+)", 0.05, 0.95, 0.30, 0.01,
                key="transference_number",
                help="Li+ transference number in the polymer SE.  PEO/LiTFSI: "
                     "~0.2-0.4 (low — anions carry most of the current, "
                     "causing severe concentration polarization).  "
                     "PAN/PPC/PVDF: ~0.5-0.7.  Low t+ is the primary driver "
                     "of Li dendrite growth in real cells.")
            sei_yield_stress_MPa = st.slider(
                "SEI yield stress (MPa)", 1.0, 500.0, 50.0, 1.0,
                key="sei_yield_stress_MPa",
                help="Hydrostatic stress at which the SEI starts to "
                     "accumulate damage (crack).  Typical Li-SEI: "
                     "10-100 MPa.  Lower = easier cracking = more "
                     "root-growth / whisker mode dendrites.")
            sei_healing_rate = st.slider(
                "SEI healing rate (1/s)", 1.0e-6, 1.0e-2, 1.0e-4,
                format="%.1e", key="sei_healing_rate",
                help="Exponential relaxation rate of SEI damage.  Lower = "
                     "damage persists longer = more whisker growth.")

            st.subheader("Operating temperature")
            temperature_C = st.slider(
                "Operating Temperature (\u00b0C)", 20.0, 90.0, 60.0, 1.0,
                key="temperature_C",
                help="Critical for PEO/LiTFSI: chain segmental motion only "
                     "enables adequate Li+ mobility above ~60 C. "
                     "PAN, PPC, and PVDF can operate at lower T. "
                     "Adjusts both the Nernst-Planck thermal voltage (RT/F) "
                     "and the Butler-Volmer exponential kinetics.")
            temperature_K = temperature_C + 273.15

            st.subheader("Grid + time")
            Lx_um = st.slider("Lx (domain width, um)", 50.0, 400.0, 200.0,
                                5.0, key="Lx_um")
            Ly_um = st.slider("Ly (domain height, um)", 50.0, 400.0, 200.0,
                                5.0, key="Ly_um")
            dx = st.slider("dx (um)", 0.5, 5.0, 1.0, 0.1)
            dy = st.slider("dy (um)", 0.5, 5.0, 1.0, 0.1)

            Nx = int(round(Lx_um / dx))
            Ny = int(round(Ly_um / dy))
            Nx = max(32, min(512, Nx))
            Ny = max(32, min(512, Ny))
            Lx_actual = Nx * dx
            Ly_actual = Ny * dy
            st.caption(f"Actual domain: {Lx_actual:.1f} x {Ly_actual:.1f} um "
                        f"(Nx={Nx}, Ny={Ny}, {Nx*Ny:,} cells)")

            time_unit = st.selectbox("Time unit", list(TIME_UNITS.keys()),
                                       index=0, key="time_unit_sel")
            dt = st.slider(f"dt ({time_unit})", 1.0e-4, 1.0e-1, 1.0e-2,
                            format="%.4f", key="dt_slider")

            st.subheader("Electrochemistry")
            applied_voltage = st.slider("Applied voltage (V)", 0.0, 1.0, 0.2,
                                          0.01)
            L_eta = st.slider("L_eta (interface mobility)", 1.0e-7, 1.0e-4,
                               1.0e-6, format="%.2e")
            L_phi = st.slider("L_phi (reaction prefactor)", 1.0e-4, 1.0e-1,
                                5.0e-3, format="%.2e",
                                help="Butler-Volmer reaction prefactor. "
                                     "Higher = faster plating. Default raised "
                                     "from 1e-3 to 5e-3 to drive visible "
                                     "dendrite growth.")
            kappa = st.slider("kappa (gradient energy)", 1.0e-16, 1.0e-13,
                                2.0e-15, format="%.2e",
                                help="Gradient energy coefficient.  Li has "
                                     "a higher surface energy than Na, so "
                                     "the default is 2e-15 (vs the original "
                                     "Na value of 1e-15).  Higher = flatter "
                                     "interfaces.")
            w_barrier = st.slider("w (barrier height)", 1.0e5, 1.0e7,
                                    2.0e6, format="%.2e",
                                    help="Double-well barrier height.  Li "
                                         "has a higher surface energy than "
                                         "Na, so the default is 2e6 (vs "
                                         "the original Na value of 1e6).")
            alpha_bv = st.slider("alpha (BV symmetry)", 0.1, 0.9, 0.5, 0.05)
            aniso_str = st.slider("Anisotropy strength", 0.0, 0.5, 0.0, 0.02)

            st.subheader("Mechanics")
            applied_stress_MPa = st.slider("Applied stress (MPa)", 0.0, 500.0,
                                              0.0, 10.0)
            theta_deg = st.slider("Loading angle (deg)", 0.0, 180.0, 0.0,
                                    5.0)
            Omega_stress = st.slider("Omega (stress-migration coupling)",
                                       0.0, 0.5, 0.05, 0.005)

            st.subheader("Run control")
            n_steps = st.slider("Steps", 10, 1000, 100, 10)
            save_freq = st.slider("Save every N steps", 1, 50, 5, 1)

            # Build params dict
            params = {
                'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'dt': dt,
                'time_unit': time_unit,
                'li_frac': li_frac,
                'se_geometry': se_geometry,
                'n_grains': n_grains,
                'grain_size_um': grain_size_um,
                'gb_width_um': gb_width_um,
                'n_particles': n_particles if se_geometry == 'spherical' else 4,
                'particle_radius_um': particle_radius_um if se_geometry == 'spherical' else 30.0,
                'seed': int(seed_in),
                'initial_dendrite': initial_dendrite,
                'notch_depth_um': notch_depth_um,
                'notch_width_um': notch_width_um,
                'n_preseeds': n_preseeds,
                'noise_amp_um': noise_amp_um,
                'se_material': se_material,
                'custom_D_bulk': custom_D_bulk,
                'custom_D_gb_ratio': custom_D_gb_ratio,
                'custom_sigma_SE': custom_sigma_SE,
                'transference_number': transference_number,
                'sei_yield_stress_MPa': sei_yield_stress_MPa,
                'sei_healing_rate': sei_healing_rate,
                'temperature_K': temperature_K,
                'temperature_C': temperature_C,
                'L_eta': L_eta, 'L_phi': L_phi, 'kappa': kappa,
                'w_barrier': w_barrier, 'alpha_bv': alpha_bv,
                'aniso_strength': aniso_str,
                'Omega_stress': Omega_stress,
                'applied_voltage': applied_voltage,
                'applied_stress': applied_stress_MPa * 1.0e6,
                'applied_stress_angle': theta_deg,
                'n_steps': n_steps, 'save_frequency': save_freq,
            }

            errors, warns = MaterialProperties.validate_parameters(params)
            for e in errors:
                st.error(e)
            for w in warns:
                st.warning(w)

            if st.button("Initialize", type="primary",
                          use_container_width=True):
                with st.spinner("Building geometry..."):
                    try:
                        solver = LiSSBSolver(params)
                        st.session_state.solver_init = solver
                        st.session_state.params = params
                        st.session_state.initialized = True
                        # Clear any stale history
                        for k in ('history', 'timesteps'):
                            if k in st.session_state:
                                del st.session_state[k]
                        meta = solver.geom_meta
                        st.success(
                            f"Initialized: Li electrode on left, "
                            f"{meta['n_grains_actual']} SE grains on right "
                            f"(geometry={meta['se_geometry']}). "
                            f"Domain {Nx*dx:.1f} x {Ny*dy:.1f} um. "
                            f"T = {params.get('temperature_C', 60.0):.1f} C "
                            f"({params.get('temperature_K', 333.15):.2f} K).")
                    except Exception as e:
                        st.error(f"Init failed: {e}")

        elif mode == "View Saved":
            sims = SimulationDatabase.list_for_ui()
            if not sims:
                st.info("No saved simulations yet.")
            else:
                names = [s['name'] for s in sims]
                chosen = st.selectbox("Select simulation", names)
                st.session_state.view_id = sims[names.index(chosen)]['id']

        with st.expander("Simulation database", expanded=False):
            n_sims = len(SimulationDatabase.all())
            st.caption(f"{n_sims} saved simulations (max {SimulationDatabase.MAX_SIMS})")
            if st.button("Clear all saved"):
                SimulationDatabase.clear_all()
                st.rerun()

    # ==============================================================
    # MAIN AREA
    # ==============================================================
    if mode == "Run New Simulation" and st.session_state.get('initialized'):
        params = st.session_state.params
        Nx, Ny = params['Nx'], params['Ny']
        dx, dy = params['dx'], params['dy']
        solver_init = st.session_state.solver_init

        tabs = st.tabs(["Geometry", "Run", "Viz Settings", "Results",
                          "Line Profiles", "Plotly", "Export"])

        # ----- Geometry tab -----
        with tabs[0]:
            st.header("Initial Li/SE Geometry")
            meta = solver_init.geom_meta
            st.caption(
                f"Li electrode: left {meta['x_li_interface']:.1f} um | "
                f"SE region: {meta['Lx_se']:.1f} um wide | "
                f"SE grains: {meta['n_grains_actual']} | "
                f"GB width: {meta['gb_width_um']:.1f} um | "
                f"Initial dendrite: {meta['initial_dendrite']}")

            extent = [0, Nx * dx, 0, Ny * dy]
            gm = solver_init.grain_map

            # Plot grain map + initial eta
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            cmap_gm = ColormapRegistry.resolve('grain_map')
            im = ax.imshow(gm, extent=extent, origin='lower', cmap=cmap_gm,
                            interpolation='nearest', aspect='equal')
            ax.axvline(meta['x_li_interface'], color='white', lw=2,
                        linestyle='--', label='Li/SE interface')
            ax.set_title(f"Voronoi grain map ({meta['se_geometry']})")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            ax.legend(loc='upper right', framealpha=0.85)

            ax = axes[1]
            fs_eta = viz_cfg['field_settings'].get('eta', {})
            cmap_eta = ColormapRegistry.resolve(
                fs_eta.get('cmap', FIELD_DEFAULT_CMAPS['eta']),
                invert=fs_eta.get('invert', False))
            im = ax.imshow(solver_init.eta, extent=extent, origin='lower',
                            cmap=cmap_eta, vmin=0, vmax=1, aspect='equal',
                            interpolation='bilinear')
            ax.axvline(meta['x_li_interface'], color='white', lw=2,
                        linestyle='--')
            plt.colorbar(im, ax=ax, label='eta (Li metal)')
            ax.set_title("Initial eta (Li order parameter)")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Plot phi_g and phi_gb
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            im = ax.imshow(solver_init.phi_g, extent=extent, origin='lower',
                            cmap='gray', vmin=0, vmax=1, aspect='equal',
                            interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='phi_g (1=interior, 0=GB)')
            ax.set_title("phi_g (grain indicator)")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')

            ax = axes[1]
            im = ax.imshow(solver_init.phi_gb, extent=extent, origin='lower',
                            cmap='gb_hot', vmin=0, vmax=1, aspect='equal',
                            interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='phi_gb (1=GB, 0=interior)')
            ax.set_title("phi_gb (GB indicator)")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Effective diffusivity map
            fig, ax = plt.subplots(figsize=(10, 5))
            D_eff_show = np.log10(np.maximum(solver_init.D_eff, 1.0e-30))
            im = ax.imshow(D_eff_show, extent=extent, origin='lower',
                            cmap='magma', aspect='equal', interpolation='bilinear')
            ax.axvline(meta['x_li_interface'], color='white', lw=2,
                        linestyle='--')
            cbar = plt.colorbar(im, ax=ax, label='log10(D_eff) [m^2/s]')
            ax.set_title("Effective diffusivity D_eff (Hart)")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ----- Run tab -----
        with tabs[1]:
            st.header("Run the Li Dendrite Simulation")
            col1, col2 = st.columns([2, 1])
            with col1:
                se_name = params.get('se_material', 'PEO_Li')
                t_unit = get_time_unit_label(params)
                D_bulk_disp = solver_init.D_SE_bulk
                D_gb_disp = solver_init.D_gb
                st.markdown(
                    f"**Configuration summary:**\n"
                    f"- SE material: `{se_name}` ({solver_init.mat_se.get('name', '?')})\n"
                    f"- Metal electrode: `{solver_init.mat_metal.get('name', 'Li')}`\n"
                    f"- D_bulk = `{D_bulk_disp:.2e}` m^2/s, "
                    f"D_gb = `{D_gb_disp:.2e}` m^2/s "
                    f"(ratio = {D_gb_disp/D_bulk_disp:.0f}x)\n"
                    f"- Transference number t+ = "
                    f"`{solver_init.transference_number:.2f}` "
                    f"(low t+ -> concentration polarization)\n"
                    f"- SEI yield stress = "
                    f"`{solver_init.sei_yield_stress_Pa/1e6:.1f} MPa`, "
                    f"healing rate = "
                    f"`{solver_init.sei_healing_rate:.1e} 1/s`\n"
                    f"- D_Li (metal self-diffusion) = "
                    f"`{solver_init.D_Li:.2e}` m^2/s\n"
                    f"- Operating temperature = "
                    f"`{params.get('temperature_C', 60.0):.1f} C` "
                    f"(`{params.get('temperature_K', 333.15):.2f} K`)\n"
                    f"- Applied voltage = `{params['applied_voltage']:.2f} V`\n"
                    f"- Applied stress = `{params['applied_stress']/1e6:.0f} MPa`\n"
                    f"- Time: `{params['n_steps']}` steps x "
                    f"`{params['dt']:.4g} {t_unit}` = "
                    f"`{params['n_steps']*params['dt']:.3g} {t_unit}` total")
            if st.button("Start Evolution", type="primary",
                          use_container_width=True):
                with st.spinner("Simulating Li dendrite growth..."):
                    try:
                        # Re-create solver fresh from params
                        solver = LiSSBSolver(params)
                        n_steps = params['n_steps']
                        save_freq = params['save_frequency']
                        history, timesteps = [], []
                        progress = st.progress(0)
                        status = st.empty()
                        for step in range(n_steps):
                            res = solver.step()
                            if res is FAILED or res is None:
                                st.error(f"Step failed: {FAILED.reason}")
                                break
                            if step % save_freq == 0 or step == n_steps - 1:
                                history.append({
                                    k: (v.copy() if isinstance(v, np.ndarray)
                                        else (dict(v) if isinstance(v, dict)
                                              else v))
                                    for k, v in res.items()
                                })
                                timesteps.append(step * params['dt'])
                            progress.progress((step + 1) / n_steps)
                            if step % max(1, n_steps // 10) == 0:
                                depth = solver.history['max_dendrite_depth_um'][-1]
                                eta_gb = solver.history['eta_gb_mean'][-1]
                                eta_b = solver.history['eta_bulk_mean'][-1]
                                status.text(
                                    f"Step {step+1}/{n_steps} | "
                                    f"dendrite depth = {depth:.2f} um | "
                                    f"eta(GB)={eta_gb:.3f} | eta(bulk)={eta_b:.3f}")
                        st.success(f"Done. {len(history)} frames saved.")
                        st.session_state.history = history
                        st.session_state.timesteps = timesteps
                        st.session_state.solver = solver
                        sid = SimulationDatabase.save(
                            params, history,
                            metadata={'created_at': datetime.now().isoformat()},
                            run_time=n_steps * params['dt'])
                        st.session_state.last_sid = sid
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        st.text(traceback.format_exc())

        # ----- Viz Settings tab -----
        with tabs[2]:
            st.header("Visualization Settings")
            sub1, sub2 = st.tabs(["Field scaling", "Overlay & vector"])
            with sub1:
                render_field_settings_panel(viz_cfg, key_prefix='fs_main')
            with sub2:
                st.subheader("Overlay")
                ov = viz_cfg['overlay']
                field_keys = list(FIELD_DEFAULT_CMAPS.keys())
                c1, c2 = st.columns(2)
                with c1:
                    base_idx = (field_keys.index(ov['base'])
                                if ov['base'] in field_keys else 0)
                    ov_base = st.selectbox("Base field", field_keys,
                                             index=base_idx, key="ov_base")
                with c2:
                    over_idx = (field_keys.index(ov['overlay'])
                                if ov['overlay'] in field_keys else 5)
                    ov_over = st.selectbox("Overlay field", field_keys,
                                             index=over_idx, key="ov_over")
                ov_alpha = st.slider("Alpha", 0.05, 1.0,
                                       float(ov['alpha']), 0.05)
                ov_mode = st.radio("Mode", ['contour', 'heatmap'], index=0)
                ov_nc = st.slider("Contour levels", 3, 30,
                                    int(ov['n_contours']), 1)
                if st.button("Save overlay settings"):
                    viz_cfg['overlay'] = {
                        'enabled': True, 'base': ov_base,
                        'overlay': ov_over, 'alpha': ov_alpha,
                        'mode': ov_mode, 'n_contours': ov_nc,
                        'show_grain_ghost': False,
                    }
                    st.success("Saved.")
                    st.rerun()

                st.subheader("Vector field")
                vv = viz_cfg['vector']
                vv_source = st.selectbox("Source field",
                                          VECTOR_FIELD_SOURCES,
                                          index=VECTOR_FIELD_SOURCES.index(
                                              vv.get('source', 'c_li')),
                                          key="vv_source")
                vv_mode = st.selectbox("Mode",
                                        ['gradient', 'flux', 'principal_stress'],
                                        index=0, key="vv_mode")
                vv_density = st.slider("Arrow density", 6, 60, 24, 2)
                vv_scale = st.slider("Arrow scale", 0.1, 5.0, 1.0, 0.1)
                if st.button("Save vector settings"):
                    viz_cfg['vector'] = {
                        'enabled': True, 'source': vv_source,
                        'mode': vv_mode, 'density': vv_density,
                        'color': 'white', 'scale': vv_scale,
                    }
                    st.success("Saved.")
                    st.rerun()

        # ----- Results tab -----
        with tabs[3]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                solver = st.session_state.solver
                idx = st.slider("Frame", 0, len(history) - 1,
                                  len(history) - 1, key="res_frame")
                vis = LiSSBVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {
                    'field_settings': viz_cfg['field_settings'],
                    'title_font_size': viz_cfg['title_font_size'],
                    'label_font_size': viz_cfg['label_font_size'],
                    'colorbar_fraction': viz_cfg['colorbar_fraction'],
                    'colorbar_pad': viz_cfg['colorbar_pad'],
                    'figure_customization': viz_cfg.get(
                        'figure_customization', {}),
                }
                mode_res = st.radio("Display mode",
                                     ["Multi-field grid", "Overlay",
                                      "Vector field"],
                                     horizontal=True, key="res_mode")

                # Inline size controls
                if mode_res == "Multi-field grid":
                    sz = ('multi_field_width', 'multi_field_height')
                elif mode_res == "Overlay":
                    sz = ('overlay_width', 'overlay_height')
                else:
                    sz = ('vector_width', 'vector_height')
                render_inline_figure_customization(
                    viz_cfg, size_keys=sz,
                    key_prefix=f'res_{mode_res[:3]}',
                    label=f'{mode_res} size')
                style['figure_customization'] = viz_cfg.get(
                    'figure_customization', {})

                if mode_res == "Multi-field grid":
                    fig = vis.create_multi_field_comparison(
                        history[idx], style)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                elif mode_res == "Overlay":
                    ov = viz_cfg['overlay']
                    fig = vis.create_overlay(
                        history[idx],
                        base_field=ov['base'],
                        overlay_field=ov['overlay'],
                        alpha=ov['alpha'],
                        mode=ov['mode'],
                        n_contours=ov['n_contours'],
                        style_params=style)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    vv = viz_cfg['vector']
                    base_choice = ('sigma_h'
                                    if vv['mode'] == 'principal_stress'
                                    else 'eta')
                    fig = vis.create_vector_overlay(
                        history[idx],
                        source=vv['source'],
                        mode=vv['mode'],
                        density=vv['density'],
                        color=vv['color'],
                        scale=vv['scale'],
                        base_field=base_choice,
                        style_params=style)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)

                # Diagnostics
                st.subheader("Diagnostics (time series)")
                fc_diag = viz_cfg.setdefault(
                    'figure_customization', default_figure_customization())
                c1, c2 = st.columns(2)
                with c1:
                    new_w = _clamped_number_input(
                        "Width (in)", value=fc_diag.get('diagnostics_width', 13.0),
                        min_value=4.0, max_value=30.0, step=0.5,
                        key="diag_w")
                with c2:
                    new_h = _clamped_number_input(
                        "Height (in)", value=fc_diag.get('diagnostics_height', 8.0),
                        min_value=3.0, max_value=30.0, step=0.5,
                        key="diag_h")
                fc_diag['diagnostics_width'] = float(new_w)
                fc_diag['diagnostics_height'] = float(new_h)
                render_diagnostics_figure(solver, params, viz_cfg)

        # ----- Line Profiles tab -----
        with tabs[4]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                profiler = EnhancedLineProfiler(Nx, Ny, dx, dy)
                idx = st.slider("Frame", 0, len(history) - 1,
                                  len(history) - 1, key="lp_frame")
                field = st.selectbox("Field",
                                      list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=0, key="lp_field")
                ptype = st.selectbox("Profile direction",
                                      ["horizontal", "vertical",
                                       "diagonal", "anti_diagonal"],
                                      key="lp_ptype")
                pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05)
                d, unit = _rescale_field(field, history[idx][field])
                out = profiler.extract_profile(d, ptype, pos)
                if out is FAILED or out is None:
                    st.warning("Profile extraction failed.")
                else:
                    dist, prof, _ = out
                    if dist is not None:
                        fig, ax = plt.subplots(figsize=(11, 4.5))
                        ax.plot(dist, prof, lw=2.0,
                                 color='#1f77b4')
                        ax.set_title(f'{field} | {ptype} profile | frame {idx}')
                        ax.set_xlabel('Position (um)')
                        ax.set_ylabel(f'{field} {unit}')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                # Multi-time compare
                st.subheader("Multi-time compare")
                field_mt = st.selectbox("Field",
                                         list(FIELD_DEFAULT_CMAPS.keys()),
                                         index=0, key="lp_mt_field")
                ptype_mt = st.selectbox("Direction",
                                         ["horizontal", "vertical",
                                          "diagonal", "anti_diagonal"],
                                         key="lp_mt_ptype")
                default_frames = list(range(0, len(history),
                                              max(1, len(history) // 6)))
                frames_sel = st.multiselect("Frames",
                                              list(range(len(history))),
                                              default=default_frames)
                use_plotly_mt = st.checkbox("Use Plotly", value=True,
                                              key="lp_mt_plotly")
                if frames_sel:
                    fig_mt = vis.create_multi_time_profiles(
                        history, field_mt, ptype_mt, frames_sel,
                        position_ratio=0.5, use_plotly=use_plotly_mt,
                        viz_cfg=viz_cfg) if 'vis' in dir() else None
                    if fig_mt is None:
                        st.warning("No profiles extracted.")
                    elif use_plotly_mt:
                        st.plotly_chart(fig_mt, use_container_width=True)
                    else:
                        st.pyplot(fig_mt)
                        plt.close(fig_mt)

        # ----- Plotly tab -----
        with tabs[5]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                vis = LiSSBVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                view = st.radio("View",
                                 ["Heatmap (frame)", "Animation",
                                  "Line profiles"], horizontal=True,
                                 key="px_view")
                if view == "Heatmap (frame)":
                    idx = st.slider("Frame", 0, len(history) - 1,
                                      len(history) - 1, key="px_frame")
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_field")
                    cmap = viz_cfg['field_settings'].get(field, {}).get(
                        'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                    fig_h = vis.create_plotly_heatmap(
                        history[idx], field, idx, cmap, viz_cfg=viz_cfg)
                    if fig_h is not None:
                        st.plotly_chart(fig_h, use_container_width=True)
                elif view == "Animation":
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_anim_field")
                    cmap = viz_cfg['field_settings'].get(field, {}).get(
                        'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                    with st.spinner("Building animation..."):
                        fig_a = vis.create_plotly_animation(
                            history, field, params['dt'], cmap, viz_cfg=viz_cfg)
                    if fig_a is not None:
                        st.plotly_chart(fig_a, use_container_width=True)
                else:
                    idx = st.slider("Frame", 0, len(history) - 1,
                                      len(history) - 1, key="px_lp_frame")
                    field = st.selectbox("Field",
                                          list(FIELD_DEFAULT_CMAPS.keys()),
                                          index=0, key="px_lp_field")
                    ptypes = st.multiselect(
                        "Directions",
                        ["horizontal", "vertical", "diagonal",
                         "anti_diagonal"],
                        default=["horizontal", "vertical"])
                    pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05)
                    if ptypes:
                        fig = go.Figure()
                        d, unit = _rescale_field(field, history[idx][field])
                        for pt in ptypes:
                            res = vis.line_profiler.extract_profile(d, pt, pos)
                            if res is FAILED or res is None:
                                continue
                            dist, prof, _ = res
                            if dist is None:
                                continue
                            fig.add_trace(go.Scatter(
                                x=dist, y=prof, mode='lines',
                                name=pt.replace('_', ' ').title()))
                        fig.update_layout(
                            title=f"{field} - frame {idx}",
                            xaxis_title="Position (um)",
                            yaxis_title=f"{field}",
                            template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

        # ----- Export tab -----
        with tabs[6]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                params = st.session_state.params
                history = st.session_state.history
                sim_data = {'metadata':
                            {'created_at': datetime.now().isoformat()}}
                name = (f"li_dendrite_ssb_Nx{params['Nx']}_Ny{params['Ny']}"
                        f"_{params['se_material']}_seed{params['seed']}")

                st.subheader("Data export")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Pickle"):
                        buf, fn = DataExporter.export_pkl(
                            sim_data, params, history, name)
                        if buf:
                            st.download_button("Download .pkl", buf, fn)
                    if st.button("JSON"):
                        buf, fn = DataExporter.export_json(
                            sim_data, params, history, name)
                        if buf:
                            st.download_button("Download .json", buf, fn)
                with col2:
                    if st.button("CSV (zip)"):
                        extent = [0, Nx * dx, 0, Ny * dy]
                        buf, fn = DataExporter.export_csv(
                            history, name, extent, Nx, Ny, dx, dy)
                        if buf:
                            st.download_button("Download .zip", buf, fn)
                    if st.button("HDF5"):
                        buf, fn = DataExporter.export_hdf5(
                            sim_data, params, history, name, Nx, Ny, dx, dy)
                        if buf:
                            st.download_button("Download .h5", buf, fn)

                st.markdown("---")
                st.subheader("Animation export")
                anim_field = st.selectbox("Field",
                                           list(FIELD_DEFAULT_CMAPS.keys()),
                                           index=0, key="anim_field")
                c1, c2, c3 = st.columns(3)
                with c1:
                    a_fmt = st.selectbox("Format", ['gif', 'mp4'], index=0)
                with c2:
                    a_fps = st.slider("FPS", 1, 30, 5, 1)
                with c3:
                    a_dpi = st.slider("DPI", 72, 300, 120, 12)
                if st.button("Generate animation"):
                    fs_anim = viz_cfg['field_settings'].get(anim_field, {})
                    cmap = fs_anim.get('cmap',
                                        FIELD_DEFAULT_CMAPS.get(anim_field,
                                                                  'viridis'))
                    vis_exp = LiSSBVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                    buf = vis_exp.create_animation_buffer(
                        history, anim_field, params['dt'],
                        output_format=a_fmt, fps=a_fps, dpi=a_dpi,
                        cmap_name=cmap, viz_cfg=viz_cfg)
                    if buf is not None:
                        st.download_button(
                            f"Download .{a_fmt}", buf,
                            f"{name}_{anim_field}.{a_fmt}")
                    else:
                        st.error("Animation failed (need ffmpeg for mp4).")

    elif mode == "View Saved" and st.session_state.get('view_id'):
        sid = st.session_state.view_id
        sim = SimulationDatabase.get(sid)
        if sim is None:
            st.error("Simulation not found.")
        else:
            st.header(f"Simulation: {sid}")
            params = sim['params']
            history = sim['results_history']
            Nx = params['Nx']
            Ny = params['Ny']
            dx = params['dx']
            dy = params.get('dy', dx)
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    v = v.item()
                safe_params[k] = v
            st.json(safe_params)
            if history:
                idx = st.slider("Frame", 0, len(history) - 1,
                                  len(history) - 1, key=f"view_frame_{sid}")
                vis = LiSSBVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {
                    'field_settings': viz_cfg['field_settings'],
                    'title_font_size': viz_cfg['title_font_size'],
                    'label_font_size': viz_cfg['label_font_size'],
                    'colorbar_fraction': viz_cfg['colorbar_fraction'],
                    'colorbar_pad': viz_cfg['colorbar_pad'],
                    'figure_customization': viz_cfg.get(
                        'figure_customization', {}),
                }
                fig = vis.create_multi_field_comparison(history[idx], style)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
            if st.button("Delete"):
                SimulationDatabase.delete(sid)
                if 'view_id' in st.session_state:
                    del st.session_state.view_id
                st.rerun()


if __name__ == "__main__":
    main()
