"""Steel section library for CivilAI-FEA.

Units throughout the project: length [m], force [kN], stress [kN/m2].
1 MPa = 1000 kN/m2.

Section properties are derived from nominal AISC W-shape dimensions
(A, d, bf, Ix, Zx, tw, tf) taken from standard tables. Secondary
properties (Iy, J, Sx, Sy, Zy, Cw) are computed from those dimensions
(fillets neglected), which is accurate to a few percent -- adequate for
analysis and optimization studies, NOT for construction documents.
"""
from dataclasses import dataclass

import numpy as np

IN = 0.0254          # m per inch
IN2, IN3, IN4 = IN**2, IN**3, IN**4


@dataclass(frozen=True)
class Section:
    name: str
    A: float    # cross-section area [m2]
    Ix: float   # strong-axis second moment of area [m4]
    Iy: float   # weak-axis second moment of area [m4]
    J: float    # St-Venant torsion constant [m4]
    Sx: float   # strong-axis elastic section modulus [m3]
    Sy: float   # weak-axis elastic section modulus [m3]
    Zx: float   # strong-axis plastic section modulus [m3]
    Zy: float   # weak-axis plastic section modulus [m3]
    d: float    # overall depth [m]
    bf: float   # flange width [m]
    tw: float   # web thickness [m]
    tf: float   # flange thickness [m]

    @property
    def Cw(self) -> float:
        # warping constant of an I-section (approximate)
        return self.Iy * (self.d - self.tf) ** 2 / 4.0

    @property
    def hw(self) -> float:
        return self.d - 2.0 * self.tf


def _w(name, A, d, bf, Ix, Zx, tw, tf):
    """Build a Section from tabulated values (inches) + derived properties."""
    d_, bf_, tw_, tf_ = d * IN, bf * IN, tw * IN, tf * IN
    A_, Ix_, Zx_ = A * IN2, Ix * IN4, Zx * IN3
    Iy = (2.0 * tf * bf**3 + (d - 2.0 * tf) * tw**3) / 12.0 * IN4
    J = (2.0 * bf * tf**3 + (d - tf) * tw**3) / 3.0 * IN4
    Sx = Ix_ / (d_ / 2.0)
    Sy = Iy / (bf_ / 2.0)
    Zy = (bf**2 * tf / 2.0 + (d - 2.0 * tf) * tw**2 / 2.0) * IN3
    return Section(name, A_, Ix_, Iy, J, Sx, Sy, Zx, Zy, d_, bf_, tw_, tf_)


# --- Column pool (W14 family + light W12) -----------------------------------
COLUMN_POOL = [
    _w("W12x26",  7.65, 12.2,  6.49,  204,  37.2, 0.23, 0.380),
    _w("W14x22",  6.49, 13.7,  5.00,  199,  33.2, 0.20, 0.335),
    _w("W14x53", 15.60, 13.9,  8.06,  541,  87.1, 0.37, 0.660),
    _w("W14x73", 21.50, 14.2, 10.00,  796, 126.0, 0.43, 0.780),
    _w("W14x90", 26.50, 14.4, 14.52,  999, 157.0, 0.44, 0.710),
    _w("W14x120", 35.30, 14.5, 14.67, 1380, 212.0, 0.59, 0.940),
    _w("W14x159", 46.70, 14.98, 15.29, 1870, 284.0, 0.66, 1.190),
    _w("W14x211", 62.00, 15.2, 15.86, 2580, 423.0, 0.78, 1.560),
    _w("W14x233", 68.50, 15.9, 15.86, 2720, 360.0, 0.83, 1.720),
]

# --- Beam pool (floor beams / girders) ---------------------------------------
BEAM_POOL = [
    _w("W12x14",  4.16, 11.9, 3.97,  88.6,  17.4, 0.175, 0.225),
    _w("W16x26",  7.68, 15.7, 5.50, 301.0,  44.2, 0.250, 0.345),
    _w("W18x35", 10.30, 17.7, 6.00, 510.0,  66.5, 0.300, 0.425),
    _w("W18x40", 11.80, 17.9, 6.02, 612.0,  78.4, 0.315, 0.525),
    _w("W18x46", 13.50, 18.1, 6.06, 712.0,  92.7, 0.360, 0.605),
    _w("W18x55", 16.20, 18.1, 7.53, 890.0, 112.0, 0.390, 0.730),
    _w("W21x50", 14.70, 20.8, 8.00, 984.0, 110.0, 0.380, 0.640),
    _w("W24x55", 16.20, 22.9, 7.00, 1350., 134.0, 0.395, 0.505),
    _w("W24x68", 20.10, 23.7, 8.97, 1830., 177.0, 0.415, 0.685),
]


def pool_arrays(pool):
    """Stack a section pool into (n,)-friendly dict of numpy arrays."""
    keys = ("A", "Ix", "Iy", "J", "Sx", "Sy", "Zx", "Zy", "d", "bf", "tw", "tf")
    out = {k: np.array([getattr(s, k) for s in pool]) for k in keys}
    out["Cw"] = np.array([s.Cw for s in pool])
    out["hw"] = np.array([s.hw for s in pool])
    out["names"] = [s.name for s in pool]
    return out


if __name__ == "__main__":
    for s in COLUMN_POOL + BEAM_POOL:
        print(f"{s.name:9s} A={s.A*1e3:7.1f} mm2  Ix={s.Ix*1e6:8.1f} cm4  "
              f"Iy={s.Iy*1e6:7.1f} cm4  J={s.J*1e6:7.2f} cm4  "
              f"Zx={s.Zx*1e3:6.1f} cm3  Zy={s.Zy*1e3:6.1f} cm3")
