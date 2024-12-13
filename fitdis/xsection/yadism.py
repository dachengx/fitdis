import os
import warnings
from copy import deepcopy
import numpy as np
from eko import interpolation
import pineappl
import yadism
from yadism.esf.exs import GEV_CM2_CONV

# import function that dumps the predictions into a Pineappl format
from yadbox.export import dump_pineappl_to_file
from ..utils import load_pdf


default_observable_card = {
    # Process type: "EM", "NC", "CC"
    "prDIS": "CC",
    # Projectile: "electron", "positron", "neutrino", "antineutrino"
    "ProjectileDIS": "neutrino",
    # Interpolation: if True use log interpolation
    "interpolation_is_log": True,
    # Interpolation: polynomial degree, 1 = linear, ...
    "interpolation_polynomial_degree": 4,
    # Projectile polarization faction, float from 0 to 1.
    "PolarizationDIS": 0.0,
    # Exchanged boson propagator correction
    "PropagatorCorrection": 0.0,
    # Restrict boson coupling to a single parton ? Monte Carlo PID or None for all partons
    "NCPositivityCharge": None,
}

default_theory_card = {
    # QCD perturbative order
    "PTO": 2,  # perturbative order in alpha_s: 0 = LO (alpha_s^0), 1 = NLO (alpha_s^1) ...
    # SM parameters and masses
    "CKM": "0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152",  # CKM matrix elements  # noqa
    "GF": 1.1663787e-05,  # [GeV^-2] Fermi coupling constant
    "MP": 0.938,  # [GeV] proton mass
    "MW": 80.398,  # [GeV] W boson mass
    "MZ": 91.1876,  # [GeV] Z boson mass
    "alphaqed": 0.007496252,  # alpha_em value
    "kcThr": 1.0,  # ratio of the charm matching scale over the charm mass
    "kbThr": 1.0,  # ratio of the bottom matching scale over the bottom mass
    "ktThr": 1.0,  # ratio of the top matching scale over the top mass
    "mc": 1.51,  # [GeV] charm mass
    "mb": 4.92,  # [GeV] bottom mass
    "mt": 172.5,  # [GeV] top mass
    # Flavor number scheme settings
    "NfFF": 4,  # (fixed) number of running flavors, only for FFNS or FFN0 schemes
    "Q0": 1.65,  # [GeV] reference scale for the flavor patch determination
    "nf0": 4,  # number of active flavors at the Q0 reference scale
    # Alphas settings and boundary conditions
    "Qref": 91.2,  # [GeV] reference scale for the alphas value
    "nfref": 5,  # number of active flavors at the reference scale Qref
    "alphas": 0.118,  # alphas value at the reference scale
    "MaxNfAs": 5,  # maximum number of flavors in running of strong coupling
    "QED": 0,  # QED correction to running of strong coupling: 0 = disabled, 1 = allowed
    # Scale Variations
    "XIF": 1.0,  # ratio of factorization scale over the hard scattering scale
    "XIR": 1.0,  # ratio of renormalization scale over the hard scattering scale
    # Other settings
    "IC": 1,  # 0 = perturbative charm only, 1 = intrinsic charm allowed
    "TMC": 1,  # include target mass corrections: 0 = disabled, 1 = leading twist, 2 = higher twist approximated, 3 = higher twist exact  # noqa
    "n3lo_cf_variation": 0,  # N3LO coefficient functions variation: -1 = lower bound, 0 = central , 1 = upper bound  # noqa
    # Other EKO settings, not relevant for Yadism
    "HQ": "POLE",  # heavy quark mass scheme (not yet implemented in yadism)
    "MaxNfPdf": 5,  # maximum number of flavors in running of PDFs (ignored by yadism)
    "ModEv": "EXA",  # evolution solver for PDFs (ignored by yadism)
}


reverted_norm = {
    "XSHERACC": lambda Q2, M2W: 1 / 4,
    "XSNUTEVCC": lambda Q2, M2W: 50 / (1 + Q2 / M2W) ** 2,
}


class YadismModel:
    def __init__(
        self,
        E_v,
        Z=18,
        A=39,
        observable_card=dict(),
        theory_card=dict(),
        observables="XSHERACC",
        heaviness="light",
        FNS="FFNS",
        n_low=30,
        n_mid=20,
        n_high=0,
        x_min=1e-6,
        x=np.linspace(1e-6, 1.0, 11),
        y=np.linspace(1e-6, 1.0, 11),
    ):
        self.E_v = float(E_v)
        self.Z = Z
        self.A = A
        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Potentially include observables other than XSHERANCAVG_charm,
        # each of them has to be: TYPE_heaviness, where heaviness can take:
        # "charm", "bottom", "top", "total" or "light".
        self._observables = observables
        self.observable = f"{self._observables}_{heaviness}"

        _observable_card = deepcopy(default_observable_card)
        # Interpolation: xgrid values
        _observable_card["interpolation_xgrid"] = interpolation.make_grid(
            n_low=n_low, n_mid=n_mid, n_high=n_high, x_min=x_min
        ).tolist()
        _theory_card = deepcopy(default_theory_card)
        # Flavour Number Scheme, options: "FFNS", "FFN0", "ZM-VFNS"
        _theory_card["FNS"] = FNS

        self.observable_card = {**_observable_card, **observable_card}
        self.theory_card = {**_theory_card, **theory_card}

    @property
    def TargetDISid(self):
        if self.Z == 1 and self.A == 1:
            return "2212"
        elif self.Z == 0 and self.A == 1:
            return "2112"
        return f"100{self.Z:03d}{self.A:03d}0"

    @property
    def lhaid(self):
        # if self.A == 1:
        #     return "NNPDF40_nnlo_as_01180"
        # elif self.Z == 1 and self.A == 2:
        #     return "NNSFnu_D_lowQ"
        # elif self.Z == 18 and self.A == 39:
        #     return "NNSFnu_Ar_lowQ"
        return "NNPDF40_nnlo_as_01180"

    @property
    def filename(self):
        return f"{self.observable}_{self.theory_card['FNS']}_{self.E_v:.2f}.pineappl.lz4"

    @property
    def exists(self):
        return os.path.exists(self.filename)

    @property
    def s(self):
        return 2 * self.theory_card["MP"] * self.E_v + self.theory_card["MP"] ** 2

    def get_Q2(self, x, y):
        Q2 = (self.s - self.theory_card["MP"] ** 2) * x * y
        return Q2

    def _get_observables(self):
        observables = []
        for _x, _y, _Q2 in zip(self.X.ravel(), self.Y.ravel(), self.get_Q2(self.X, self.Y).ravel()):
            observables.append({"x": _x, "y": _y, "Q2": _Q2})
        return observables

    def run(self):
        _observable_card = deepcopy(self.observable_card)
        _observable_card["observables"] = {self.observable: self._get_observables()}

        # Scattering target: "proton", "neutron", "isoscalar", "lead", "iron", "neon" or "marble"
        _observable_card["TargetDIS"] = {"Z": self.Z, "A": self.A}
        _observable_card["TargetDISid"] = self.TargetDISid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # skip noisy warnings
            self.out = yadism.run_yadism(self.theory_card, _observable_card)

    def dump(self, replace=False):
        if self.exists and not replace:
            raise FileExistsError(
                f"{self.filename} already exists. Set replace=True to overwrite it."
            )
        dump_pineappl_to_file(self.out, self.filename, self.observable)

    @property
    def norm(self):
        n = self.theory_card["GF"] ** 2 * self.s
        M2W = self.theory_card["MW"] ** 2
        Q2 = self.get_Q2(self.X, self.Y)
        n /= 4 * np.pi * (1 + Q2 / M2W) ** 2
        n /= reverted_norm[self._observables](Q2, M2W)
        n *= GEV_CM2_CONV
        return n

    def apply(self, member=0):
        pdf = load_pdf(self.lhaid, member)
        form = self.out.apply_pdf(pdf)
        form = np.array([r["result"] for r in form[self.observable]])
        form = form.reshape(self.X.shape)
        return form * self.norm

    def load(self, member=0):
        pdf = load_pdf(self.lhaid, member)
        if not self.exists:
            raise FileNotFoundError(f"{self.filename} does not exist.")
        grid = pineappl.grid.Grid.read(self.filename)
        form = np.array(grid.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2))
        return form

    def d2sdxdy(self, member=0):
        # in 10^-38 cm^2
        form = self.load(member=member)
        form = form.reshape(self.X.shape)
        return form * self.norm

    def sigma(self, member=0):
        d2sdxdy = self.d2sdxdy(member=member)
        if np.std(np.diff(self.x)) > 1e-4 or np.std(np.diff(self.y)) > 1e-4:
            raise ValueError("x and y must be evenly spaced.")
        dx = np.diff(self.x)[0]
        dy = np.diff(self.y)[0]
        return np.sum(d2sdxdy) * dx * dy
