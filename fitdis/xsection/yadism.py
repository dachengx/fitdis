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
    "XSHERACC": 4.0,
}


class YadismModel:
    def __init__(
        self,
        E_v,
        Z=18,
        A=40,
        observable_card=dict(),
        theory_card=dict(),
        observables="XSHERACC",
        heaviness="light",
        FNS="FFNS",
        n_low=30,
        n_mid=20,
        n_high=0,
        x_min=1e-7,
        x=np.linspace(1e-7, 1.0, 11),
        y=np.linspace(1e-7, 1.0, 11),
    ):
        self.E_v = float(E_v)
        self.Z = Z
        self.A = A
        self.x = x
        self.y = y

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
        self.out = dict()

    @property
    def suffix(self):
        return f"_{self.observable}_{self.theory_card['FNS']}_{self.E_v:.2f}.pineappl.lz4"

    @property
    def filenames(self):
        return [f"{target}{self.suffix}" for target in ["proton", "neutron"]]

    @property
    def exists(self):
        return all([os.path.exists(filename) for filename in self.filenames])

    def avg_isotope(self, results):
        avg_results = results["proton"] * self.Z + results["neutron"] * (self.A - self.Z)
        avg_results /= self.A
        avg_results *= reverted_norm[self._observables]
        return avg_results

    @property
    def s(self):
        return 2 * self.theory_card["MP"] * self.E_v

    def get_Q2(self, x, y):
        Q2 = (self.s - self.theory_card["MP"] ** 2) * x * y
        return Q2

    def get_observables(self, x=None, y=None):
        if isinstance(x, (int, float)):
            x = [x]
        if isinstance(y, (int, float)):
            y = [y]

        if x is None:
            X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        else:
            if y is None:
                raise ValueError("y must be provided if x is provided.")
            X, Y = np.meshgrid(x, y, indexing="ij")
        Q2 = self.get_Q2(X, Y)

        observables = []
        for _x, _y, _Q2 in zip(X.ravel(), Y.ravel(), Q2.ravel()):
            observables.append({"x": _x, "y": _y, "Q2": _Q2})
        return observables

    def run(self, x=None, y=None):
        _observable_card = deepcopy(self.observable_card)
        _observable_card["observables"] = {self.observable: self.get_observables(x=x, y=y)}

        # Scattering target: "proton", "neutron", "isoscalar", "lead", "iron", "neon" or "marble"
        for target in ["proton", "neutron"]:
            _observable_card["TargetDIS"] = target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # skip noisy warnings
                self.out[target] = yadism.run_yadism(self.theory_card, _observable_card)

    def apply(self, lhaid="NNPDF40_nnlo_as_01180", member=0):
        results = dict()
        pdf = load_pdf(lhaid, member)
        for target in ["proton", "neutron"]:
            results[target] = self.out[target].apply_pdf(pdf)
            results[target] = np.array([r["result"] for r in results[target][self.observable]])
        avg_results = self.avg_isotope(results)
        return avg_results

    def dump(self, replace=False):
        for target in ["proton", "neutron"]:
            filename = f"{target}{self.suffix}"
            if os.path.exists(filename) and not replace:
                raise FileExistsError(
                    f"{filename} already exists. Set replace=True to overwrite it."
                )
            dump_pineappl_to_file(self.out[target], filename, self.observable)

    def load(self, pid=2212, lhaid="NNPDF40_nnlo_as_01180", member=0):
        results = dict()
        pdf = load_pdf(lhaid, member)
        for target in ["proton", "neutron"]:
            filename = f"{target}{self.suffix}"
            if not os.path.exists(filename):
                raise FileNotFoundError(f"{filename} does not exist.")
            grid = pineappl.grid.Grid.read(filename)
            results[target] = np.array(grid.convolve_with_one(pid, pdf.xfxQ2, pdf.alphasQ2))
        avg_results = self.avg_isotope(results)
        return avg_results

    def d2sdxdy(self, pid=2212, lhaid="NNPDF40_nnlo_as_01180", member=0):
        # in 10^-38 cm^2
        form = self.load(lhaid=lhaid, member=member)
        form = form.reshape((len(self.x), len(self.y)))
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        Q2 = self.get_Q2(X, Y)
        r = self.theory_card["GF"] ** 2 * self.s
        r /= 4 * np.pi * (1 + Q2 / self.theory_card["MW"] ** 2) ** 2
        r *= form * GEV_CM2_CONV
        return r

    def sigma(self, pid=2212, lhaid="NNPDF40_nnlo_as_01180", member=0):
        d2sdxdy = self.d2sdxdy(pid=pid, lhaid=lhaid, member=member)
        if np.std(np.diff(self.x)) > 1e-4 or np.std(np.diff(self.y)) > 1e-4:
            raise ValueError("x and y must be evenly spaced.")
        dx = np.diff(self.x)[0]
        dy = np.diff(self.y)[0]
        return np.sum(d2sdxdy) * dx * dy
