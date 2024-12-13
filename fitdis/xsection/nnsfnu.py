import numpy as np
from yadism.esf.exs import GEV_CM2_CONV


from ..utils import load_pdf


GF = 1.1663787e-05
MP = 0.938
MW = 80.398
M2W = MW**2

PDF = dict()


def d2sdxdy(
    E_v,
    member=0,
    lhaid="NNSFnu_Ar_lowQ",
    x=np.linspace(1e-6, 1.0, 101),
    y=np.linspace(1e-6, 1.0, 101),
):
    X, Y = np.meshgrid(x, y, indexing="ij")

    s = 2 * MP * E_v  # + MP ** 2
    Q2 = (s - MP**2) * X * Y
    ys = np.array([1 + (1 - Y.ravel()) ** 2, -Y.ravel() ** 2, 1 - (1 - Y.ravel()) ** 2])
    norm = GF**2 * s
    norm /= 4 * np.pi * (1 + Q2 / M2W) ** 2
    norm *= GEV_CM2_CONV

    if member in PDF:
        pdf = PDF[member]
    else:
        pdf = load_pdf(lhaid, member)
        PDF[member] = pdf
    form = pdf.xfxQ2(X.ravel(), Q2.ravel())
    form = np.array([[f[1001], f[1002], f[1003]] for f in form])
    form *= ys.T
    form = form.sum(axis=1).reshape(X.shape)
    return form * norm


def sigma(
    E_v,
    member=0,
    lhaid="NNSFnu_Ar_lowQ",
    x=np.linspace(1e-6, 1.0, 101),
    y=np.linspace(1e-6, 1.0, 101),
):
    d2sdxdy_ = d2sdxdy(E_v, member=member, lhaid=lhaid)
    return np.trapz(np.trapz(d2sdxdy_, x=x, axis=0), x=y)
