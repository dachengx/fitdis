import numpy as np


E_vs = np.linspace(1.0, 8.0, 8)

MASS = 10e3  # tonne
BASELINE = 1300  # km
ARGON_MOL_MASS = 40  # g/mol
AVOGADRO = 6.02e23  # mol-1
POT_PER_YEAR = 1.1e21

# GeV-1 cm-2 POT-1
# https://arxiv.org/abs/2002.03005 Fig. 4.10
ND_FLUX = 10 ** np.array(
    [
        -7.632699253771122,
        -7.525527023889229,
        -7.596600045780022,
        -8.03012667102482,
        -8.729032802149964,
        -9.14160556076473,
        -9.241528053241701,
        -9.296836835767206,
    ]
)

# https://arxiv.org/abs/2002.03005 Fig. 4.8
ND_FD_RATIO = 10e-6 * np.array(
    [
        0.13279713924535858,
        0.1253973808718798,
        0.19525414932174434,
        0.25896666975339877,
        0.12934588556550708,
        0.12037835439848554,
        0.13846976330507177,
        0.1420517365021869,
    ]
)

SIN_2_THETA_12 = 3.07e-1
SIN_2_THETA_23 = 5.72e-1
SIN_2_THETA_13 = 2.20e-2
DELTA_M_2_21 = 7.41e-5  # eV^2
DELTA_M_2_32 = 2.51e-3  # eV^2

A_L = BASELINE / 3500


def appearance(
    E_v,
    sin_2_theta_12=SIN_2_THETA_12,
    sin_2_theta_23=SIN_2_THETA_23,
    sin_2_theta_13=SIN_2_THETA_13,
    delta_m_2_21=DELTA_M_2_21,
    delta_m_2_32=DELTA_M_2_32,
    delta_cp=0,
):
    d2_31 = 1.27 * (delta_m_2_32 + delta_m_2_21) * BASELINE / E_v
    d2_21 = 1.27 * delta_m_2_21 * BASELINE / E_v
    ps = [
        sin_2_theta_23
        * (1 - (1 - 2 * sin_2_theta_13) ** 2)
        * (np.sin(d2_31 - A_L) / (d2_31 - A_L)) ** 2
        * d2_31**2,
        (1 - (1 - 2 * sin_2_theta_23) ** 2) ** 0.5
        * (1 - (1 - 2 * sin_2_theta_13) ** 2) ** 0.5
        * (1 - (1 - 2 * sin_2_theta_12) ** 2) ** 0.5
        * np.sin(d2_31 - A_L)
        / (d2_31 - A_L)
        * d2_31
        * (np.sin(A_L) / A_L)
        * d2_21
        * np.cos(d2_31 + delta_cp),
        (1 - sin_2_theta_23)
        * (1 - (1 - 2 * sin_2_theta_12) ** 2)
        * (np.sin(A_L) / A_L) ** 2
        * d2_21**2,
    ]
    p = np.sum(ps, axis=0)
    return p
