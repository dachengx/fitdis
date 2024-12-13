from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from fitdis.xsection.nnsfnu import sigma

n_members = 201
E_vs = np.linspace(1.0, 20.0, 20)

sigmas = []
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for member in tqdm(range(n_members)):
    _sigma = np.array([sigma(E_v, member=member) for E_v in E_vs])
    sigmas.append(_sigma)
_sigmas = np.vstack(
    [
        np.quantile(sigmas[1:], norm.cdf(-1), axis=0),
        sigmas[0],
        np.quantile(sigmas[1:], norm.cdf(1), axis=0),
    ]
)
ax.plot(
    E_vs,
    _sigmas[1] / E_vs,
    marker="o",
    color="k",
)
ax.fill_between(
    E_vs,
    _sigmas[0] / E_vs,
    _sigmas[2] / E_vs,
    alpha=0.3,
    color="k",
)
np.save("sigmas.npy", sigmas)
ax.set_xlabel("$E$ [GeV]")
ax.set_ylabel(r"$\sigma/E$ [$10^{-38}$ cm$^2$ / GeV / nucleon]")
# ax.set_xscale("log")

fig.savefig("plots/sigma.svg", transparent=True, dpi=800)
plt.close(fig)
