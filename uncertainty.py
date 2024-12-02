from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")

import numpy as np

# from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from fitdis.xsection.yadism import YadismModel


E_vs = np.linspace(1.0, 20.0, 20)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

cmap = cm.cool
_norm = Normalize(vmin=0, vmax=100)
sigmas = []
for member in range(101):
    sigma = np.array(
        [YadismModel(E_v, theory_card={"TMC": 0}).sigma(member=member) / E_v for E_v in E_vs]
    )
    ax.plot(
        E_vs,
        sigma / E_vs,
        # color="k" if member == 0 else "gray",
        marker="o",
        alpha=0.5,
        color=cmap(_norm(member)),
    )
    sigmas.append(sigma)
# _sigmas = np.vstack(
#     [
#         np.min(sigmas[1:], axis=0),
#         np.max(sigmas[1:], axis=0),
#         np.quantile(sigmas[1:], norm.cdf([-1, 1]), axis=0),
#         sigmas[0],
#     ]
# )
# np.save("sigmas.npy", _sigmas)
np.save("sigmas.npy", sigmas)
ax.set_xlabel("$E$ [GeV]")
ax.set_ylabel(r"$\sigma/E$ [$10^{-38}$ cm$^2$ / GeV / nucleon]")

fig.savefig("sigma.svg", transparent=True, dpi=800)
plt.close(fig)
