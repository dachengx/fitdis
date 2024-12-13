from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")

import numpy as np
import matplotlib.pyplot as plt
from fitdis.xsection.nnsfnu import d2sdxdy


E_vs = np.linspace(1.0, 20.0, 20)
x = np.linspace(1e-6, 1.0, 101)
y = np.linspace(1e-6, 1.0, 101)
X, Y = np.meshgrid(x, y, indexing="ij")

pdfs = dict()
for E_v in E_vs:
    _d2sdxdy = d2sdxdy(E_v, member=0, x=x, y=y)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    pc = ax.pcolormesh(X, Y, _d2sdxdy)
    c = fig.colorbar(pc, ax=ax, aspect=50)
    c.ax.set_ylabel(r"$\frac{\mathrm{d}^2\sigma}{\mathrm{d}x\mathrm{d}y}$ [$10^{-38}$ cm$^2$]")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # fig.savefig(f"plots/d2sdxdy_{E_v:.2f}.svg", transparent=True, dpi=800)
    fig.savefig(f"plots/d2sdxdy_{E_v:.2f}.png", transparent=True, dpi=800)
    plt.close(fig)
