from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")

import numpy as np
import matplotlib.pyplot as plt
from fitdis.xsection.yadism import YadismModel


E_vs = np.linspace(1.0, 20.0, 20)

for E_v in E_vs:
    m = YadismModel(E_v, theory_card={"TMC": 0})
    if not m.exists:
        m.run()
        m.dump()
        # d2sdxdy = m.apply()
    d2sdxdy_approx = m.d2sdxdy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    X, Y = np.meshgrid(m.x, m.y, indexing="ij")
    # pc = ax.pcolormesh(X, Y, (d2sdxdy - d2sdxdy_approx).reshape(X.shape))
    pc = ax.pcolormesh(X, Y, d2sdxdy_approx.reshape(X.shape))
    c = fig.colorbar(pc, ax=ax, aspect=50)
    c.ax.set_ylabel(r"$\frac{\mathrm{d}^2\sigma}{\mathrm{d}x\mathrm{d}y}$ [$10^{-38}$ cm$^2$]")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.savefig(f"d2sdxdy_{E_v:.2f}.svg", transparent=True, dpi=800)
    plt.close(fig)
