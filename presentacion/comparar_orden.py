"""T creciendo durante la simulación: ¿importa en qué orden se agregan las aristas del árbol?

Dos órdenes que terminan en el MISMO árbol generador (Kruskal aleatorio, caja 9×9, 99 aristas):
  - Kruskal:        las aristas entran en el orden en que Kruskal las acepta (lo del §8.5);
  - árbol revelado: primero se sortea el árbol completo y luego sus aristas se revelan en orden
                    uniforme (una permutación uniforme de las 99 aristas del árbol).
Se agrega una arista cada m eventos. Diseño pareado: la réplica i usa la misma semilla con los
dos órdenes, así que tiene la misma condición inicial y el mismo árbol final; solo cambia el orden.
Las semillas son las del §8.5 (las primeras 600 réplicas de Kruskal son exactamente las del notebook).

Uso:  python presentacion/comparar_orden.py [--recalcular]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estudio_configuracion as ec          # motor del notebook, estilo y paleta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

nb = ec.nb
AQUI = Path(__file__).resolve().parent
BOX = (9, 9)
M_VALUES = [1, 3, 10, 30, 100]
R = 2000
T_STAR = 50.0
MAX_EVENTOS = 3_000_000
CONDICIONES = ["1 partícula", "lazo ∂f"]
ORDENES = ["Kruskal", "árbol revelado"]
COLOR = {"1 partícula": "#1E8C66", "lazo ∂f": "#B5652B"}
ESTILO_LINEA = {"Kruskal": dict(ls="-", marker="o"), "árbol revelado": dict(ls="--", marker="s", mfc="white")}


def orden_arbol(cx, rng, cual):
    """Orden de agregado de las aristas del árbol generador."""
    orden = nb.order_random_tree(cx, rng)
    return orden if cual == "Kruskal" else rng.permutation(orden)


def trozo(ic, cual, m, seeds):
    """Como `growing_chunk` del notebook, con el orden elegido. Fila: (τ, eventos, ¿extinto?, |T| final)."""
    cx = nb.build_cubical_complex(BOX)
    init = nb.init_single_particle if ic == "1 partícula" else nb.init_loop
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta0 = init(cx, rng)
        out.append(nb.run_growing_absorption(theta0, cx, orden_arbol(cx, rng, cual), m, MAX_EVENTOS, rng))
    return out


def calcular():
    res = {}
    for i, ic in enumerate(CONDICIONES):
        for cual in ORDENES:
            for m in M_VALUES:
                t0 = time.time()
                seeds = nb.replica_seeds(nb.SEED + 5000 + 100 * i + m, R)       # semillas del §8.5
                partes = [list(c) for c in np.array_split(np.array(seeds, dtype=object), 40)]
                b = Parallel(n_jobs=-1)(delayed(trozo)(ic, cual, m, c) for c in partes)
                a = np.array([f for x in b for f in x], dtype=float)
                res[(ic, cual, m)] = a
                print(f"{ic:12s} {cual:15s} m={m:3d}  E[τ]={a[:, 0].mean():7.1f}  "
                      f"P(τ≤50)={np.mean(a[:, 2].astype(bool) & (a[:, 0] <= T_STAR)):.3f}  "
                      f"|T| al extinguirse={a[:, 3].mean():5.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def resultados():
    cache = AQUI / "comparar_orden_resultados.pkl"
    if cache.exists() and "--recalcular" not in sys.argv:
        return pickle.loads(cache.read_bytes())
    res = calcular()
    cache.write_bytes(pickle.dumps(res))
    return res


def figura(res):
    with plt.rc_context(ec.ESTILO):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        for ic in CONDICIONES:
            for cual in ORDENES:
                a = [res[(ic, cual, m)] for m in M_VALUES]
                media = np.array([x[:, 0].mean() for x in a])
                ee = np.array([x[:, 0].std(ddof=1) / np.sqrt(len(x)) for x in a])
                axes[0].errorbar(M_VALUES, media, yerr=1.96 * ee, color=COLOR[ic], ms=7, lw=1.6, capsize=3,
                                 **ESTILO_LINEA[cual])
                k = np.array([np.sum(x[:, 2].astype(bool) & (x[:, 0] <= T_STAR)) for x in a])
                p = k / R
                lo, hi = nb.wilson_interval(k, R)
                axes[1].errorbar(M_VALUES, p, yerr=[p - lo, hi - p], color=COLOR[ic], ms=7, lw=1.6, capsize=3,
                                 **ESTILO_LINEA[cual])
        for ax in axes:
            ax.set_xscale("log"); ax.set_xticks(M_VALUES, [str(m) for m in M_VALUES])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("m = eventos entre aristas agregadas")
        axes[0].set_yscale("log"); axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:g}")
        axes[0].set_ylabel(r"$\mathbb{E}[\tau]$"); axes[0].set_title("Tiempo medio de extinción")
        axes[1].set_ylabel(r"$\hat{\mathbb{P}}(\tau\leq 50)$"); axes[1].set_title("Extinción antes de t = 50")
        h = [Line2D([], [], color=COLOR[ic], lw=2.5, label=ic) for ic in CONDICIONES] + \
            [Line2D([], [], color="#8C7B70", lw=1.6, label=f"orden {cual}", **ESTILO_LINEA[cual]) for cual in ORDENES]
        fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=11.5)
        plt.tight_layout()
        fig.savefig(ec.FIGURAS / "comparar_orden.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    figura(resultados())
