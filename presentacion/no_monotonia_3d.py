"""«¿Más absorción, extinción más rápida? No siempre» en la caja 5×5×5 (|V| = 216, |E| = 540).

Igual que §8.4 del notebook, pero en Z^3:
  - condición inicial: 1 partícula +1 en la arista central, o lazo ∂f de la cara horizontal central;
  - T = primeras k aristas de la regla (uniforme: orden aleatorio de todas las aristas;
    árbol aleatorio: Kruskal, que termina en un árbol generador en k = |V| - 1 = 215);
  - P(τ ≤ t*) con t* = 50 y R réplicas por punto (Wilson 95 %);
  - línea discontinua: P(τ < ∞) exacta por cargas (N_UMBRAL muestras de k*).

Uso:  python presentacion/no_monotonia_3d.py [--piloto] [--recalcular]
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

nb = ec.nb
AQUI = Path(__file__).resolve().parent
BOX = (5, 5, 5)
T_STAR = 50.0
R = 1000
N_UMBRAL = 5000
MAX_EVENTOS = 2_000_000
SEMILLA = 85
CONDICIONES = ["1 partícula", "lazo ∂f"]
REGLAS = ["uniforme", "árbol aleatorio"]
COLOR = {"uniforme": "#B5652B", "árbol aleatorio": "#1E8C66"}
MARCA = {"uniforme": "o", "árbol aleatorio": "s"}
N_TROZOS = 20


def inicial(ic):
    return nb.init_single_particle if ic == "1 partícula" else nb.init_loop


def orden(regla):
    return nb.order_uniform if regla == "uniforme" else nb.order_random_tree


def grilla_k(regla, cx):
    if regla == "uniforme":
        return np.linspace(0, len(cx.edges), 11).astype(int)
    return np.linspace(0, len(cx.vertices) - 1, 10).astype(int)


# ----------------------------------------------------------------------------- trabajadores
def trozo_simulacion(ic, regla, k, seeds):
    """Réplicas: θ_0, T = primeras k aristas de la regla, simulación hasta t* (censura).
    Devuelve por réplica (¿se extinguió antes de t*?, n.º de eventos)."""
    cx = nb.build_cubical_complex(BOX)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta0 = inicial(ic)(cx, rng)
        T = np.zeros(len(cx.edges), bool)
        T[orden(regla)(cx, rng)[:k]] = True
        r = nb.run_branching_process(theta0, MAX_EVENTOS, cx, T, random_seed=rng, t_max=T_STAR)
        out.append((bool(r.extinct and r.log.times[-1] <= T_STAR), len(r.log.times) - 1))
    return out


def trozo_umbral(ic, regla, seeds):
    """k* (menor k con todas las cargas nulas) para cada réplica, como `threshold_chunk` del notebook."""
    cx = nb.build_cubical_complex(BOX)
    tail, head = (x.tolist() for x in nb.edge_endpoints(cx.d1))
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta0 = inicial(ic)(cx, rng)
        out.append(nb.extinction_threshold(orden(regla)(cx, rng), cx.d1 @ theta0, tail, head, len(cx.vertices)))
    return out


def trozos(seeds, n=N_TROZOS):
    return [list(c) for c in np.array_split(np.array(seeds, dtype=object), n)]


# ----------------------------------------------------------------------------- cálculo
def calcular(r=R, n_umbral=N_UMBRAL):
    cx = nb.build_cubical_complex(BOX)
    res = {"R": r, "finito": {}, "eventos": {}, "umbral": {}}
    for i, ic in enumerate(CONDICIONES):
        for j, regla in enumerate(REGLAS):
            t0 = time.time()
            seeds = nb.replica_seeds([nb.SEED, SEMILLA, 0, i, j], n_umbral)
            b = Parallel(n_jobs=-1)(delayed(trozo_umbral)(ic, regla, c) for c in trozos(seeds))
            res["umbral"][(ic, regla)] = np.concatenate(b)
            exitos, eventos = [], []
            for k in grilla_k(regla, cx):
                seeds = nb.replica_seeds([nb.SEED, SEMILLA, 1, i, j, int(k)], r)
                b = Parallel(n_jobs=-1)(delayed(trozo_simulacion)(ic, regla, int(k), c) for c in trozos(seeds))
                a = np.array([fila for x in b for fila in x])
                exitos.append(int(a[:, 0].sum()))
                eventos.append(int(a[:, 1].max()))
            res["finito"][(ic, regla)] = np.array(exitos)
            res["eventos"][(ic, regla)] = np.array(eventos)
            p = np.array(exitos) / r
            print(f"{ic:12s} {regla:16s} P(τ≤50): {' '.join(f'{x:.3f}' for x in p)}   "
                  f"máx eventos {max(eventos)}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def resultados():
    cache = AQUI / "no_monotonia_3d_resultados.pkl"
    if cache.exists() and "--recalcular" not in sys.argv:
        return pickle.loads(cache.read_bytes())
    res = calcular()
    cache.write_bytes(pickle.dumps(res))
    return res


# ----------------------------------------------------------------------------- figura
def figura(res):
    cx = nb.build_cubical_complex(BOX)
    V, E, r = len(cx.vertices), len(cx.edges), res["R"]
    k_todo = np.arange(E + 1)
    with plt.rc_context(ec.ESTILO):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        for ax, ic in zip(axes, CONDICIONES):
            for regla in REGLAS:
                ks = res["umbral"][(ic, regla)]
                ax.plot(k_todo, (ks[:, None] <= k_todo[None, :]).mean(axis=0), ls="--", lw=1.1,
                        color=COLOR[regla], alpha=0.6)
                kg = grilla_k(regla, cx)
                n = res["finito"][(ic, regla)]
                p = n / r
                lo, hi = nb.wilson_interval(n, r)
                ax.errorbar(kg, p, yerr=[p - lo, hi - p], color=COLOR[regla], marker=MARCA[regla], ms=6,
                            lw=1.6, capsize=2.5, elinewidth=1.1, label=regla)
            ax.axvline(V - 1, color=ec.INK, ls=":", lw=1, alpha=0.7)
            ax.set_xlabel("k = aristas absorbentes")
        axes[0].set_title("1 partícula: más rápida")
        axes[1].set_title("lazo ∂f: no se acelera hasta k ≈ |V|")
        axes[0].set_ylabel(rf"$\hat{{\mathbb{{P}}}}(\tau\leq {T_STAR:.0f})$")
        proxy = Line2D([], [], color=ec.INK, ls="--", lw=1.1, alpha=0.6, label="extinción eventual")
        axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + [proxy], loc="upper left")
        plt.tight_layout()
        fig.savefig(ec.FIGURAS / "aristas_tiempo_finito_3d.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def piloto():
    """Mide el costo con pocas réplicas en los extremos de la grilla."""
    for ic in CONDICIONES:
        for regla in REGLAS:
            for k in [0, 107, 215, 540] if regla == "uniforme" else [0, 107, 215]:
                t0 = time.time()
                seeds = nb.replica_seeds([999, k], 40)
                b = Parallel(n_jobs=-1)(delayed(trozo_simulacion)(ic, regla, k, c) for c in trozos(seeds, 8))
                a = np.array([f for x in b for f in x])
                print(f"{ic:12s} {regla:16s} k={k:3d}  P(τ≤50)={a[:, 0].mean():.2f}  "
                      f"eventos: mediana {np.median(a[:, 1]):.0f}, máx {a[:, 1].max()}  ({time.time() - t0:.1f}s)",
                      flush=True)


if __name__ == "__main__":
    if "--piloto" in sys.argv:
        piloto()
    else:
        figura(resultados())
