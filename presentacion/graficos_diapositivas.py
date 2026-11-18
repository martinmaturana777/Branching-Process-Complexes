"""Versiones para diapositivas (paleta verde y café) de dos gráficos del §8 del notebook,
con la paleta de la presentación.

  figuras/verificacion_cargas.png   §8.1: T = k aristas al azar, ¿cargas nulas?, ¿se extinguió?
                                    (semilla nueva y 1000 réplicas por k, independiente del notebook)
  figuras/aristas_dinamico.png      §8.5: T creciendo durante la simulación (mismas semillas que el notebook)

Uso:  python presentacion/graficos_diapositivas.py      (unos minutos)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estudio_configuracion as ec          # motor del notebook, estilo y paleta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator, PercentFormatter

nb = ec.nb
FIGURAS = ec.FIGURAS
VERDE, CAFE, ROJO, INK = "#1E8C66", "#B5652B", "#D62728", ec.INK


# ----------------------------------------------------------------------------- §8.1 cargas
# Verificación independiente de la del notebook: semilla nueva y más réplicas.
K_CHECK = [0, 20, 40, 60, 80, 100, 120, 140]
R_CHECK = 1000                         # réplicas por k (8000 en total)
TOPE_NULA, TOPE_NO_NULA = 10_000_000, 100_000
SEMILLA_CARGAS = 81                    # flujo de semillas [SEED, 81, k] (el notebook usa SEED + k)
N_TROZOS = 10                          # trozos por k para repartir entre procesos


def verificar(k, seeds):
    """Como `check_charge_prediction` del notebook: una partícula en la arista central, T = k aristas
    uniformes; se simula hasta TOPE_NULA eventos si las cargas son nulas y TOPE_NO_NULA si no.
    Devuelve por réplica (¿cargas nulas?, ¿se extinguió?, n.º de eventos simulados)."""
    cx = nb.build_cubical_complex((9, 9))
    filas = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta0 = nb.init_single_particle(cx, rng)
        T = nb.kill_random_k(cx, theta0, rng, k)
        nula = nb.extinction_possible(theta0, T, cx.d1)
        r = nb.run_branching_process(theta0, TOPE_NULA if nula else TOPE_NO_NULA, cx, T, random_seed=rng)
        filas.append((bool(nula), bool(r.extinct), len(r.log.times) - 1))
    return filas


def contar_cargas():
    """Resultados por k (en caché): arreglo (R_CHECK, 3) con columnas nula, extinta, eventos."""
    cache = Path(__file__).resolve().parent / "verificacion_cargas_resultados.pkl"
    if cache.exists() and "--recalcular" not in sys.argv:
        return pickle.loads(cache.read_bytes())
    tareas = []
    for k in K_CHECK:
        seeds = nb.replica_seeds([nb.SEED, SEMILLA_CARGAS, k], R_CHECK)
        tareas += [(k, list(c)) for c in np.array_split(np.array(seeds, dtype=object), N_TROZOS)]
    trozos = Parallel(n_jobs=-1, verbose=5)(delayed(verificar)(k, c) for k, c in tareas)
    res = {k: [] for k in K_CHECK}
    for (k, _), filas in zip(tareas, trozos):
        res[k] += filas
    res = {k: np.array(v, dtype=np.int64) for k, v in res.items()}
    cache.write_bytes(pickle.dumps(res))
    return res


def tabla_2x2(res):
    """Filas: carga nula, carga no nula · columnas: se extinguió, sigue viva."""
    a = np.concatenate(list(res.values()))
    nula, ext = a[:, 0].astype(bool), a[:, 1].astype(bool)
    return np.array([[np.sum(nula & ext), np.sum(nula & ~ext)], [np.sum(~nula & ext), np.sum(~nula & ~ext)]])


def figura_cargas():
    """Matriz 2×2 sumada sobre todos los k: predicción por cargas versus resultado de la simulación.
    Error tipo I: carga nula y no se extinguió. Error tipo II: carga no nula y se extinguió."""
    res = contar_cargas()
    for k, a in res.items():
        nula, ext = a[:, 0].astype(bool), a[:, 1].astype(bool)
        print(f"k = {k:3d}  cargas nulas: {nula.sum():4d} (se extinguieron {np.sum(nula & ext):4d})   "
              f"cargas no nulas: {(~nula).sum():4d} (siguen vivas {np.sum(~nula & ~ext):4d})")
    a = np.concatenate(list(res.values()))
    nula_ext = a[a[:, 0].astype(bool) & a[:, 1].astype(bool)]
    print(f"extinción más lenta con carga nula: {nula_ext[:, 2].max()} eventos (tope {TOPE_NULA})")
    tabla = tabla_2x2(res)
    total = int(tabla.sum())
    # cota superior de Wilson al 95 % para cada tasa de error, dentro de su fila
    cotas = [[None, nb.wilson_interval(tabla[0, 1], tabla[0].sum())[1]],
             [nb.wilson_interval(tabla[1, 0], tabla[1].sum())[1], None]]
    fondos = [[VERDE, "#F6E3E1"], ["#F6E3E1", "#E2B683"]]
    textos = [["white", ROJO], [ROJO, INK]]
    nombres = [[None, "error tipo I"], ["error tipo II", None]]
    with plt.rc_context(ec.ESTILO):
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for i in range(2):
            for j in range(2):
                y = 1 - i
                ax.add_patch(Rectangle((j, y), 0.97, 0.97, color=fondos[i][j], lw=0))
                n = int(tabla[i, j])
                cy = y + 0.485
                if nombres[i][j]:
                    ax.text(j + 0.485, cy + 0.2, nombres[i][j], ha="center", va="center",
                            fontsize=11.5, color=textos[i][j])
                ax.text(j + 0.485, cy, f"{100 * n / total:.1f} %".replace(".", ","), ha="center", va="center",
                        fontsize=22, fontweight="bold", color=textos[i][j])
                ax.text(j + 0.485, cy - 0.2, f"{n} réplicas", ha="center", va="center",
                        fontsize=11.5, color=textos[i][j])
                if cotas[i][j] is not None:
                    ax.text(j + 0.485, cy - 0.34, f"IC 95 %: < {100 * cotas[i][j]:.2f} %".replace(".", ","),
                            ha="center", va="center", fontsize=10.5, color=textos[i][j])
        ax.set_xlim(0, 1.97); ax.set_ylim(0, 1.97)
        ax.set_xticks([0.485, 1.485], ["se extinguió", "sigue viva"])
        ax.set_yticks([1.485, 0.485], ["carga\nnula", "carga\nno nula"])
        ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
        ax.set_xlabel("simulación", labelpad=10)
        ax.set_ylabel("predicción por cargas", labelpad=10)
        ax.tick_params(length=0)
        ax.grid(False)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_aspect("equal")
        fig.savefig(FIGURAS / "verificacion_cargas.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ----------------------------------------------------------------------------- §8.5 T creciente
M_VALUES, R_DYN, T_STAR = [1, 3, 10, 30, 100], 600, 50.0
IC = {"1 partícula": VERDE, "lazo ∂f": CAFE}


def bloque_creciente(ic, m, seeds):
    init = nb.init_single_particle if ic == "1 partícula" else nb.init_loop
    return nb.growing_chunk((9, 9), init, nb.order_random_tree, m, seeds, 3_000_000)


def replicas_crecientes(ic, m, master):
    seeds = nb.replica_seeds(master, R_DYN)
    trozos = [list(c) for c in np.array_split(np.array(seeds, dtype=object), 40)]
    bloques = Parallel(n_jobs=-1)(delayed(bloque_creciente)(ic, m, c) for c in trozos)
    return np.array([fila for b in bloques for fila in b], dtype=float)


def figura_dinamico():
    dyn = {ic: [replicas_crecientes(ic, m, nb.SEED + 5000 + 100 * i + m) for m in M_VALUES]
           for i, ic in enumerate(IC)}
    for ic, res in dyn.items():
        print(ic, " ".join(f"E[τ]={np.mean(a[:, 0]):.1f}" for a in res))
    with plt.rc_context(ec.ESTILO):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        for ic, res in dyn.items():
            axes[0].plot(M_VALUES, [np.mean(a[:, 0]) for a in res], marker="o", ms=7, color=IC[ic], label=ic)
            k = np.array([np.sum(a[:, 2].astype(bool) & (a[:, 0] <= T_STAR)) for a in res])
            p = k / R_DYN
            lo, hi = nb.wilson_interval(k, R_DYN)
            axes[1].errorbar(M_VALUES, p, yerr=[p - lo, hi - p], color=IC[ic], marker="o", ms=7, lw=1.6, capsize=3)
        for ax in axes:
            ax.set_xscale("log"); ax.set_xticks(M_VALUES, [str(m) for m in M_VALUES])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("m = eventos entre aristas agregadas")
        axes[0].set_yscale("log"); axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:g}")
        axes[0].set_ylabel(r"$\mathbb{E}[\tau]$"); axes[0].set_title("Tiempo medio de extinción")
        axes[1].set_ylabel(r"$\hat{\mathbb{P}}(\tau\leq 50)$"); axes[1].set_title("Extinción antes de t = 50")
        axes[0].legend(loc="center left")
        plt.tight_layout()
        fig.savefig(FIGURAS / "aristas_dinamico.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    figura_cargas()
    if "--solo-cargas" not in sys.argv:
        figura_dinamico()
