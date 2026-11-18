"""Con k aristas absorbentes fijas, ¿qué configuración de T maximiza la extinción?

Estudio complementario al §8 del notebook (mismo motor; las definiciones se leen del notebook).
Caja 9×9 (|V| = 100, |E| = 180). Una partícula +1 parte en una arista UNIFORME FUERA de T:
el obstáculo se fija antes de saber dónde está la partícula.

Extinción eventual (exacta, por cargas): la partícula en (u, w) se extingue sii u y w están en la
misma componente de (V, T). Con n = k+1 vértices conectados caben a lo más ⌊2n − 2√n⌋ aristas de Z²
(Harary–Harborth, 1976), y repartir T en varias componentes nunca ayuda; por lo tanto

    P(τ < ∞) ≤ (⌊2n − 2√n⌋ − k) / (|E| − k),

con igualdad para cualquier árbol de un bloque compacto (casi cuadrado) de k+1 vértices.
Extinción antes de t* = 50: simulación con R réplicas por punto e intervalos de Wilson al 95 %.

Uso:  python presentacion/estudio_configuracion.py      (unos minutos)
Salida: presentacion/figuras/config_barras.png, presentacion/figuras/config_curvas.png
"""
import ast
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from joblib import Parallel, delayed

AQUI = Path(__file__).resolve().parent
NOTEBOOK = AQUI.parent / "branching_process_complexes.ipynb"
FIGURAS = AQUI / "figuras"

BOX = (9, 9)
T_STAR = 50.0
R = 2000
K_SIM = [15, 35, 63, 80]       # k con simulación de tiempo finito
K_BARRAS = 63                  # k de la figura de barras
N_T_SAMPLES = 2000             # muestras de T para la curva exacta de las familias aleatorias

# Paleta de la presentación: café y verde
INK = "#3E271C"
COLORES = {"bloque central": "#1E8C66", "bloque en esquina": "#7CC47C", "peine extendido": "#B5652B",
           "árbol aleatorio": "#7A543A", "uniforme": "#D9A66B"}
MARCAS = {"bloque central": "o", "bloque en esquina": "^", "peine extendido": "s",
          "árbol aleatorio": "D", "uniforme": "v"}
VERDE_T = "#8FD18F"
# configuraciones que se muestran en las figuras (las otras dos se calculan igual y se imprimen)
EN_FIGURAS = ["bloque central", "árbol aleatorio", "uniforme"]
ESTILO = {
    "font.size": 13, "axes.titlesize": 14, "axes.titlelocation": "left", "axes.labelsize": 13.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E9DFD0", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
}


# ----------------------------------------------------------------------------- motor del notebook
def cargar_motor():
    """Módulo con las definiciones (imports, funciones, clases y constantes) de las celdas del notebook."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    trozos = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code" or i > 43:
            continue
        src = celda.source.replace("from IPython.display import Video, Image, display", "")
        lineas = src.splitlines()
        for nodo in ast.parse(src).body:
            if isinstance(nodo, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                ini = (nodo.decorator_list[0].lineno if getattr(nodo, "decorator_list", None) else nodo.lineno) - 1
                trozos.append("\n".join(lineas[ini:nodo.end_lineno]))
            elif (isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name)
                  and nodo.targets[0].id in {"SEED"}):
                trozos.append(ast.get_source_segment(src, nodo))
    modulo = types.ModuleType("motor")
    modulo.__file__ = str(AQUI / "motor_desde_notebook.py")
    exec(compile("\n\n".join(trozos), modulo.__file__, "exec"), modulo.__dict__)
    sys.modules["motor"] = modulo
    return modulo


class _Motor:
    """Acceso perezoso al motor: cada proceso de joblib lo carga del notebook la primera vez que lo usa."""
    def __getattr__(self, nombre):
        if nombre.startswith("__"):
            raise AttributeError(nombre)
        return getattr(sys.modules.get("motor") or cargar_motor(), nombre)


nb = _Motor()


# ----------------------------------------------------------------------------- configuraciones
def coords(cx):
    return np.rint(cx.vertices / 2).astype(int)


def extremos(cx):
    tail, head = nb.edge_endpoints(cx.d1)
    return np.asarray(tail), np.asarray(head)


def bloque_compacto(cx, n, centro):
    """n vértices que forman un bloque casi cuadrado: se agrega siempre el vértice vecino con más
    vecinos ya elegidos (desempate: más cerca de `centro`). Crece como una espiral."""
    xy = coords(cx)
    tail, head = extremos(cx)
    vecinos = [[] for _ in range(len(xy))]
    for a, b in zip(tail, head):
        vecinos[a].append(b); vecinos[b].append(a)
    dist = np.linalg.norm(xy - np.asarray(centro, float), axis=1)
    elegido = np.zeros(len(xy), bool)
    elegido[np.argmin(dist + 1e-6 * np.arange(len(xy)))] = True
    while elegido.sum() < n:
        cuenta = np.zeros(len(xy))
        for v in np.flatnonzero(elegido):
            for w in vecinos[v]:
                cuenta[w] += 1
        cuenta[elegido] = -1
        cand = np.flatnonzero(cuenta == cuenta.max())
        elegido[cand[np.argmin(dist[cand] + 1e-6 * cand)]] = True
    return elegido


def kruskal(aristas, tail, head, n_vertices, rng):
    """Árbol (o bosque) aleatorio: recorre `aristas` en orden uniforme y acepta las que no cierran ciclo."""
    padre = list(range(n_vertices))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    out = []
    for e in rng.permutation(aristas).tolist():
        a, b = raiz(tail[e]), raiz(head[e])
        if a != b:
            padre[a] = b
            out.append(e)
    return np.array(out, dtype=int)


def aristas_peine(cx, k):
    """Peine extendido: lomo horizontal en y = 4 y dientes verticales completos en x = 1, 4, 7, 0, 9, ...
    (un árbol) hasta completar k aristas."""
    xy = coords(cx)
    idx = {tuple(p): i for i, p in enumerate(xy)}
    tail, head = extremos(cx)
    arista = {frozenset((int(a), int(b))): e for e, (a, b) in enumerate(zip(tail, head))}
    n = BOX[0]
    pares = [((x, 4), (x + 1, 4)) for x in range(n)]
    for x in [1, 4, 7, 0, 9, 2, 5, 8, 3, 6]:
        pares += [((x, y), (x, y + 1)) for y in range(n)]
    return np.array([arista[frozenset((idx[p], idx[q]))] for p, q in pares[:k]], dtype=int)


def configuraciones(k):
    """Cada familia es una función (cx, rng) -> máscara de T con exactamente k aristas."""
    def mascara(cx, ids):
        m = np.zeros(len(cx.edges), bool)
        m[ids] = True
        assert m.sum() == k
        return m

    def bloque(centro):
        def f(cx, rng):
            tail, head = extremos(cx)
            dentro = bloque_compacto(cx, k + 1, centro)
            ids = np.flatnonzero(dentro[tail] & dentro[head])
            return mascara(cx, kruskal(ids, tail, head, len(cx.vertices), rng))
        return f

    return {
        "bloque central": bloque((BOX[0] / 2, BOX[1] / 2)),
        "bloque en esquina": bloque((0, 0)),
        "peine extendido": lambda cx, rng: mascara(cx, aristas_peine(cx, k)),
        "árbol aleatorio": lambda cx, rng: mascara(cx, nb.order_random_tree(cx, rng)[:k]),
        "uniforme": lambda cx, rng: mascara(cx, rng.choice(len(cx.edges), size=k, replace=False)),
    }


def particula_fuera_de_T(cx, T, rng):
    theta = np.zeros(len(cx.edges), np.int64)
    theta[rng.choice(np.flatnonzero(~T))] = 1
    return theta


def prob_eventual(cx, T):
    """P(τ < ∞) exacta para una partícula uniforme en E \\ T."""
    _, lab = nb.components_of(T, cx.d1)
    tail, head = extremos(cx)
    return np.mean((lab[tail] == lab[head])[~T])


def cota(k, n_edges):
    n = k + 1
    return (np.floor(2 * n - 2 * np.sqrt(n) + 1e-9) - k) / (n_edges - k)


# ----------------------------------------------------------------------------- simulación
def bloque_replicas(k, nombre, semillas):
    cx = nb.build_cubical_complex(BOX)
    fam = configuraciones(k)[nombre]
    out = []
    for s in semillas:
        rng = np.random.default_rng(s)
        T = fam(cx, rng)
        theta0 = particula_fuera_de_T(cx, T, rng)
        r = nb.run_branching_process(theta0, 2_000_000, cx, T, random_seed=rng, t_max=T_STAR)
        out.append(r.extinct and r.log.times[-1] <= T_STAR)     # el último evento puede caer después de t*
    return out


def tiempo_finito(k, nombre, i):
    semillas = nb.replica_seeds([nb.SEED, 77, k, i], R)
    trozos = [list(c) for c in np.array_split(np.array(semillas, dtype=object), 40)]
    res = Parallel(n_jobs=-1)(delayed(bloque_replicas)(k, nombre, c) for c in trozos)
    return int(sum(sum(r) for r in res))


def bloque_eventual(k, nombre, semillas):
    cx = nb.build_cubical_complex(BOX)
    fam = configuraciones(k)[nombre]
    return [prob_eventual(cx, fam(cx, np.random.default_rng(s))) for s in semillas]


def eventual_medio(k, nombre, i, n):
    semillas = nb.replica_seeds([nb.SEED, 78, k, i], n)
    trozos = [list(c) for c in np.array_split(np.array(semillas, dtype=object), 8)]
    return float(np.mean(np.concatenate(Parallel(n_jobs=-1)(delayed(bloque_eventual)(k, nombre, c) for c in trozos))))


# ----------------------------------------------------------------------------- figuras
def dibujar_T(ax, cx, T):
    seg = nb.edge_segments(cx.edges)
    ax.add_collection(LineCollection(seg, colors="#E4D8C6", linewidths=0.8, zorder=1))
    ax.add_collection(LineCollection(seg[T], colors="#4FAF4F", linewidths=3.0, capstyle="round", zorder=2))
    ax.set_xlim(-0.4, BOX[0] + 0.4); ax.set_ylim(-0.4, BOX[1] + 0.4)
    ax.set_aspect("equal"); ax.axis("off")


def figura_barras(cx, exacta, finitos):
    nombres = EN_FIGURAS
    with plt.rc_context(ESTILO):
        fig = plt.figure(figsize=(7.6, 5.0))
        gs = fig.add_gridspec(2, len(nombres), height_ratios=[1, 1.55], hspace=0.12, wspace=0.08)
        rng = np.random.default_rng(3)
        for j, n in enumerate(nombres):
            ax = fig.add_subplot(gs[0, j])
            dibujar_T(ax, cx, configuraciones(K_BARRAS)[n](cx, rng))
            ax.set_title(n, loc="center", fontsize=12.5, color=INK)
        ax = fig.add_subplot(gs[1, :])
        x = np.arange(len(nombres))
        ax.bar(x, [exacta[n] for n in nombres], width=0.55, color="#E2B683", label="eventual (exacta)")
        for j, n in enumerate(nombres):
            p = finitos[n] / R
            lo, hi = nb.wilson_interval(finitos[n], R)
            ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=8, color="#1E8C66", mec="white", mew=1.2,
                        capsize=3, label=r"antes de $t=50$ (simulación)" if j == 0 else None)
        ax.axhline(cota(K_BARRAS, len(cx.edges)), color=INK, ls=":", lw=1.2)
        ax.annotate("máximo posible", xy=(len(nombres) - 0.62, cota(K_BARRAS, len(cx.edges))), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=11)
        ax.set_xticks(x, [])
        ax.set_xlim(-0.5, len(nombres) - 0.5)
        ax.set_ylim(0, 0.5)
        ax.set_ylabel("P(extinción)")
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
        fig.savefig(FIGURAS / "config_barras.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def figura_curvas(n_edges, ks, curvas, finitos_k):
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(9.6, 4.4))
        for n in EN_FIGURAS:
            ax.plot(ks, curvas[n], color=COLORES[n], lw=2.0, alpha=0.9)
        for n in EN_FIGURAS:
            p = np.array([finitos_k[(k, n)] for k in K_SIM]) / R
            lo, hi = nb.wilson_interval(np.array([finitos_k[(k, n)] for k in K_SIM]), R)
            ax.errorbar(K_SIM, p, yerr=[p - lo, hi - p], fmt=MARCAS[n], ms=7, color=COLORES[n], mec="white",
                        mew=1.0, capsize=2.5, label=n, ls="none")
        ax.axvline(99, color=INK, ls=":", lw=1, alpha=0.6)
        ax.annotate("árbol completo (k = 99)", xy=(99, 0.03), xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", fontsize=10.5, alpha=0.85)
        ax.set_xlim(0, 103); ax.set_ylim(0, 1.03)
        ax.set_xlabel("k = aristas absorbentes")
        ax.set_ylabel("P(extinción)")
        leyenda = ax.legend(loc="upper left", ncol=1)
        ax.add_artist(leyenda)
        tipos = [Line2D([], [], color="#8C7B70", lw=2.0, label="eventual (exacta)"),
                 Line2D([], [], color="#8C7B70", marker="o", ls="none", ms=7, label="antes de t = 50 (simulación)")]
        ax.legend(handles=tipos, loc="upper left", bbox_to_anchor=(0.0, 0.5))
        fig.savefig(FIGURAS / "config_curvas.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    import pickle
    cache = AQUI / "estudio_configuracion_resultados.pkl"
    cx = nb.build_cubical_complex(BOX)
    n_edges = len(cx.edges)
    # el bloque compacto alcanza la cota de Harary–Harborth para todo n
    tail, head = extremos(cx)
    for n in range(2, len(cx.vertices) + 1):
        dentro = bloque_compacto(cx, n, (BOX[0] / 2, BOX[1] / 2))
        assert np.sum(dentro[tail] & dentro[head]) == np.floor(2 * n - 2 * np.sqrt(n) + 1e-9), n

    nombres = list(COLORES)
    if cache.exists() and "--recalcular" not in sys.argv:
        finitos_k, curvas = pickle.loads(cache.read_bytes())
        ks = np.arange(0, 100)
        return terminar(cx, n_edges, nombres, finitos_k, curvas, ks)
    finitos_k = {}
    for k in K_SIM:
        for i, n in enumerate(nombres):
            finitos_k[(k, n)] = tiempo_finito(k, n, i)
            print(f"k = {k:3d}  {n:18s}  P(τ ≤ {T_STAR:.0f}) = {finitos_k[(k, n)] / R:.3f}", flush=True)

    ks = np.arange(0, 100)
    curvas = {}
    for i, n in enumerate(nombres):
        aleatoria = n in ("árbol aleatorio", "uniforme")
        curvas[n] = np.array([eventual_medio(int(k), n, i, N_T_SAMPLES if aleatoria else 1) for k in ks])
    cache.write_bytes(pickle.dumps((finitos_k, curvas)))
    terminar(cx, n_edges, nombres, finitos_k, curvas, ks)


def terminar(cx, n_edges, nombres, finitos_k, curvas, ks):
    exacta = {n: curvas[n][K_BARRAS] for n in nombres}
    for n in nombres:
        print(f"k = {K_BARRAS}  {n:18s}  P(τ < ∞) = {exacta[n]:.3f}   P(τ ≤ {T_STAR:.0f}) = {finitos_k[(K_BARRAS, n)] / R:.3f}")
    print(f"cota en k = {K_BARRAS}: {cota(K_BARRAS, n_edges):.3f}")

    figura_barras(cx, exacta, {n: finitos_k[(K_BARRAS, n)] for n in nombres})
    figura_curvas(n_edges, ks, curvas, finitos_k)


if __name__ == "__main__":
    main()
