"""Diapositivas paso a paso del proceso para la presentación (2 eventos, explicados).

Genera un PNG por paso en figuras/proceso/ (estado inicial + 3 pasos por evento), y cada uno va en su propia
diapositiva del Beamer. Usa las funciones del notebook (las extrae de sus celdas de definiciones).

La trayectoria es una simulación real del motor del notebook en una caja 4×4 con absorción en un árbol
generador. Como es una ilustración, se usa la primera semilla (en orden) en que los 2 eventos muestran
ramificación, aniquilación y absorción; la regla está en `es_ilustrativa`.
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from matplotlib.collections import LineCollection

AQUI = Path(__file__).resolve().parent
NOTEBOOK = AQUI.parent / "branching_process_complexes.ipynb"
SALIDA = AQUI / "figuras" / "proceso"
N_EVENTOS = 2

# paleta de la presentación
CREMA, CAFE, DULCE, DULCE_OSC, LECHE = "#FBF6EE", "#3E271C", "#C4864A", "#A0642C", "#F3E6D3"
ROJO, AZUL, VERDE, VERDE_OSC, GRIS = "#D62728", "#2166C2", "#8FD18F", "#2E8B3D", "#B9AFA6"


def cargar_funciones_del_notebook(path):
    """Ejecuta solo los imports, definiciones y constantes del notebook (sin las simulaciones largas)."""
    nb = nbformat.read(path, 4)
    g = {"__name__": "notebook"}
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        try:
            tree = ast.parse(c.source)
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                exec(compile(ast.Module(body=[n], type_ignores=[]), "<notebook>", "exec"), g)
            elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
                try:   # constantes (SEED, colores, ...); se omiten las que dependen de resultados de simulaciones
                    exec(compile(ast.Module(body=[n], type_ignores=[]), "<notebook>", "exec"), g)
                except Exception:
                    pass
    return g


nbf = cargar_funciones_del_notebook(NOTEBOOK)


def evento_detallado(theta, cx, T, rng, lam, D_csr, D_csc):
    """Un evento del motor (mismas llamadas al generador que el notebook), con todo lo necesario para dibujarlo."""
    w = nbf["particle_weights"](theta, lam, "generator")
    total = w.sum()
    e = nbf["sample_edge"](w, total, rng)
    f, incidence = nbf["sample_adjacent_face"](e, D_csr, rng)
    c = nbf["ring_charge"](theta[e], incidence)
    edges, charges = nbf["incoming_charges"](f, c, D_csc)
    antes = theta.copy()
    vivas = ~T[edges]
    choque = vivas & (antes[edges] * charges < 0)
    nbf["apply_charges"](theta, edges[vivas], charges[vivas])
    dt = nbf["holding_time"](total, rng)
    return dict(e=int(e), f=int(f), edges=edges, charges=charges, antes=antes, despues=theta.copy(), dt=dt,
                absorbidas=edges[~vivas], aniquiladas=edges[choque], nacen=edges[vivas & ~choque])


def simular(cx, semilla, n_eventos=N_EVENTOS):
    """θ_0 = una partícula central, T = árbol generador que la evita, y n_eventos eventos detallados."""
    rng = np.random.default_rng(semilla)
    theta = nbf["init_single_particle"](cx, rng)
    T = nbf["kill_tree"](cx, theta, rng)
    D_csr, D_csc = cx.d2.tocsr(), cx.d2.tocsc()
    lam = nbf["edge_face_counts"](cx.d2)
    theta0, eventos, t = theta.copy(), [], 0.0
    for _ in range(n_eventos):
        if not theta.any():
            break
        ev = evento_detallado(theta, cx, T, rng, lam, D_csr, D_csc)
        t += ev["dt"]
        ev["t"] = t
        eventos.append(ev)
    return theta0, T, eventos


def es_ilustrativa(eventos, n_eventos=N_EVENTOS):
    """Regla de selección: los eventos ocurren, en cada uno nacen partículas, y entre ambos se ven una
    absorción y una aniquilación además de la partícula detonante."""
    if len(eventos) < n_eventos or any(len(ev["nacen"]) == 0 for ev in eventos):
        return False
    absorbidas = sum(len(ev["absorbidas"]) for ev in eventos)
    extra = sum(len([a for a in ev["aniquiladas"] if a != ev["e"]]) for ev in eventos)
    return absorbidas >= 1 and extra >= 1


# ----------------------------------------------------------------------------------------- dibujo
def dibujar_reticulado(ax, cx, T):
    seg = nbf["edge_segments"](cx.edges)
    ax.add_collection(LineCollection(seg[~T], colors=GRIS, linewidths=1.2, zorder=1))
    ax.add_collection(LineCollection(seg[T], colors=VERDE, linewidths=6, zorder=1.2, capstyle="round"))
    ax.set_xlim(-0.35, cx.box_shape[0] + 0.35); ax.set_ylim(-0.35, cx.box_shape[1] + 0.35)
    ax.set_aspect("equal"); ax.axis("off")


def dibujar_particulas(ax, cx, theta):
    seg, mid = nbf["edge_segments"](cx.edges), nbf["edge_midpoints"](cx.edges)
    for e in np.flatnonzero(theta):
        col = ROJO if theta[e] > 0 else AZUL
        ax.add_collection(LineCollection(seg[[e]], colors=col, linewidths=7, zorder=3, capstyle="round"))
        ax.scatter(*mid[e], s=170, color=col, edgecolors="white", linewidths=1.5, zorder=4)
        if abs(theta[e]) > 1:
            ax.text(mid[e][0] + 0.12, mid[e][1] + 0.12, f"×{abs(theta[e])}", fontsize=13, color=col, zorder=5, weight="bold")


def resaltar_arista(ax, cx, e, etiqueta):
    seg, mid = nbf["edge_segments"](cx.edges), nbf["edge_midpoints"](cx.edges)
    ax.add_collection(LineCollection(seg[[e]], colors=DULCE, linewidths=22, alpha=0.45, zorder=2, capstyle="round"))
    ax.text(*(mid[e] + np.array([0.0, -0.32])), etiqueta, fontsize=20, color=DULCE_OSC, ha="center", va="center", zorder=6)


def resaltar_cara(ax, cx, ev):
    poly = nbf["face_polygons"](cx.faces)[ev["f"]]
    ax.fill(poly[:, 0], poly[:, 1], color=DULCE, alpha=0.30, zorder=0.5)
    centro = poly.mean(axis=0)
    ax.text(*centro, "$f$", fontsize=24, color=DULCE_OSC, ha="center", va="center", zorder=6)
    mid = nbf["edge_midpoints"](cx.edges)
    for e, q in zip(ev["edges"], ev["charges"]):   # carga que recibe cada arista del borde: -c ∂f
        p = mid[e] - 0.25 * (centro - mid[e]) / np.linalg.norm(centro - mid[e])   # por fuera de la cara
        ax.text(*p, "+1" if q > 0 else "−1", fontsize=15, color=ROJO if q > 0 else AZUL, ha="center", va="center",
                weight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc=CREMA, ec="none", alpha=0.85))


def marcar_resultado(ax, cx, ev):
    mid = nbf["edge_midpoints"](cx.edges)
    for e in ev["aniquiladas"]:
        ax.scatter(*mid[e], marker="x", s=420, linewidths=4, color=CAFE, zorder=7)
    for e in ev["absorbidas"]:
        ax.scatter(*mid[e], marker="o", s=520, facecolors="none", edgecolors=VERDE_OSC, linewidths=3.5, zorder=7)


PASOS = ["Suena el reloj de\nuna partícula (arista $e$)",
         "Elige una cara vecina $f$",
         "Se resta el borde de $f$:\nla partícula muere\ny nacen otras"]


def panel_texto(ax, titulo, paso_actual, detalle, pie):
    ax.axis("off")
    ax.text(0.0, 0.95, titulo, fontsize=26, color=CAFE, weight="bold", va="top")
    y = 0.78
    for i, txt in enumerate(PASOS):
        activo = (i == paso_actual)
        ax.text(0.0, y, f"{i + 1}.", fontsize=18, color=DULCE_OSC if activo else GRIS, weight="bold", va="top")
        ax.text(0.07, y, txt, fontsize=18, color=CAFE if activo else GRIS, weight="bold" if activo else "normal", va="top")
        y -= 0.06 + 0.055 * (txt.count("\n") + 1)
    ax.text(0.0, 0.27, detalle, fontsize=17, color=CAFE, va="top", linespacing=1.5)
    ax.text(0.0, 0.03, pie, fontsize=15, color=CAFE, alpha=0.75, va="bottom")


def leyenda_resultado(ev):
    partes = [f"nacen: {len(ev['nacen'])}"]
    if len(ev["aniquiladas"]):
        partes.append(f"✕ aniquiladas: {len(ev['aniquiladas'])} (+ con −)")
    if len(ev["absorbidas"]):
        partes.append(f"○ absorbidas por $T$: {len(ev['absorbidas'])}")
    return "\n".join(partes)


def cuadro(cx, T, theta, titulo, paso, detalle, pie, extras=()):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=CREMA)
    ax = fig.add_axes([0.02, 0.04, 0.56, 0.92], facecolor=CREMA)
    ax_t = fig.add_axes([0.60, 0.06, 0.38, 0.88], facecolor=CREMA)
    dibujar_reticulado(ax, cx, T)
    for f in extras:
        f(ax)
    dibujar_particulas(ax, cx, theta)
    panel_texto(ax_t, titulo, paso, detalle, pie)
    return fig


def generar_cuadros(cx, theta0, T, eventos):
    SALIDA.mkdir(parents=True, exist_ok=True)
    for old in SALIDA.glob("paso_*.png"):
        old.unlink()
    cuadros = []
    pie0 = "verde = conjunto absorbente $T$ (árbol generador)"
    cuadros.append(cuadro(cx, T, theta0, "Estado inicial", -1,
                          "Una partícula $+1$ (roja).\nLas partículas $-1$ se ven azules.", pie0))
    for k, ev in enumerate(eventos, start=1):
        titulo = f"Evento {k} de {len(eventos)}"
        n_antes, n_despues = np.abs(ev["antes"]).sum(), np.abs(ev["despues"]).sum()
        pie = f"partículas: {n_antes}   ·   t = {ev['t'] - ev['dt']:.2f}"
        cuadros.append(cuadro(cx, T, ev["antes"], titulo, 0, "", pie,
                              extras=[lambda ax, e=ev["e"]: resaltar_arista(ax, cx, e, "$e$")]))
        cuadros.append(cuadro(cx, T, ev["antes"], titulo, 1, "Cada arista del borde de $f$\nrecibe una carga $\\pm1$.", pie,
                              extras=[lambda ax, ev=ev: resaltar_cara(ax, cx, ev)]))
        cuadros.append(cuadro(cx, T, ev["despues"], titulo, 2, leyenda_resultado(ev),
                              f"partículas: {n_antes} → {n_despues}   ·   t = {ev['t']:.2f}",
                              extras=[lambda ax, ev=ev: marcar_resultado(ax, cx, ev)]))
    for i, fig in enumerate(cuadros):
        fig.savefig(SALIDA / f"paso_{i:02d}.png", facecolor=CREMA)
        plt.close(fig)
    return len(cuadros)


if __name__ == "__main__":
    cx = nbf["build_cubical_complex"]((4, 4))
    seeds = nbf["replica_seeds"](nbf["SEED"], 2000)
    for k, s in enumerate(seeds):
        theta0, T, eventos = simular(cx, s)
        if es_ilustrativa(eventos):
            break
    print(f"semilla n.º {k} (primera que cumple la regla de selección)")
    for i, ev in enumerate(eventos, 1):
        print(f"  evento {i}: e={ev['e']} f={ev['f']} nacen={len(ev['nacen'])} aniquiladas={len(ev['aniquiladas'])} "
              f"absorbidas={len(ev['absorbidas'])} partículas={np.abs(ev['antes']).sum()}→{np.abs(ev['despues']).sum()}")
    n = generar_cuadros(cx, theta0, T, eventos)
    print(f"{n} cuadros en {SALIDA}")
