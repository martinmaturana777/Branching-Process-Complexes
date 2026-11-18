"""Tiempo de extinción con un árbol generador absorbente, en d = 2 y d = 3, según la condición
inicial y el tamaño de la caja.

Condiciones iniciales:
  - 1 partícula +1 en la arista central;
  - N partículas +1 en aristas uniformes de toda la caja (pueden apilarse), con N ~ Uniforme{1,…,10};
  - lazo ∂f de la cara central (en 3D, la cara horizontal central).
Absorción: árbol generador aleatorio que evita la condición inicial (`kill_tree` del notebook).

Motor: el mismo algoritmo de Gillespie que `run_until_extinction` del notebook (reloj 'generator':
la arista detonante se elige con peso |θ_e|·λ_e, la cara uniforme entre las adyacentes en ∂_T, cargas
−c·∂_T f y tiempo de espera Exp(Σ|θ_e|λ_e)), compilado con numba. Los pesos son enteros, así que la
suma total es exacta. `validar()` compara la distribución de τ con el motor del notebook.

Uso:  python presentacion/tiempos_extincion.py [--validar] [--piloto] [--recalcular]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estudio_configuracion as ec          # motor del notebook, estilo y paleta

nb = ec.nb
AQUI = Path(__file__).resolve().parent
CONDICIONES = ["1 partícula", "N ~ U{1,…,10} partículas", "lazo ∂f"]


# ----------------------------------------------------------------------------- motor compilado
@njit(cache=True)
def gillespie(theta, lam, r_ptr, r_idx, r_val, c_ptr, c_idx, c_val, max_eventos, semilla):
    """Simula hasta la extinción o hasta max_eventos; el tiempo de espera usa la tasa total antes
    del evento, como el notebook. Devuelve (τ, n.º de eventos, ¿extinto?)."""
    np.random.seed(semilla)
    E = theta.shape[0]
    w = np.empty(E, np.int64)
    total = 0
    for e in range(E):
        w[e] = abs(theta[e]) * lam[e]
        total += w[e]
    t = 0.0
    n = 0
    while n < max_eventos:
        if total == 0:
            return t, n, True
        tasa = total
        u = np.random.random() * total
        acum = 0
        e = E - 1
        for i in range(E):
            acum += w[i]
            if acum > u:
                e = i
                break
        ini, fin = r_ptr[e], r_ptr[e + 1]
        j = ini + np.random.randint(fin - ini)
        f = r_idx[j]
        c = r_val[j] * (1 if theta[e] > 0 else -1)
        for k in range(c_ptr[f], c_ptr[f + 1]):
            a = c_idx[k]
            total -= w[a]
            theta[a] -= c * c_val[k]
            w[a] = abs(theta[a]) * lam[a]
            total += w[a]
        t += -np.log(1.0 - np.random.random()) / tasa
        n += 1
    return t, n, total == 0


def preparar(cx, T):
    """Arreglos CSR (filas = aristas) y CSC (columnas = caras) de ∂_T, y λ_e."""
    B = nb.co_restricted_boundary(cx.d2, T).tocsr()
    B.sort_indices()
    Bc = B.tocsc()
    Bc.sort_indices()
    lam = np.asarray(nb.edge_face_counts(cx.d2), dtype=np.int64)
    return (lam, B.indptr.astype(np.int64), B.indices.astype(np.int64), B.data.astype(np.int64),
            Bc.indptr.astype(np.int64), Bc.indices.astype(np.int64), Bc.data.astype(np.int64))


def inicial(ic, cx, rng):
    if ic == "1 partícula":
        return nb.init_single_particle(cx, rng)
    if ic == "lazo ∂f":
        return nb.init_loop(cx, rng)
    n = int(rng.integers(1, 11))
    return np.bincount(rng.choice(len(cx.edges), size=n), minlength=len(cx.edges)).astype(np.int64)


def replica(cx, ic, seed, max_eventos):
    """Una réplica: θ_0, árbol generador que lo evita, simulación hasta la extinción."""
    rng = np.random.default_rng(seed)
    theta0 = inicial(ic, cx, rng)
    T = nb.kill_tree(cx, theta0, rng)
    theta = nb.project_out_killed(theta0, T).astype(np.int64)
    return gillespie(theta, *preparar(cx, T), max_eventos, int(rng.integers(2**31 - 1)))


def trozo(box, ic, seeds, max_eventos):
    cx = nb.build_cubical_complex(box)
    return [replica(cx, ic, s, max_eventos) for s in seeds]


# ----------------------------------------------------------------------------- validación
def trozo_notebook(box, ic, seeds):
    """Mismas réplicas con el motor del notebook (run_branching_process)."""
    cx = nb.build_cubical_complex(box)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta0 = inicial(ic, cx, rng)
        T = nb.kill_tree(cx, theta0, rng)
        r = nb.run_branching_process(theta0, 2_000_000, cx, T, random_seed=rng)
        out.append((r.log.times[-1], len(r.log.times) - 1, r.extinct))
    return out


def validar(box=(5, 5), n=4000):
    from scipy.stats import ks_2samp
    for i, ic in enumerate(CONDICIONES):
        seeds = nb.replica_seeds([nb.SEED, 90, i], n)
        partes = [list(c) for c in np.array_split(np.array(seeds, dtype=object), 40)]
        a = np.array([f for x in Parallel(n_jobs=-1)(delayed(trozo)(box, ic, c, 10**8) for c in partes) for f in x])
        seeds = nb.replica_seeds([nb.SEED, 91, i], n)
        partes = [list(c) for c in np.array_split(np.array(seeds, dtype=object), 40)]
        b = np.array([f for x in Parallel(n_jobs=-1)(delayed(trozo_notebook)(box, ic, c) for c in partes) for f in x])
        ks = ks_2samp(a[:, 0], b[:, 0])
        print(f"{ic:26s} E[τ] numba {a[:, 0].mean():8.2f} ± {1.96 * a[:, 0].std() / np.sqrt(n):.2f}   "
              f"notebook {b[:, 0].mean():8.2f} ± {1.96 * b[:, 0].std() / np.sqrt(n):.2f}   "
              f"eventos {a[:, 1].mean():9.0f} vs {b[:, 1].mean():9.0f}   KS p = {ks.pvalue:.2f}", flush=True)


def piloto():
    for box, reps in [((9, 9), 40), ((15, 15), 20), ((2, 2, 2), 40), ((3, 3, 3), 10)]:
        for ic in CONDICIONES:
            t0 = time.time()
            seeds = nb.replica_seeds([999, len(box), box[0]], reps)
            partes = [list(c) for c in np.array_split(np.array(seeds, dtype=object), min(reps, 10))]
            a = np.array([f for x in Parallel(n_jobs=-1)(delayed(trozo)(box, ic, c, 2 * 10**9) for c in partes) for f in x])
            print(f"{str(box):10s} {ic:26s} E[τ]={a[:, 0].mean():12.1f}  mediana τ={np.median(a[:, 0]):12.1f}  "
                  f"eventos medios={a[:, 1].mean():14,.0f}  extintas {int(a[:, 2].sum())}/{reps}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- estudio
CAJAS_2D = [3, 5, 7, 9, 13, 17, 21, 25]
CAJAS_3D = [2, 3, 4]
R_2D, R_3D = 2000, 1000
MAX_EVENTOS = 4 * 10**9
# 4×4×4: la mayoría de las réplicas sigue viva después de 4·10^9 eventos (t ≈ 2·10^7), así que se corre
# con tope y se reporta una cota inferior (censura). 5×5×5 queda fuera de alcance.
R_CAJA = {(3, 4): 100}
TOPE_CAJA = {(3, 4): 10**9}
COLOR = {"1 partícula": "#1E8C66", "N ~ U{1,…,10} partículas": "#7A543A", "lazo ∂f": "#B5652B"}
MARCA = {"1 partícula": "o", "N ~ U{1,…,10} partículas": "D", "lazo ∂f": "s"}


def calcular(cajas_3d=CAJAS_3D):
    res = {}
    for d, cajas, r in [(2, CAJAS_2D, R_2D), (3, cajas_3d, R_3D)]:
        for L in cajas:
            for i, ic in enumerate(CONDICIONES):
                t0 = time.time()
                box = (L,) * d
                rr = R_CAJA.get((d, L), r)
                tope = TOPE_CAJA.get((d, L), MAX_EVENTOS)
                seeds = nb.replica_seeds([nb.SEED, 95, d, L, i], rr)
                partes = [list(c) for c in np.array_split(np.array(seeds, dtype=object), min(40, rr))]
                a = np.array([f for x in Parallel(n_jobs=-1)(delayed(trozo)(box, ic, c, tope) for c in partes)
                              for f in x])
                res[(d, L, ic)] = a
                print(f"d={d} L={L:2d} {ic:26s} E[τ]={a[:, 0].mean():12.1f}  mediana={np.median(a[:, 0]):12.1f}  "
                      f"eventos medios={a[:, 1].mean():14,.0f}  extintas {int(a[:, 2].sum())}/{rr}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return res


def resultados():
    cache = AQUI / "tiempos_extincion_resultados.pkl"
    if cache.exists() and "--recalcular" not in sys.argv:
        return pickle.loads(cache.read_bytes())
    res = calcular()
    cache.write_bytes(pickle.dumps(res))
    return res


def exponente(res, ic, L_min=5):
    """Pendiente de log E[τ] vs log L en 2D (cajas L ≥ L_min): E[τ] ∝ L^α."""
    cajas = sorted(L for (d, L, i) in res if d == 2 and i == ic and L >= L_min)
    y = [np.log(res[(2, L, ic)][:, 0].mean()) for L in cajas]
    return np.polyfit(np.log(cajas), y, 1)[0]


def panel(ax, res, d, con_exponente=False):
    """Media (IC 95 %) y mediana de τ vs L; cajas con censura como cota inferior con flecha."""
    cajas = sorted({L for (dd, L, _) in res if dd == d})
    etiquetas = []
    for ic in CONDICIONES:
        completas = [L for L in cajas if res[(d, L, ic)][:, 2].all()]
        a = [res[(d, L, ic)] for L in completas]
        media = np.array([x[:, 0].mean() for x in a])
        ee = np.array([x[:, 0].std(ddof=1) / np.sqrt(len(x)) for x in a])
        mediana = np.array([np.median(x[:, 0]) for x in a])
        ax.errorbar(completas, media, yerr=1.96 * ee, color=COLOR[ic], marker=MARCA[ic], ms=7, lw=1.8, capsize=3)
        ax.plot(completas, mediana, color=COLOR[ic], marker=MARCA[ic], ms=6, lw=1.0, ls=":", mfc="white")
        for L in cajas:                     # cajas con censura: E[min(τ, tope)] es cota inferior de E[τ]
            if L in completas:
                continue
            cota = res[(d, L, ic)][:, 0].mean()
            xL = L + 0.07 * (CONDICIONES.index(ic) - 1)      # separa las tres flechas
            ax.plot([completas[-1], xL], [media[-1], cota], color=COLOR[ic], lw=1.0, ls="--", alpha=0.6)
            ax.annotate("", xy=(xL, cota * 8), xytext=(xL, cota),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR[ic], lw=1.8, mutation_scale=14))
            ax.plot([xL], [cota], marker="_", ms=12, mew=2.5, color=COLOR[ic])
        etiquetas.append(ic + (rf"  (media $\propto L^{{{exponente(res, ic):.1f}}}$)".replace(".", ",")
                               if con_exponente else ""))
    ax.set_yscale("log")
    ax.set_xticks(cajas, [str(L) for L in cajas])
    ax.set_xlabel("lado de la caja L")
    return etiquetas


def leyenda_tipos(censura):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#8C7B70", lw=1.8, label="media (IC 95 %)"),
         Line2D([], [], color="#8C7B70", lw=1.0, ls=":", marker="o", mfc="white", label="mediana")]
    if censura:
        h.append(Line2D([], [], color="#8C7B70", lw=0, marker=r"$\uparrow$", ms=12, label="cota inferior (tope de $10^9$ eventos)"))
    return h


def figura(res):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with plt.rc_context(ec.ESTILO):
        for d in [2, 3]:                    # una figura por dimensión (diapositivas)
            fig, ax = plt.subplots(figsize=(8.6, 4.6))
            etiquetas = panel(ax, res, d, con_exponente=(d == 2))
            if d == 3:
                ax.set_ylim(top=1e8)
            ax.set_ylabel(r"tiempo de extinción $\tau$")
            h = [Line2D([], [], color=COLOR[ic], marker=MARCA[ic], lw=1.8, label=e) for ic, e in zip(CONDICIONES, etiquetas)]
            leg = ax.legend(handles=h, loc="upper left", fontsize=11.5)
            ax.add_artist(leg)
            if d == 2:
                ax.legend(handles=leyenda_tipos(False), loc="lower right", bbox_to_anchor=(1.0, 0.17), fontsize=11)
            else:
                ax.legend(handles=leyenda_tipos(True), loc="lower right", fontsize=11)
            fig.savefig(ec.FIGURAS / f"tiempos_extincion_{d}d.png", dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    for ic in CONDICIONES:
        print(f"2D {ic:26s} E[τ] ∝ L^{exponente(res, ic):.2f}")
    for ic in CONDICIONES:
        x = res[(3, 4, ic)]
        print(f"3D 4×4×4 {ic:26s} extintas antes del tope: {int(x[:, 2].sum())}/{len(x)}   "
              f"cota inferior E[τ] ≥ {x[:, 0].mean():.3g}   τ al tope ≈ {x[~x[:, 2].astype(bool), 0].mean():.3g}")


if __name__ == "__main__":
    if "--validar" in sys.argv:
        validar()
    if "--piloto" in sys.argv:
        piloto()
    if "--estudio" in sys.argv:
        figura(resultados())
