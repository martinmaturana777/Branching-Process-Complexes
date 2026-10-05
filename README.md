# Proceso de ramificación en complejos celulares

Simulación estocástica de un sistema de partículas con **ramificación y aniquilación** sobre el complejo cúbico $\mathbb{Z}^d$, restringido a una caja finita. Proyecto del curso **Simulación Estocástica** (Ingeniería Civil Matemática, Universidad de Chile, Primavera 2026), basado en el trabajo en preparación *Branching process on complexes* de D. Cid y A. Sepúlveda.

<p align="center">
  <img src="img/ramificacion_arbol.gif" width="49%" alt="Dinámica en Z^2 con absorción en un árbol generador">
  <img src="img/ramificacion_3d_arbol.gif" width="49%" alt="Dinámica en Z^3 con absorción en un árbol generador">
</p>

## El modelo

Una configuración $\theta \in \mathbb{Z}^{E^+}$ asigna a cada arista una cantidad entera de partículas con signo. En cada evento, una partícula sobre la arista $e$ elige una cara $f$ que la contiene: la partícula se aniquila y nacen partículas en el resto del borde de $f$,

$$\theta \longmapsto \theta - \langle e, \partial f\rangle \operatorname{sgn}(\theta_e)\, \partial f .$$

Las partículas de signo opuesto que caen en una misma arista se aniquilan, y las que llegan a un conjunto absorbente $T$ mueren. El generador cumple $\mathcal{L} I(\theta) = -\Delta_1^{\uparrow}\theta$, donde $\Delta_1^{\uparrow}$ es el Laplaciano de Hodge superior.

## Qué hace el notebook

1. **Topología del complejo.** Construye vértices, aristas y caras de la caja $[0,n_1]\times\cdots\times[0,n_d]$ en coordenadas dobladas (solo enteros), para cualquier dimensión $d$.
2. **Operadores de frontera y Laplacianos de Hodge.** Arma $\partial_1$, $\partial_2$ y $\Delta_1 = \Delta_1^{\downarrow} + \Delta_1^{\uparrow}$ como matrices dispersas (SciPy) y verifica $\partial\partial = 0$.
3. **Condiciones de absorción.** Implementa tres variantes: absorción en un árbol generador (el caso del paper), en el borde de la caja o sin absorción. También explica por qué hace falta el árbol para que el proceso se extinga.
4. **Motor estocástico.** Simulación exacta en tiempo continuo (algoritmo de Gillespie), con actualizaciones locales $O(1)$ y registro de la trayectoria independiente del tamaño de la caja.
5. **Verificación numérica de la teoría.**
   - El drift estimado por Monte Carlo coincide con la fórmula cerrada $-\Delta^{\uparrow}\theta$.
   - Cota de Lyapunov $\mathcal{L}\|\theta\|_2^2 \le C_d\|\theta\|_1 - 2\|\nabla\theta\|_2^2$ (Teorema 2.1).
   - Constante de Poincaré y cantidades conservadas (Lema 2.3).
6. **Tiempo de extinción.** Monte Carlo con réplicas independientes y reproducibles (`SeedSequence.spawn`); la cola empírica $\mathbb{P}(\tau > t)$ es compatible con decaimiento exponencial.
7. **El caso $\mathbb{Z}^3$.** Videos 3D y una observación sobre la meseta cuasi-estacionaria que aparece por la pequeña constante de Poincaré.

## Videos

| Video | Descripción |
|---|---|
| [`ramificacion_arbol.mp4`](videos/ramificacion_arbol.mp4) | $\mathbb{Z}^2$, absorción en un árbol generador aleatorio |
| [`ramificacion_borde.mp4`](videos/ramificacion_borde.mp4) | $\mathbb{Z}^2$, absorción en el borde de la caja, partiendo de un lazo |
| [`ramificacion_3d_arbol.mp4`](videos/ramificacion_3d_arbol.mp4) | $\mathbb{Z}^3$, una sola partícula inicial, absorción en un árbol |
| [`ramificacion_3d_borde.mp4`](videos/ramificacion_3d_borde.mp4) | $\mathbb{Z}^3$, absorción en la superficie de la caja |

## Cómo ejecutarlo

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook branching_process_complexes.ipynb
```

Para generar los videos se necesita además `ffmpeg`.

## Referencias

- D. Cid, A. Sepúlveda. *Branching process on complexes*, borrador (2026).
- S. Mukherjee, J. Steenbergen. *Random walks on simplicial complexes and harmonics*, Random Structures & Algorithms (2016).
- T. M. Liggett. *Interacting Particle Systems*, Springer (1985).
- D. T. Gillespie. *Exact stochastic simulation of coupled chemical reactions*, J. Phys. Chem. (1977).

**Autor:** Martín Maturana Acevedo
