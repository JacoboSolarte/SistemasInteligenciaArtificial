"""
Ejecuta Hill Climbing (básico y steepest ascent) y la estrategia evolutiva (μ+λ) con el MISMO evaluador
(mismo subconjunto y mismas épocas), compara los métodos y guarda la mejor configuración en
resultados/mejor_config.json para el entrenamiento final:

    python optimizacion/ejecutar_optimizacion.py
    python entrenamiento/entrenar_cnn.py --usar-mejor-config
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizacion import estrategia_evolutiva, hill_climbing  # noqa: E402
from optimizacion.espacio_busqueda import Evaluador  # noqa: E402
from utilidades.comun import guardar_json, ruta  # noqa: E402


def graficar(hc, es_reg, registro, archivo):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for nombre, (_, _, h) in hc.items():
        h = pd.DataFrame(h)
        ax[0].plot(h.iteracion, h.f1_candidato, "o", alpha=0.45, label=f"{nombre}: candidato")
        ax[0].plot(h.iteracion, h.f1_actual, "-", lw=2, label=f"{nombre}: mejor actual")
    ax[0].set(title="Hill Climbing: fitness por iteración", xlabel="iteración", ylabel="F1 validación")
    g = pd.DataFrame(es_reg)
    ax[1].plot(g.generacion, g.mejor, "-o", label="mejor")
    ax[1].plot(g.generacion, g.promedio, "-s", label="promedio padres")
    ax[1].fill_between(g.generacion, g.peor, g.mejor, alpha=0.2)
    if "promedio_hijos" in g:
        ax[1].plot(g.generacion, g.promedio_hijos, "--^", label="promedio hijos")
    ax[1].set(title="Estrategia evolutiva (μ+λ) por generación", xlabel="generación", ylabel="F1 validación")
    r = pd.DataFrame(registro)
    for m in r.metodo.unique():
        sub = r[r.metodo == m].reset_index(drop=True)
        ax[2].plot(np.arange(1, len(sub) + 1), sub.f1.cummax(), lw=2, label=m)
    ax[2].set(title="Comparación: mejor F1 vs. nº de evaluaciones", xlabel="evaluaciones", ylabel="mejor F1 hasta el momento")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)


def main():
    ev = Evaluador()
    hc = hill_climbing.main(ev)
    es = estrategia_evolutiva.main(ev)
    res = ruta(ev.cfg, "resultados")
    graficar(hc, es[2], ev.registro, res / "optimizacion_hiperparametros.png")
    pd.DataFrame(ev.registro).to_csv(res / "optimizacion_registro.csv", index=False)
    candidatos = [("Hill Climbing", hc["HC"][0], hc["HC"][1]), ("HC steepest ascent", hc["HC-steepest"][0], hc["HC-steepest"][1]),
                  ("Estrategia evolutiva (μ+λ)", es[0], es[1])]
    tabla = pd.DataFrame([{"metodo": m, "f1_val": f, **c} for m, c, f in candidatos])
    tabla["evaluaciones_unicas"] = [sum(1 for r in ev.registro if r["metodo"] == k) for k in ("HC", "HC-steepest", "ES")]
    tabla.to_csv(res / "optimizacion_comparacion.csv", index=False)
    metodo, mejor, f = max(candidatos, key=lambda t: t[2])
    guardar_json({"metodo": metodo, "config": mejor, "f1_val": f}, res / "mejor_config.json")
    print(tabla.to_string(index=False))
    print(f"\nMejor configuración ({metodo}): {mejor}  F1 val = {f:.4f}")


if __name__ == "__main__":
    main()
