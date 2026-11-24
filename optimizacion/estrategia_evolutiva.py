"""
Estrategia evolutiva (μ + λ) sobre los hiperparámetros de la CNN (método basado en población).

    P ← μ individuos (la config inicial + aleatorios)
    por cada generación:
        Hijos ← λ individuos: cada uno es un padre elegido al azar + mutación gaussiana N(0, σ²) en [0,1]^4
        evaluar los hijos (fitness = F1 de validación)
        P ← los μ mejores de (Padres ∪ Hijos)          ← "μ + λ": los padres compiten con sus hijos (elitismo)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizacion.espacio_busqueda import Evaluador, clave, codificar, config_inicial, decodificar  # noqa: E402
from utilidades.comun import guardar_json, ruta  # noqa: E402


def mutar(c, rng, sigma):
    return decodificar(codificar(c) + rng.normal(0, sigma, size=4))


def estrategia_mu_mas_lambda(evaluar, inicio, mu, lam, generaciones, sigma, rng):
    poblacion = [inicio] + [decodificar(rng.uniform(0, 1, 4)) for _ in range(mu - 1)]
    fitness = [evaluar(c, "ES") for c in poblacion]
    registro = [{"generacion": 0, "mejor": max(fitness), "promedio": float(np.mean(fitness)),
                 "peor": min(fitness), "poblacion": [clave(c) for c in poblacion]}]
    print(f"  Gen 0: mejor {max(fitness):.4f}  promedio {np.mean(fitness):.4f}")
    for g in range(1, generaciones + 1):
        hijos = [mutar(poblacion[rng.integers(mu)], rng, sigma) for _ in range(lam)]
        f_hijos = [evaluar(h, "ES") for h in hijos]
        todos = list(zip(poblacion + hijos, fitness + f_hijos))
        todos.sort(key=lambda t: -t[1])
        # selección: los μ mejores sin duplicados
        vistos, nueva, nf = set(), [], []
        for c, f in todos:
            if clave(c) not in vistos:
                vistos.add(clave(c))
                nueva.append(c)
                nf.append(f)
            if len(nueva) == mu:
                break
        poblacion, fitness = nueva, nf
        registro.append({"generacion": g, "mejor": max(fitness), "promedio": float(np.mean(fitness)),
                         "peor": min(fitness), "promedio_hijos": float(np.mean(f_hijos)),
                         "poblacion": [clave(c) for c in poblacion]})
        print(f"  Gen {g}: mejor {max(fitness):.4f}  promedio padres {np.mean(fitness):.4f}  promedio hijos {np.mean(f_hijos):.4f}")
    return poblacion[0], fitness[0], registro


def main(evaluador=None):
    ev = evaluador or Evaluador()
    cfg = ev.cfg
    ce = cfg["optimizacion"]["evolutiva"]
    rng = np.random.default_rng(cfg["semilla"] + 7)
    print(f"=== Estrategia evolutiva ({ce['mu']}+{ce['lambda']}) ===")
    mejor, f, reg = estrategia_mu_mas_lambda(ev, config_inicial(cfg), ce["mu"], ce["lambda"], ce["generaciones"], ce["sigma"], rng)
    guardar_json({"mejor": mejor, "f1": f, "generaciones": reg}, ruta(cfg, "resultados") / "estrategia_evolutiva.json")
    return mejor, f, reg


if __name__ == "__main__":
    main()
