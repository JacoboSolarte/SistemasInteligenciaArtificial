"""
Hill Climbing sobre los hiperparámetros de la CNN (implementado a mano, sin librerías de tuning).

Algoritmo básico:
    S ← configuración inicial ; f(S) ← F1 de validación
    repetir:
        R ← Tweak(S)                (mutación pequeña de lr, dropout, filtros y tamaño de lote)
        si f(R) > f(S):  S ← R      (solo se acepta si mejora)

Variante de máxima pendiente (steepest ascent):
    en cada paso se generan k vecinos con Tweak(S), se toma el mejor y se acepta si mejora a S.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizacion.espacio_busqueda import Evaluador, clave, codificar, config_inicial, decodificar  # noqa: E402
from utilidades.comun import guardar_json, ruta  # noqa: E402


def tweak(c: dict, rng, sigma=0.15) -> dict:
    """Pequeña mutación: ruido gaussiano en el espacio normalizado. Repite hasta obtener una config distinta."""
    u = codificar(c)
    for _ in range(20):
        nuevo = decodificar(u + rng.normal(0, sigma, size=u.shape))
        if clave(nuevo) != clave(c):
            return nuevo
    return nuevo


def hill_climbing(evaluar, inicio, iteraciones, rng):
    S, fS = inicio, evaluar(inicio, "HC")
    historia = [{"iteracion": 0, "f1_actual": fS, "f1_candidato": fS, "aceptado": True, "config": clave(S)}]
    for it in range(1, iteraciones + 1):
        R = tweak(S, rng)
        fR = evaluar(R, "HC")
        aceptado = fR > fS
        if aceptado:
            S, fS = R, fR
        historia.append({"iteracion": it, "f1_actual": fS, "f1_candidato": fR, "aceptado": aceptado, "config": clave(R)})
        print(f"  HC iter {it:2d}: candidato {fR:.4f} {'ACEPTADO' if aceptado else 'rechazado'} | mejor {fS:.4f}")
    return S, fS, historia


def steepest_ascent(evaluar, inicio, iteraciones, vecinos, rng):
    S, fS = inicio, evaluar(inicio, "HC-steepest")
    historia = [{"iteracion": 0, "f1_actual": fS, "f1_candidato": fS, "aceptado": True, "config": clave(S)}]
    for it in range(1, iteraciones + 1):
        candidatos = [tweak(S, rng) for _ in range(vecinos)]
        fits = [evaluar(c, "HC-steepest") for c in candidatos]
        j = int(np.argmax(fits))
        aceptado = fits[j] > fS
        if aceptado:
            S, fS = candidatos[j], fits[j]
        historia.append({"iteracion": it, "f1_actual": fS, "f1_candidato": fits[j], "aceptado": aceptado,
                         "config": clave(candidatos[j])})
        print(f"  Steepest iter {it}: mejor vecino {fits[j]:.4f} {'ACEPTADO' if aceptado else 'rechazado'} | mejor {fS:.4f}")
    return S, fS, historia


def main(evaluador=None):
    ev = evaluador or Evaluador()
    cfg = ev.cfg
    ch = cfg["optimizacion"]["hill_climbing"]
    rng = np.random.default_rng(cfg["semilla"])
    inicio = config_inicial(cfg)
    print("=== Hill Climbing ===")
    S1, f1, h1 = hill_climbing(ev, inicio, ch["iteraciones"], rng)
    print("=== Hill Climbing (máxima pendiente) ===")
    S2, f2, h2 = steepest_ascent(ev, inicio, ch["iteraciones_steepest"], ch["vecinos_steepest"], rng)
    res = ruta(cfg, "resultados")
    guardar_json({"hill_climbing": {"mejor": S1, "f1": f1, "historia": h1},
                  "steepest_ascent": {"mejor": S2, "f1": f2, "historia": h2}}, res / "hill_climbing.json")
    return {"HC": (S1, f1, h1), "HC-steepest": (S2, f2, h2)}


if __name__ == "__main__":
    main()
