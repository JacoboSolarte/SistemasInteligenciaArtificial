"""
Espacio de búsqueda de hiperparámetros de la CNN y función de fitness (F1 de validación).

Cada configuración se representa como un vector u ∈ [0,1]^4 (espacio normalizado) que se decodifica así:
    u0 -> learning rate  = 10^(-4 + 2·u0)          (1e-4 ... 1e-2, escala logarítmica)
    u1 -> dropout        = 0.1 + 0.6·u1            (0.1 ... 0.7)
    u2 -> filtros base   ∈ {8, 12, 16, 24, 32}      (los 3 bloques usan [b, 2b, 4b])
    u3 -> tamaño de lote ∈ {32, 64, 128, 256}
Trabajar en [0,1] permite usar la misma mutación gaussiana para todos los genes.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrenamiento.entrenar_cnn import cargar_datos_cnn, entrenar_cnn, metricas, particiones, predecir_probs  # noqa: E402
from utilidades.comun import cargar_config, obtener_dispositivo  # noqa: E402

FILTROS = [8, 12, 16, 24, 32]
LOTES = [32, 64, 128, 256]


def decodificar(u) -> dict:
    u = np.clip(np.asarray(u, dtype=float), 0, 1)
    b = FILTROS[int(round(u[2] * (len(FILTROS) - 1)))]
    return {"lr": float(f"{10 ** (-4 + 2 * u[0]):.2e}"), "dropout": round(0.1 + 0.6 * u[1], 3),
            "filtros": [b, 2 * b, 4 * b], "tam_lote": LOTES[int(round(u[3] * (len(LOTES) - 1)))]}


def codificar(c: dict) -> np.ndarray:
    return np.array([(math.log10(c["lr"]) + 4) / 2, (c["dropout"] - 0.1) / 0.6,
                     FILTROS.index(c["filtros"][0]) / (len(FILTROS) - 1),
                     LOTES.index(c["tam_lote"]) / (len(LOTES) - 1)]).clip(0, 1)


def clave(c: dict) -> str:
    return f"lr={c['lr']:.1e}|do={c['dropout']:.2f}|f={c['filtros'][0]}|b={c['tam_lote']}"


class Evaluador:
    """Entrena la CNN pocas épocas sobre un subconjunto y devuelve el F1 de validación (fitness).
    Guarda en caché las configuraciones ya evaluadas y lleva el registro de todas las evaluaciones."""

    def __init__(self, cfg=None):
        self.cfg = cfg or cargar_config()
        co = self.cfg["optimizacion"]
        self.dev = obtener_dispositivo()
        self.X, meta = cargar_datos_cnn(self.cfg)
        self.y = np.zeros(len(self.X), dtype=np.int64)
        self.y[meta.idx.to_numpy()] = meta.etiqueta.to_numpy()
        part = particiones(meta, "general", self.cfg["semilla"])
        rng = np.random.default_rng(self.cfg["semilla"])
        self.idx_tr = rng.choice(part["train"], min(co["muestras_train"], len(part["train"])), replace=False)
        self.idx_va = rng.choice(part["val"], min(co["muestras_val"], len(part["val"])), replace=False)
        self.epocas = co["epocas_por_evaluacion"]
        self.cache, self.registro = {}, []

    def __call__(self, c: dict, metodo: str = "") -> float:
        k = clave(c)
        if k in self.cache:
            f1 = self.cache[k]
        else:
            t0 = time.time()
            conf = dict(self.cfg["cnn"], **c, optimizador="adam", paciencia=99)
            modelo, _ = entrenar_cnn(conf, self.X, self.y, self.idx_tr, self.idx_va, self.dev,
                                     epocas=self.epocas, semilla=self.cfg["semilla"], verbose=False)
            idx, prob = predecir_probs(modelo, self.X, self.idx_va, self.dev)
            f1 = float(metricas(self.y[idx], prob)["f1"])
            self.cache[k] = f1
            print(f"    [{metodo}] {k}  ->  F1 val = {f1:.4f}  ({time.time() - t0:.1f}s)")
        self.registro.append({"metodo": metodo, "evaluacion": len(self.registro) + 1, "config": clave(c), "f1": f1})
        return f1


def config_inicial(cfg) -> dict:
    c = cfg["cnn"]
    return decodificar(codificar({"lr": c["lr"], "dropout": c["dropout"], "filtros": c["filtros"], "tam_lote": c["tam_lote"]}))
