"""Funciones comunes: configuración, semillas, dispositivo y rutas."""
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]


def _fusionar(base: dict, extra: dict) -> dict:
    """Fusiona diccionarios de forma recursiva (extra sobrescribe a base)."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _fusionar(base[k], v)
        else:
            base[k] = v
    return base


def cargar_config(ruta: str | Path | None = None) -> dict:
    """Lee config/config.yaml y, si existe, config/rutas_locales.yaml (rutas propias de cada PC)."""
    ruta = Path(ruta) if ruta else RAIZ / "config" / "config.yaml"
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = RAIZ / "config" / "rutas_locales.yaml"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            _fusionar(cfg, yaml.safe_load(f) or {})
    # Variable de entorno opcional (útil en Colab)
    if os.environ.get("PARQUEADERO_DATOS"):
        cfg["rutas"]["datos"] = os.environ["PARQUEADERO_DATOS"]
    return cfg


def ruta(cfg: dict, clave: str) -> Path:
    """Devuelve una ruta absoluta de la sección 'rutas' (datos, pesos, resultados)."""
    p = Path(cfg["rutas"][clave])
    p = p if p.is_absolute() else RAIZ / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def fijar_semillas(semilla: int = 42) -> None:
    """Fija las semillas de random, numpy y torch para que los resultados sean reproducibles."""
    import torch
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def obtener_dispositivo():
    """'cuda' si hay GPU disponible (p. ej. la T4 de Colab); si no, 'cpu'."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guardar_json(obj, ruta_archivo: str | Path) -> None:
    Path(ruta_archivo).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def leer_json(ruta_archivo: str | Path):
    with open(ruta_archivo, encoding="utf-8") as f:
        return json.load(f)


def contar_parametros(modelo) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
