"""Recorte de espacios: carga del JSON de espacios y transformación de perspectiva a 64x64.

La MISMA función `recortar_espacio` se usa para generar los recortes de entrenamiento de PKLot
y en el pipeline de inferencia, así la CNN ve en producción exactamente el tipo de imagen con
el que fue entrenada.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def cargar_espacios(ruta_json) -> list[dict]:
    """Lee config/espacios_<nombre>.json  ->  [{"id": "A1", "poligono": [[x,y],...]}, ...]"""
    with open(ruta_json, encoding="utf-8") as f:
        espacios = json.load(f)
    validar_espacios(espacios)
    return espacios


def validar_espacios(espacios) -> None:
    if not isinstance(espacios, list) or not espacios:
        raise ValueError("El JSON de espacios debe ser una lista no vacía")
    ids = set()
    for e in espacios:
        if "id" not in e or "poligono" not in e:
            raise ValueError(f"Espacio sin 'id' o 'poligono': {e}")
        if len(e["poligono"]) < 3:
            raise ValueError(f"El espacio {e['id']} necesita al menos 3 vértices")
        if e["id"] in ids:
            raise ValueError(f"ID repetido: {e['id']}")
        ids.add(e["id"])


def guardar_espacios(espacios, ruta_json) -> None:
    validar_espacios(espacios)
    Path(ruta_json).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(espacios, f, ensure_ascii=False, indent=1)


def cuatro_vertices(poligono) -> np.ndarray:
    """Reduce cualquier polígono a 4 vértices ordenados: sup-izq, sup-der, inf-der, inf-izq."""
    p = np.asarray(poligono, dtype=np.float32)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    suma, resta = p.sum(axis=1), np.diff(p, axis=1).ravel()   # resta = y - x
    ordenado = np.array([p[np.argmin(suma)], p[np.argmin(resta)], p[np.argmax(suma)], p[np.argmax(resta)]])
    if len({tuple(v) for v in ordenado.tolist()}) < 4:    # caso degenerado (rotación de 45°)
        c = p.mean(axis=0)
        ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
        ordenado = p[np.argsort(ang)]                       # orden horario empezando arriba-izquierda
        k = np.argmin(ordenado.sum(axis=1))
        ordenado = np.roll(ordenado, -k, axis=0)
    return ordenado.astype(np.float32)


def recortar_espacio(imagen: np.ndarray, poligono, tam: int = 64) -> np.ndarray:
    """Aplica una transformación de perspectiva que lleva el cuadrilátero del espacio a un cuadrado tam x tam."""
    origen = cuatro_vertices(poligono)
    destino = np.array([[0, 0], [tam - 1, 0], [tam - 1, tam - 1], [0, tam - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(origen, destino)
    return cv2.warpPerspective(imagen, M, (tam, tam), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def recortar_todos(imagen: np.ndarray, espacios, tam: int = 64) -> np.ndarray:
    """Devuelve un arreglo [N, tam, tam, 3] con el recorte de cada espacio."""
    return np.stack([recortar_espacio(imagen, e["poligono"], tam) for e in espacios])
