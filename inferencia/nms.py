"""IoU y Non-Maximum Suppression implementados a mano (sin funciones de NMS de librerías)."""
import numpy as np


def iou_cajas(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """IoU entre una caja [x1,y1,x2,y2] y un arreglo de cajas [M,4]."""
    caja, cajas = np.asarray(caja, dtype=np.float64), np.asarray(cajas, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(caja[0], cajas[:, 0])
    y1 = np.maximum(caja[1], cajas[:, 1])
    x2 = np.minimum(caja[2], cajas[:, 2])
    y2 = np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(cajas: np.ndarray, puntajes: np.ndarray, umbral_iou: float = 0.45) -> list[int]:
    """Algoritmo NMS:
    1. Ordenar las cajas por puntaje de mayor a menor.
    2. Tomar la mejor, conservarla y eliminar todas las que se solapen con ella más que `umbral_iou`.
    3. Repetir con las que quedan.
    Devuelve los índices conservados."""
    cajas, puntajes = np.asarray(cajas).reshape(-1, 4), np.asarray(puntajes).ravel()
    orden = np.argsort(-puntajes)
    conservar = []
    while orden.size > 0:
        mejor = orden[0]
        conservar.append(int(mejor))
        if orden.size == 1:
            break
        ious = iou_cajas(cajas[mejor], cajas[orden[1:]])
        orden = orden[1:][ious <= umbral_iou]
    return conservar


def postprocesar(puntajes, cajas, umbral_conf=0.4, umbral_nms=0.45):
    """Umbral de confianza + NMS para una imagen. Entradas numpy [M], [M,4]."""
    mascara = puntajes >= umbral_conf
    p, c = puntajes[mascara], cajas[mascara]
    if len(p) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    idx = nms(c, p, umbral_nms)
    return c[idx], p[idx]
