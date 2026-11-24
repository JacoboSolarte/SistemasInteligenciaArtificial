"""
Regla de fusión CNN + detector.

Para cada espacio la CNN entrega p = P(ocupado) y su confianza c = max(p, 1 - p).

    1. Si c >= umbral_cnn            -> se usa la decisión de la CNN (fuente = "cnn").
    2. Si c <  umbral_cnn (la CNN duda) y el detector está activo:
           IoU_max = máximo IoU entre el polígono del espacio y las cajas de vehículos detectadas
           ocupado si IoU_max >= umbral_iou, libre en caso contrario      (fuente = "detector").
    3. Si la CNN duda y no hay detector -> se usa la CNN de todos modos (fuente = "cnn_dudosa").

Además se marca `discrepancia = True` cuando la CNN y el detector no coinciden; el asistente lo reporta.
"""
import cv2
import numpy as np


def iou_poligono_caja(poligono, caja) -> float:
    """IoU entre un polígono convexo (espacio) y una caja [x1,y1,x2,y2] usando cv2.intersectConvexConvex."""
    p = cv2.convexHull(np.asarray(poligono, dtype=np.float32))
    x1, y1, x2, y2 = [float(v) for v in caja]
    r = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(p, r)
    area_p = cv2.contourArea(p)
    area_r = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = area_p + area_r - inter
    return float(inter / union) if union > 0 else 0.0


def fusionar(espacios, prob_ocupado, cajas_detector=None, umbral_cnn=0.8, umbral_iou=0.3):
    """Devuelve una lista de dicts por espacio con estado, confianza, fuente y discrepancia."""
    resultados = []
    for e, p in zip(espacios, prob_ocupado):
        p = float(p)
        conf_cnn = max(p, 1 - p)
        ocupado_cnn = p >= 0.5
        r = {"id": e["id"], "prob_ocupado": round(p, 4), "confianza": round(conf_cnn, 4),
             "estado": "ocupado" if ocupado_cnn else "libre", "fuente": "cnn", "discrepancia": False}
        if cajas_detector is not None:
            iou_max = max((iou_poligono_caja(e["poligono"], c) for c in cajas_detector), default=0.0)
            ocupado_det = iou_max >= umbral_iou
            r["iou_detector"] = round(iou_max, 3)
            r["discrepancia"] = bool(ocupado_det != ocupado_cnn)
            if conf_cnn < umbral_cnn:
                r["estado"] = "ocupado" if ocupado_det else "libre"
                r["fuente"] = "detector"
        elif conf_cnn < umbral_cnn:
            r["fuente"] = "cnn_dudosa"
        resultados.append(r)
    return resultados
