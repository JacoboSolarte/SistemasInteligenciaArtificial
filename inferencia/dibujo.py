"""Funciones de dibujo con OpenCV (las imágenes se manejan en BGR, como las lee cv2)."""
import cv2
import numpy as np

VERDE = (60, 200, 60)
ROJO = (40, 40, 230)
AMARILLO = (0, 215, 255)
AZUL = (255, 160, 0)


def dibujar_espacios(imagen, espacios, por_espacio, mostrar_conf=None):
    """Polígonos semitransparentes verdes (libre) o rojos (ocupado) con su ID y su confianza.
    Con muchos espacios (>50) la confianza solo se escribe en los dudosos para que el texto no se amontone."""
    if mostrar_conf is None:
        mostrar_conf = len(espacios) <= 50
    salida = imagen.copy()
    capa = imagen.copy()
    escala = max(imagen.shape[1] / 1280, 0.4)
    for e, r in zip(espacios, por_espacio):
        pts = np.asarray(e["poligono"], dtype=np.int32)
        color = ROJO if r["estado"] == "ocupado" else VERDE
        cv2.fillPoly(capa, [pts], color)
    cv2.addWeighted(capa, 0.30, salida, 0.70, 0, salida)
    for e, r in zip(espacios, por_espacio):
        pts = np.asarray(e["poligono"], dtype=np.int32)
        color = ROJO if r["estado"] == "ocupado" else VERDE
        borde = AMARILLO if r.get("dudoso") else color
        cv2.polylines(salida, [pts], True, borde, max(1, int(2 * escala)))
        cx, cy = pts.mean(axis=0).astype(int)
        texto = f"{e['id']}" + (f" {r['confianza']:.2f}" if mostrar_conf or r.get("dudoso") else "")
        fs = 0.38 * escala
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(salida, (cx - tw // 2 - 1, cy - th - 1), (cx + tw // 2 + 1, cy + 2), (0, 0, 0), -1)
        cv2.putText(salida, texto, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return salida


def dibujar_cajas(imagen, cajas, puntajes=None, color=AZUL):
    salida = imagen.copy()
    for i, c in enumerate(cajas):
        x1, y1, x2, y2 = [int(v) for v in c]
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, 1)
        if puntajes is not None:
            cv2.putText(salida, f"{puntajes[i]:.2f}", (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return salida


def dibujar_panel(imagen, estado):
    """Panel superior izquierdo con libres, ocupados y porcentaje de ocupación."""
    salida = imagen.copy()
    escala = max(imagen.shape[1] / 1280, 0.5)
    lineas = [f"Libres: {estado['libres']}", f"Ocupados: {estado['ocupados']}",
              f"Ocupacion: {estado['porcentaje_ocupacion']:.1f}%"]
    alto = int(28 * escala)
    ancho = int(250 * escala)
    cv2.rectangle(salida, (8, 8), (8 + ancho, 14 + alto * len(lineas)), (0, 0, 0), -1)
    colores = [VERDE, ROJO, (255, 255, 255)]
    for i, (t, c) in enumerate(zip(lineas, colores)):
        cv2.putText(salida, t, (16, 8 + alto * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * escala, c, max(1, int(2 * escala)), cv2.LINE_AA)
    return salida
