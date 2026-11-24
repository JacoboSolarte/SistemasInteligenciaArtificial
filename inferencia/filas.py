"""
Regla geométrica de FILAS para el modo automático (complementa al DetectorUniversal).

Motivación: el detector aprende la clase "vehículo" con muchos datos de dron (CARPK), pero los espacios
libres vistos desde arriba tienen pocos ejemplos etiquetados. Los carros parqueados forman filas, así que:

  1. Se separan los vehículos por orientación (más altos que anchos = parqueados "de frente" en una fila
     horizontal; más anchos que altos = fila vertical) y se agrupan en filas por su coordenada transversal.
  2. Una fila con >= 3 vehículos alineados es una fila de parqueo.
  3. Vehículo de una fila que no está dentro de ningún espacio detectado -> espacio OCUPADO (su caja
     ampliada un poco).
  4. Hueco entre dos vehículos consecutivos de la fila: si mide k veces el paso típico entre carros
     (2 <= k <= max_hueco+1), se agregan k-1 espacios LIBRES "inferidos" en el hueco (si no chocan con un
     vehículo ni con un espacio ya detectado).
  No se infieren espacios en los extremos de las filas ni en filas completamente vacías: ahí no hay
  evidencia suficiente (es una limitación conocida).
"""
import numpy as np

from inferencia.nms import iou_cajas


def _filas_por_eje(cajas, eje_largo):
    """Agrupa cajas en filas. eje_largo=0: carros lado a lado a lo largo de x (fila horizontal),
    se agrupan por y. eje_largo=1: fila vertical, se agrupan por x."""
    if len(cajas) == 0:
        return []
    c = (cajas[:, :2] + cajas[:, 2:]) / 2
    trans = 1 - eje_largo
    tam_trans = np.median(cajas[:, 2 + trans] - cajas[:, trans])
    tol = 0.5 * tam_trans
    orden = np.argsort(c[:, trans])
    filas, actual = [], [orden[0]]
    for i in orden[1:]:
        if abs(c[i, trans] - np.mean(c[actual, trans])) <= tol:
            actual.append(i)
        else:
            filas.append(actual)
            actual = [i]
    filas.append(actual)
    # dentro de una "banda" puede haber carros muy separados (dos bloques distintos): se ordena por el eje largo
    return [sorted(f, key=lambda j: c[j, eje_largo]) for f in filas]


def inferir_por_filas(espacios, vehiculos, min_fila=3, max_hueco=4, conf_inferida=0.5):
    """Modifica y devuelve `espacios` (lista de dicts del postproceso automático) agregando espacios
    ocupados (vehículos en fila sin espacio) y libres inferidos (huecos dentro de las filas)."""
    if len(vehiculos) < min_fila:
        return espacios
    V = np.array([v["caja"] for v in vehiculos], np.float32)
    E = np.array([e["caja"] for e in espacios], np.float32).reshape(-1, 4)
    ancho, alto = V[:, 2] - V[:, 0], V[:, 3] - V[:, 1]
    nuevos = []
    for eje_largo, mascara in ((0, alto >= ancho), (1, ancho > alto)):
        idx = np.where(mascara)[0]
        for fila in _filas_por_eje(V[idx], eje_largo):
            fila = [idx[j] for j in fila]
            if len(fila) < min_fila:
                continue
            cajas_f = V[fila]
            cen = (cajas_f[:, eje_largo] + cajas_f[:, 2 + eje_largo]) / 2
            lado = np.median(cajas_f[:, 2 + eje_largo] - cajas_f[:, eje_largo])        # ancho del carro en la fila
            prof = np.median(cajas_f[:, 3 - eje_largo] - cajas_f[:, 1 - eje_largo])    # largo del carro
            pos_trans = np.median((cajas_f[:, 1 - eje_largo] + cajas_f[:, 3 - eje_largo]) / 2)
            d = np.diff(cen)
            juntos = d[d < 1.8 * lado]
            paso = float(np.median(juntos)) if len(juntos) >= 2 else 1.3 * float(lado)
            paso = max(paso, 1.05 * float(lado))
            # 3. vehículos de la fila sin espacio detectado -> ocupado
            for j in fila:
                if vehiculos[j]["en_espacio"]:
                    continue
                x1, y1, x2, y2 = V[j]
                mx, my = 0.06 * (x2 - x1), 0.06 * (y2 - y1)
                nuevos.append({"caja": [float(x1 - mx), float(y1 - my), float(x2 + mx), float(y2 + my)],
                               "estado": "ocupado", "confianza": float(vehiculos[j]["confianza"]),
                               "corregido": False, "origen": "vehiculo_en_fila"})
                vehiculos[j]["en_espacio"] = True
            # 4. huecos entre vehículos consecutivos -> libres inferidos
            for a, b, gap in zip(cen[:-1], cen[1:], d):
                k = int(round(gap / paso))
                if k < 2 or k > max_hueco + 1:
                    continue
                for m in range(1, k):
                    centro = a + m * gap / k
                    caja = np.zeros(4, np.float32)
                    caja[eje_largo], caja[2 + eje_largo] = centro - 0.45 * paso, centro + 0.45 * paso
                    caja[1 - eje_largo], caja[3 - eje_largo] = pos_trans - prof / 2, pos_trans + prof / 2
                    if len(V) and iou_cajas(caja, V).max() > 0.15:
                        continue
                    if len(E) and iou_cajas(caja, E).max() > 0.3:
                        continue
                    if nuevos and iou_cajas(caja, np.array([n["caja"] for n in nuevos])).max() > 0.3:
                        continue
                    nuevos.append({"caja": caja.tolist(), "estado": "libre", "confianza": conf_inferida,
                                   "corregido": False, "origen": "inferido_por_filas"})
    return espacios + nuevos
