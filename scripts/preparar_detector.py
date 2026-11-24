"""
Prepara el dataset del Modelo 2 (MiniDetector) a partir de las imágenes completas de PKLot.

PKLot NO trae cajas de vehículos, pero sí el contorno de cada espacio y si está ocupado.
Etiqueta aproximada: para cada espacio OCUPADO se toma el rectángulo envolvente (cv2.boundingRect)
de su contorno como caja del vehículo, y se convierte a formato normalizado
    [clase, x_centro, y_centro, ancho, alto]   (clase 0 = vehículo, valores en [0,1])

Salida (en <datos>/procesados/detector/):
    imagenes.npy    uint8 [N, 416, 416, 3] (RGB, imagen estirada a 416x416)
    etiquetas.json  lista de N listas de cajas [0, cx, cy, w, h]
    metadatos.csv   idx, imagen, parqueadero, clima, n_vehiculos, particion
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas.pklot_a_json import leer_xml_pklot, listar_pklot  # noqa: E402
from utilidades.comun import cargar_config, fijar_semillas, guardar_json, leer_json, ruta  # noqa: E402


def cajas_desde_xml(ruta_xml, ancho, alto):
    """Cajas normalizadas [0, cx, cy, w, h] a partir de los espacios ocupados del XML."""
    cajas = []
    for e in leer_xml_pklot(ruta_xml):
        if e["ocupado"] != 1:
            continue
        x, y, w, h = cv2.boundingRect(np.asarray(e["poligono"], dtype=np.int32))
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, ancho), min(y + h, alto)
        if x2 <= x1 or y2 <= y1:
            continue
        cajas.append([0, (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto, (x2 - x1) / ancho, (y2 - y1) / alto])
    return cajas


def main():
    cfg = cargar_config()
    fijar_semillas(cfg["semilla"])
    tam = cfg["datos"]["detector"]["tam_imagen"]
    por_grupo = cfg["datos"]["detector"]["imagenes_por_grupo"]
    crudos = ruta(cfg, "datos") / "crudos"
    salida = ruta(cfg, "datos") / "procesados" / "detector"
    salida.mkdir(parents=True, exist_ok=True)
    archivo_res = ruta(cfg, "datos") / "procesados" / "dias_reservados.json"
    reservados = leer_json(archivo_res) if archivo_res.exists() else {}

    imagenes = listar_pklot(crudos / "pklot")
    rng = random.Random(cfg["semilla"] + 1)
    elegidas = []
    for parq in ("UFPR04", "UFPR05", "PUCPR"):
        for clima in ("soleado", "nublado", "lluvioso"):
            g = [f for f in imagenes if f["parqueadero"] == parq and f["clima"] == clima and reservados.get(parq) != f["fecha"]]
            rng.shuffle(g)
            elegidas += g[:por_grupo]
    print(f"{len(elegidas)} imágenes elegidas para el detector")

    arr = np.lib.format.open_memmap(salida / "imagenes.npy", mode="w+", dtype=np.uint8, shape=(len(elegidas), tam, tam, 3))

    def procesar(i_info):
        i, info = i_info
        img = cv2.imread(info["imagen"])
        if img is None:
            return i, None
        alto, ancho = img.shape[:2]
        arr[i] = cv2.cvtColor(cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return i, cajas_desde_xml(info["xml"], ancho, alto)

    etiquetas = [None] * len(elegidas)
    with ThreadPoolExecutor(8) as ex:
        for i, cajas in ex.map(procesar, enumerate(elegidas)):
            etiquetas[i] = cajas
    arr.flush()

    meta = pd.DataFrame([{"idx": i, "imagen": f["imagen"], "parqueadero": f["parqueadero"], "clima": f["clima"],
                          "n_vehiculos": len(etiquetas[i] or []), "valido": etiquetas[i] is not None}
                         for i, f in enumerate(elegidas)])
    etiquetas = [c or [] for c in etiquetas]
    estrato = meta["parqueadero"] + "_" + meta["clima"]
    p_tr, p_va, p_te = cfg["datos"]["proporciones"]
    tr, resto = train_test_split(meta.index, test_size=p_va + p_te, stratify=estrato, random_state=cfg["semilla"])
    va, te = train_test_split(resto, test_size=p_te / (p_va + p_te), stratify=estrato[resto], random_state=cfg["semilla"])
    meta["particion"] = ""
    meta.loc[tr, "particion"], meta.loc[va, "particion"], meta.loc[te, "particion"] = "train", "val", "test"
    meta.to_csv(salida / "metadatos.csv", index=False)
    guardar_json(etiquetas, salida / "etiquetas.json")
    print(meta.groupby(["parqueadero", "particion"]).size().unstack())
    print(f"Vehículos (cajas) totales: {meta['n_vehiculos'].sum()}  |  promedio por imagen: {meta['n_vehiculos'].mean():.1f}")


if __name__ == "__main__":
    main()
