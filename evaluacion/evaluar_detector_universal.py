"""
Evaluación del Modelo 4 (DetectorUniversal) en el conjunto de PRUEBA de cada fuente:
AP@0.5 por (fuente, clase), precisión/recall/F1 al umbral de operación, y ejemplos visuales.
También evalúa el CONTEO (lo que ve el usuario): error en número de libres / ocupados por imagen.

    python evaluacion/evaluar_detector_universal.py
    python evaluacion/evaluar_detector_universal.py --imagenes "C:/fotos/*.jpg"   # ejemplos sin etiquetas
"""
import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluacion.metricas_detector import average_precision, precision_recall_en_umbral  # noqa: E402
from entrenamiento.entrenar_detector_universal import predecir_filas  # noqa: E402
from inferencia.automatico import AnalizadorAutomatico  # noqa: E402
from modelos.detector_universal import CLASES  # noqa: E402
from utilidades.comun import cargar_config, guardar_json, leer_json, ruta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenes", default=None, help="patrón glob de imágenes propias (sin etiquetas) para ejemplos")
    a = ap.parse_args()
    cfg = cargar_config()
    cu = cfg["detector_universal"]
    raiz = ruta(cfg, "datos") / "procesados" / "universal"
    res, pesos = ruta(cfg, "resultados"), ruta(cfg, "pesos")
    an = AnalizadorAutomatico(pesos / "detector_universal.pth", cfg)
    dev = an.dispositivo
    filas = [f for f in leer_json(raiz / "anotaciones.json") if f["particion"] == "test"]
    preds = predecir_filas(an.modelo, filas, raiz, dev, cu["lado_inferencia"], cu["umbral_nms"], detector=an.detectar)

    por_fuente = defaultdict(list)
    for f, p in zip(filas, preds):
        por_fuente[f["fuente"]].append((f, p))
    filas_res = []
    for fuente, items in sorted(por_fuente.items()):
        etiq = items[0][0]["etiquetadas"]
        for c in range(3):
            if not etiq[c]:
                continue
            gt = [np.array([b[:4] for b in f["cajas"] if b[4] == c], np.float32).reshape(-1, 4) for f, _ in items]
            pr = [p[c] for _, p in items]
            if sum(len(g) for g in gt) == 0:
                continue
            ap50, _, _ = average_precision(pr, gt)
            op = [(cc[pp >= cu["umbral_confianza"]], pp[pp >= cu["umbral_confianza"]]) for cc, pp in pr]
            P, R, F1 = precision_recall_en_umbral(op, gt)
            filas_res.append({"fuente": fuente, "clase": CLASES[c], "n_imagenes": len(items),
                              "n_objetos": int(sum(len(g) for g in gt)), "AP50": ap50, "precision": P, "recall": R, "f1": F1})
    df = pd.DataFrame(filas_res)
    df.to_csv(res / "metricas_detector_universal.csv", index=False)
    print(df.round(3).to_string(index=False))
    resumen = {"mAP50_medio": float(df["AP50"].mean()),
               "mAP50_por_clase": df.groupby("clase")["AP50"].mean().round(4).to_dict()}

    # Conteo por imagen con el pipeline completo (reglas de postproceso incluidas)
    conteo = []
    for f in filas:
        if not f["etiquetadas"][0]:
            continue
        img = cv2.imread(str(raiz / f["archivo"]))
        _, est = an.analizar_imagen(img, dibujar=False)
        real_l = sum(b[4] == 0 for b in f["cajas"])
        real_o = sum(b[4] == 1 for b in f["cajas"])
        conteo.append({"fuente": f["fuente"], "libres_real": real_l, "libres_pred": est["libres"],
                       "ocupados_real": real_o, "ocupados_pred": est["ocupados"]})
    dc = pd.DataFrame(conteo)
    dc["error_libres"] = (dc["libres_pred"] - dc["libres_real"]).abs()
    dc["error_ocupados"] = (dc["ocupados_pred"] - dc["ocupados_real"]).abs()
    dc["ocup_real_%"] = 100 * dc["ocupados_real"] / (dc["libres_real"] + dc["ocupados_real"]).clip(lower=1)
    dc["ocup_pred_%"] = 100 * dc["ocupados_pred"] / (dc["libres_pred"] + dc["ocupados_pred"]).clip(lower=1)
    dc["error_ocup_pp"] = (dc["ocup_pred_%"] - dc["ocup_real_%"]).abs()
    tabla_conteo = dc.groupby("fuente")[["libres_real", "error_libres", "ocupados_real", "error_ocupados", "error_ocup_pp"]].mean()
    tabla_conteo.round(2).to_csv(res / "conteo_detector_universal.csv")
    print("\nError medio de conteo por imagen (MAE) y error del % de ocupación (puntos porcentuales):")
    print(tabla_conteo.round(2).to_string())
    resumen["error_ocupacion_pp_medio"] = float(dc["error_ocup_pp"].mean())
    guardar_json(resumen, res / "metricas_detector_universal.json")

    # Ejemplos visuales: 1 imagen de prueba por fuente
    fuentes = sorted(por_fuente)
    fig, axs = plt.subplots(2, (len(fuentes) + 1) // 2, figsize=(6 * ((len(fuentes) + 1) // 2), 9))
    for ax, fuente in zip(axs.ravel(), fuentes):
        f = por_fuente[fuente][len(por_fuente[fuente]) // 2][0]
        img = cv2.imread(str(raiz / f["archivo"]))
        anot, est = an.analizar_imagen(img)
        ax.imshow(cv2.cvtColor(anot, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{fuente}: {est['libres']} libres, {est['ocupados']} ocupados, {est['vehiculos_detectados']} vehículos",
                     fontsize=9)
        ax.axis("off")
    for ax in axs.ravel()[len(fuentes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(res / "detector_universal_ejemplos.png", dpi=90)
    plt.close(fig)

    if a.imagenes:
        destino = res / "automatico_propias"
        destino.mkdir(exist_ok=True)
        for p in sorted(glob.glob(a.imagenes)):
            img = cv2.imread(p)
            if img is None:
                continue
            anot, est = an.analizar_imagen(img)
            cv2.imwrite(str(destino / (Path(p).stem + "_auto.jpg")), anot)
            print(f"{Path(p).name}: {est['libres']} libres, {est['ocupados']} ocupados, "
                  f"{est['vehiculos_detectados']} vehículos ({est['vehiculos_fuera_de_espacio']} fuera de espacio)")


if __name__ == "__main__":
    main()
